"""The remote recognizer must behave like the local ones: bytes in, text out.

RED against a missing adapter: there is no way to point the dictation
client at the AN GPU. These tests pin the contract that the daemon relies
on - a PCM buffer in, a cleaned-up string out - and the failure behaviour,
because a dictation that silently returns an empty string would be
indistinguishable from silence.
"""

import importlib.util
import io
import json
import pathlib
import struct
import unittest
import urllib.error

ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "wayland_whisper"
MODULE = ROOT / "remote_recognizer.py"


def load():
    spec = importlib.util.spec_from_file_location("remote_recognizer", MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class PcmToWavTests(unittest.TestCase):
    def setUp(self):
        self.module = load()

    def test_the_wav_header_names_the_right_format(self):
        wav = self.module.pcm_to_wav(b"\x00\x00" * 100)
        self.assertEqual(wav[:4], b"RIFF")
        self.assertEqual(wav[8:12], b"WAVE")
        self.assertEqual(wav[12:16], b"fmt ")
        # PCM, one channel, 16 kHz, 16 bit
        self.assertEqual(struct.unpack("<H", wav[20:22])[0], 1)
        self.assertEqual(struct.unpack("<H", wav[22:24])[0], 1)
        self.assertEqual(struct.unpack("<I", wav[24:28])[0], 16000)
        self.assertEqual(struct.unpack("<H", wav[34:36])[0], 16)

    def test_the_audio_is_appended_unchanged(self):
        pcm = b"\x01\x02\x03\x04" * 50
        wav = self.module.pcm_to_wav(pcm)
        self.assertEqual(wav[44:], pcm)
        self.assertEqual(struct.unpack("<I", wav[40:44])[0], len(pcm))

    def test_the_data_chunk_size_matches_the_pcm(self):
        pcm = b"\x00\x01" * 320
        wav = self.module.pcm_to_wav(pcm)
        self.assertEqual(struct.unpack("<I", wav[4:8])[0], 36 + len(pcm))


class RemoteTranscribeTests(unittest.TestCase):
    def setUp(self):
        self.module = load()
        self.captured = {}

    def recognizer(self, response=None, error=None):
        module = self.module
        outer = self

        class FakeUrlopen:
            def __call__(self, request, timeout=None):
                outer.captured['request'] = request
                outer.captured['body'] = request.data
                outer.captured['headers'] = request.headers
                outer.captured['timeout'] = timeout
                if error is not None:
                    raise error
                return FakeResponse(response)

        module.urllib.request.urlopen = FakeUrlopen()
        return module.RemoteWhisperRecognizer(
            url="http://192.168.188.51:8888/v1/audio/transcriptions",
            model="whisper-large-v3-turbo")

    def test_a_transcript_comes_back_as_cleaned_text(self):
        recognizer = self.recognizer(response=json.dumps(
            {"text": "  Die Klang und Lichtshow.\n"}).encode())
        self.assertEqual(recognizer.transcribe(b"\x00\x00" * 100),
                         "Die Klang und Lichtshow.")

    def test_the_request_carries_the_audio_as_a_wav_file(self):
        recognizer = self.recognizer(response=json.dumps({"text": "ok"}).encode())
        recognizer.transcribe(b"\x01\x02" * 100)
        body = self.captured['body']
        self.assertIn(b"RIFF", body)
        self.assertIn(b'name="file"', body)
        self.assertIn(b'name="model"', body)
        self.assertIn(b"whisper-large-v3-turbo", body)
        content_type = self.captured['headers']['Content-type']
        self.assertIn("multipart/form-data", content_type)

    def test_empty_pcm_is_not_sent_anywhere(self):
        recognizer = self.recognizer(response=json.dumps({"text": "x"}).encode())
        self.assertEqual(recognizer.transcribe(b""), "")
        self.assertNotIn('request', self.captured)

    def test_a_server_error_raises_instead_of_returning_silence(self):
        error = urllib.error.HTTPError(
            "http://x", 502, "Bad Gateway", {}, io.BytesIO(b"backend operation failed"))
        recognizer = self.recognizer(error=error)
        with self.assertRaises(RuntimeError) as caught:
            recognizer.transcribe(b"\x00\x00" * 100)
        self.assertIn("502", str(caught.exception))

    def test_a_non_json_answer_raises(self):
        recognizer = self.recognizer(response=b"<html>no</html>")
        with self.assertRaises(RuntimeError):
            recognizer.transcribe(b"\x00\x00" * 100)

    def test_a_missing_text_key_yields_an_empty_string(self):
        recognizer = self.recognizer(response=json.dumps({"other": 1}).encode())
        self.assertEqual(recognizer.transcribe(b"\x00\x00" * 100), "")


if __name__ == '__main__':
    unittest.main(verbosity=2)
