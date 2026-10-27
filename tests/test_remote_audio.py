"""WAV inside the LAN, MP3 outside it - and never a lost phrase.

The rule exists for one reason: a phrase dictated from a hotspot uploads
its own size again as raw PCM (32 kB/s). Inside the home network that is
free, so the raw path stays. These tests pin the decision, the encoder
fallback and the server-side rejection path, because a dictation that
silently drops a phrase is the failure mode this client must not have.
"""

import importlib
import io
import json
import pathlib
import struct
import sys
import unittest
import urllib.error

ROOT = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))

remote_module = importlib.import_module("wayland_whisper.remote_recognizer")
audio_module = remote_module  # Policy und Encoder leben im Recognizer-Modul

LAN_SOURCE = "192.168.188.41"
TUNNEL_SOURCE = "10.8.0.3"
GATEWAY = "192.168.188.51"


def pcm(seconds=1.0):
    """Silence is enough: the payload logic never looks at the samples."""
    return b"\x00\x00" * int(16000 * seconds)


class SubnetTests(unittest.TestCase):
    def test_the_laptop_in_the_lan_is_direct(self):
        self.assertTrue(audio_module.same_subnet(LAN_SOURCE, GATEWAY))

    def test_the_tunnel_address_is_not_direct(self):
        self.assertFalse(audio_module.same_subnet(TUNNEL_SOURCE, GATEWAY))

    def test_garbage_is_not_direct(self):
        self.assertFalse(audio_module.same_subnet("", GATEWAY))
        self.assertFalse(audio_module.same_subnet("nonsense", GATEWAY))


class PolicyTests(unittest.TestCase):
    def policy(self, mode="auto", source=LAN_SOURCE, **kwargs):
        calls = []

        def probe(host, port):
            calls.append((host, port))
            return source

        instance = audio_module.AudioPolicy(mode, probe=probe, **kwargs)
        instance.calls = calls
        return instance

    def test_auto_in_the_lan_sends_wav(self):
        policy = self.policy()
        payload, filename, content_type, label = policy.payload(pcm(), GATEWAY, 8888)
        self.assertEqual(label, "pcm")
        self.assertEqual(filename, "audio.wav")
        self.assertEqual(content_type, "audio/wav")
        self.assertEqual(payload[:4], b"RIFF")

    def test_auto_through_the_tunnel_sends_mp3(self):
        policy = self.policy(source=TUNNEL_SOURCE)
        payload, filename, content_type, label = policy.payload(pcm(), GATEWAY, 8888)
        self.assertEqual(label, "mp3")
        self.assertEqual(filename, "audio.mp3")
        self.assertEqual(content_type, "audio/mpeg")
        self.assertNotEqual(payload[:4], b"RIFF")
        self.assertLess(len(payload), len(pcm()) + 44)

    def test_pcm_mode_stays_raw_even_through_the_tunnel(self):
        policy = self.policy(mode="pcm", source=TUNNEL_SOURCE)
        payload, _, _, label = policy.payload(pcm(), GATEWAY, 8888)
        self.assertEqual(label, "pcm")
        self.assertEqual(payload[:4], b"RIFF")

    def test_mp3_mode_compresses_even_in_the_lan(self):
        policy = self.policy(mode="mp3")
        _, _, _, label = policy.payload(pcm(), GATEWAY, 8888)
        self.assertEqual(label, "mp3")

    def test_an_unknown_route_falls_back_to_raw(self):
        policy = self.policy(source=None)
        _, _, _, label = policy.payload(pcm(), GATEWAY, 8888)
        self.assertEqual(label, "pcm")

    def test_the_route_is_probed_once_per_cache_window(self):
        policy = self.policy(source=TUNNEL_SOURCE, cache_seconds=30.0)
        for _ in range(3):
            policy.payload(pcm(0.1), GATEWAY, 8888)
        self.assertEqual(len(policy.calls), 1)

    def test_a_broken_encoder_never_loses_the_phrase(self):
        policy = self.policy(source=TUNNEL_SOURCE)
        original = audio_module.encode_mp3

        def explode(pcm_bytes, bit_rate=None):
            raise audio_module.EncoderUnavailable("kein Encoder")

        audio_module.encode_mp3 = explode
        try:
            payload, filename, _, label = policy.payload(pcm(), GATEWAY, 8888)
        finally:
            audio_module.encode_mp3 = original
        self.assertEqual(label, "pcm-fallback")
        self.assertEqual(filename, "audio.wav")
        self.assertEqual(payload[:4], b"RIFF")


class RecognizerUploadTests(unittest.TestCase):
    def setUp(self):
        self.requests = []

    def recognizer(self, statuses=(200,), audio="mp3", probe=None):
        outer = self

        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload

            def read(self):
                return self._payload

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class FakeUrlopen:
            def __call__(self, request, timeout=None):
                outer.requests.append(request.data)
                status = statuses[min(len(outer.requests) - 1, len(statuses) - 1)]
                if status != 200:
                    raise urllib.error.HTTPError(
                        "http://x", status, "Bad Request", {},
                        io.BytesIO(b"Invalid request"))
                return FakeResponse(json.dumps({"text": "ok"}).encode())

        remote_module.urllib.request.urlopen = FakeUrlopen()
        return remote_module.RemoteWhisperRecognizer(
            url=f"http://{GATEWAY}:8888/v1/audio/transcriptions",
            model="whisper-large-v3-turbo",
            api_key="dummy",
            audio=audio,
            probe=probe or (lambda host, port: TUNNEL_SOURCE),
        )

    def test_mp3_goes_out_as_an_mp3_file(self):
        recognizer = self.recognizer()
        self.assertEqual(recognizer.transcribe(pcm()), "ok")
        self.assertIn(b'filename="audio.mp3"', self.requests[0])
        self.assertIn(b"Content-Type: audio/mpeg", self.requests[0])
        self.assertEqual(recognizer.last_audio(), "mp3")

    def test_an_auto_route_in_the_lan_stays_wav(self):
        recognizer = self.recognizer(audio="auto", probe=lambda host, port: LAN_SOURCE)
        recognizer.transcribe(pcm())
        self.assertIn(b"RIFF", self.requests[0])
        self.assertEqual(recognizer.last_audio(), "pcm")

    def test_a_400_on_mp3_is_retried_once_as_wav(self):
        recognizer = self.recognizer(statuses=(400, 200))
        self.assertEqual(recognizer.transcribe(pcm()), "ok")
        self.assertEqual(len(self.requests), 2)
        self.assertIn(b'filename="audio.mp3"', self.requests[0])
        self.assertIn(b"RIFF", self.requests[1])
        self.assertEqual(recognizer.last_audio(), "pcm-retry")

    def test_other_errors_still_raise(self):
        recognizer = self.recognizer(statuses=(502,))
        with self.assertRaises(RuntimeError):
            recognizer.transcribe(pcm())


class RealEncoderTests(unittest.TestCase):
    """The encoder is real here - PyAV ships with faster-whisper."""

    def setUp(self):
        try:
            import av  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("PyAV/numpy nicht installiert")

    def test_a_three_second_phrase_becomes_a_decodable_mp3(self):
        encoded = audio_module.encode_mp3(pcm(3.0))
        self.assertGreater(len(encoded), 1000)
        self.assertLess(len(encoded), len(pcm(3.0)) + 44)
        self.assertTrue(encoded[:3] == b"ID3" or encoded[0] == 0xFF)
        import av

        with av.open(io.BytesIO(encoded)) as container:
            stream = container.streams.audio[0]
            seconds = float(stream.duration * stream.time_base) if stream.duration else 0.0
            frames = sum(1 for _ in container.decode(audio=0))
        self.assertGreater(frames, 0)
        self.assertAlmostEqual(seconds, 3.0, delta=0.2)

    def test_empty_input_is_refused(self):
        with self.assertRaises(audio_module.EncoderUnavailable):
            audio_module.encode_mp3(b"")


if __name__ == "__main__":
    unittest.main(verbosity=2)