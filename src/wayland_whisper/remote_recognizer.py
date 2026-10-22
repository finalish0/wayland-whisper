"""Remote Whisper transcription against the AN gateway.

The contract is the same as the local engines: ``transcribe(pcm: bytes) ->
str``. Capture, VAD, phrase spacing, suspend/close, LED and Enter ordering
stay in the daemon - this module only replaces the decode.

The remote side is the llama-swap gateway on AN
(``http://192.168.188.51:8888``), which serves the whisper.cpp child
``whisper-large-v3-turbo`` behind the OpenAI-compatible
``/v1/audio/transcriptions`` route. The gateway fixes the decode profile
(German, json, greedy); the client sends only the audio.

The PCM is 16 kHz mono signed 16 bit, which is what the capture produces and
what the whisper server expects. It is wrapped in a WAV container here
because the route is a multipart file upload.
"""

from __future__ import annotations

import io
import os
import struct
import urllib.error
import urllib.request
import uuid

SAMPLE_RATE = 16000


def pcm_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw signed 16 bit mono PCM in a WAV container."""
    buffer = io.BytesIO()
    data_size = len(pcm)
    buffer.write(b"RIFF")
    buffer.write(struct.pack("<I", 36 + data_size))
    buffer.write(b"WAVEfmt ")
    buffer.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate,
                             sample_rate * 2, 2, 16))
    buffer.write(b"data")
    buffer.write(struct.pack("<I", data_size))
    buffer.write(pcm)
    return buffer.getvalue()


class RemoteWhisperRecognizer:
    """Transcribe locally captured PCM on the AN GPU.

    Unlike the local engines this one holds no model: every phrase is a
    request. The measured latency on AN is about 0.5 s warm for a short
    phrase (2.1 s for the first, cold phrase while the child starts).
    """

    def __init__(
        self,
        url: str,
        model: str = "whisper-large-v3-turbo",
        language: str = "de",
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._url = url
        self._model = model
        self._language = language
        self._api_key = api_key or self._key_from_environment()
        self._timeout = timeout

    @staticmethod
    def _key_from_environment() -> str:
        """The bearer token, from the variable or from the file it names.

        The file keeps the token out of the unit file and out of
        `systemctl show`; the variable is the fallback for manual runs.
        """
        key = os.environ.get("WHISPER_REMOTE_API_KEY", "")
        if key:
            return key.strip()
        path = os.environ.get("WHISPER_REMOTE_API_KEY_FILE", "")
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                return handle.read().strip()
        return ""

    def transcribe(self, pcm: bytes) -> str:
        if not pcm:
            return ""
        body = self._multipart(pcm_to_wav(pcm))
        request = urllib.request.Request(
            self._url,
            data=body,
            headers={
                "Content-Type": "multipart/form-data; boundary=%s" % _BOUNDARY,
                **({"Authorization": "Bearer %s" % self._api_key}
                   if self._api_key else {}),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read()[:200].decode("utf-8", "replace")
            raise RuntimeError(
                "remote transcription failed: HTTP %s %s" % (error.code, detail)
            ) from error
        except Exception as error:
            raise RuntimeError(
                "remote transcription failed: %s" % type(error).__name__
            ) from error
        import json

        try:
            text = json.loads(payload).get("text", "")
        except ValueError as error:
            raise RuntimeError("remote transcription returned no JSON") from error
        return str(text).strip()

    def _multipart(self, wav: bytes) -> bytes:
        parts = [
            b"--%s\r\n" % _BOUNDARY.encode(),
            b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n',
            b"Content-Type: audio/wav\r\n\r\n",
            wav,
            b"\r\n",
            b"--%s\r\n" % _BOUNDARY.encode(),
            b'Content-Disposition: form-data; name="model"\r\n\r\n',
            self._model.encode(),
            b"\r\n",
            b"--%s--\r\n" % _BOUNDARY.encode(),
        ]
        return b"".join(parts)


_BOUNDARY = "waylandwhisper" + uuid.uuid4().hex[:16]
