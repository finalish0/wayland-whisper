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
import ipaddress
import os
import socket
import struct
import time
import urllib.error
import urllib.request
import uuid

SAMPLE_RATE = 16000
DEFAULT_MP3_BIT_RATE = 48000
ROUTE_CACHE_SECONDS = 5.0
PCM = "pcm"
MP3 = "mp3"


class EncoderUnavailable(RuntimeError):
    """The MP3 encoder cannot be used; the caller falls back to raw PCM."""


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


def route_source(host: str, port: int, timeout: float = 1.0) -> str | None:
    """Return the local address the kernel would use to reach ``host``.

    A connected UDP socket performs no handshake and sends no packet; it
    only resolves the route. ``None`` means the route is unknown.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(timeout)
            sock.connect((host, port))
            return str(sock.getsockname()[0])
    except (OSError, ValueError):
        return None


def same_subnet(local_address: str, host: str, prefix: int = 24) -> bool:
    """True when source and gateway share a /24, i.e. the traffic is LAN."""
    try:
        source = ipaddress.ip_address(local_address)
        target = ipaddress.ip_address(host)
    except ValueError:
        return False
    if source.version != target.version:
        return False
    return source in ipaddress.ip_network(f"{target}/{prefix}", strict=False)


def encode_mp3(pcm: bytes, bit_rate: int = DEFAULT_MP3_BIT_RATE) -> bytes:
    """Encode signed 16 bit mono PCM to MP3 in-process (no subprocess).

    PyAV arrives with faster-whisper, so this adds no dependency. Measured
    on enlil: 9-23 ms for 3-12 s phrases; a ffmpeg subprocess would pay
    34 ms of process start on top.
    """
    if not pcm:
        raise EncoderUnavailable("kein Audio")
    try:
        import av
        import numpy as np
    except ImportError as error:
        raise EncoderUnavailable(str(error)) from error
    try:
        samples = np.frombuffer(pcm, dtype=np.int16).reshape(1, -1)
        buffer = io.BytesIO()
        with av.open(buffer, mode="w", format="mp3") as container:
            stream = container.add_stream("libmp3lame", rate=SAMPLE_RATE)
            stream.layout = "mono"
            stream.bit_rate = bit_rate
            frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
            frame.sample_rate = SAMPLE_RATE
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
        encoded = buffer.getvalue()
    except Exception as error:  # noqa: BLE001 - any encoder failure means PCM
        raise EncoderUnavailable(str(error)) from error
    if not encoded:
        raise EncoderUnavailable("Encoder lieferte keine Daten")
    return encoded


class AudioPolicy:
    """Choose WAV or MP3 per phrase: raw inside the LAN, MP3 outside it.

    An unknown route counts as direct: raw audio is what every deployment
    accepts, so the safe answer never costs compatibility.
    """

    def __init__(self, mode: str = PCM, bit_rate: int = DEFAULT_MP3_BIT_RATE,
                 probe=route_source, clock=time.monotonic,
                 cache_seconds: float = ROUTE_CACHE_SECONDS) -> None:
        if mode not in ("auto", PCM, MP3):
            raise ValueError(f"unbekannter Audiomodus: {mode}")
        self._mode = mode
        self._bit_rate = bit_rate
        self._probe = probe
        self._clock = clock
        self._cache_seconds = cache_seconds
        self._cached: tuple[bool, float] | None = None
        self.last_label = ""

    def direct(self, host: str, port: int) -> bool:
        now = self._clock()
        if self._cached and now - self._cached[1] < self._cache_seconds:
            return self._cached[0]
        source = self._probe(host, port)
        is_direct = source is None or same_subnet(source, host)
        self._cached = (is_direct, now)
        return is_direct

    def wants_mp3(self, host: str, port: int) -> bool:
        if self._mode == MP3:
            return True
        if self._mode == PCM:
            return False
        return not self.direct(host, port)

    def payload(self, pcm: bytes, host: str, port: int) -> tuple[bytes, str, str, str]:
        """Return (bytes, filename, content type, label) for one phrase."""
        if pcm and self.wants_mp3(host, port):
            try:
                audio = encode_mp3(pcm, self._bit_rate)
            except EncoderUnavailable:
                self.last_label = "pcm-fallback"
                return pcm_to_wav(pcm), "audio.wav", "audio/wav", "pcm-fallback"
            self.last_label = MP3
            return audio, "audio.mp3", "audio/mpeg", MP3
        self.last_label = PCM
        return pcm_to_wav(pcm), "audio.wav", "audio/wav", PCM


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
        audio: str = "pcm",
        mp3_bit_rate: int = 48000,
        probe=None,
    ) -> None:
        self._url = url
        self._model = model
        self._language = language
        self._api_key = api_key or self._key_from_environment()
        self._timeout = timeout
        from urllib.parse import urlsplit

        parts = urlsplit(url)
        self._host = parts.hostname or ""
        self._port = parts.port or 80
        if probe is None:
            self._audio = AudioPolicy(audio, mp3_bit_rate)
        else:
            self._audio = AudioPolicy(audio, mp3_bit_rate, probe=probe)

    def last_audio(self) -> str:
        """Which payload the last phrase used: pcm, mp3 or pcm-fallback."""
        return self._audio.last_label

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
        audio, filename, content_type, label = self._audio.payload(
            pcm, self._host, self._port)
        body = self._multipart(audio, filename, content_type)
        try:
            payload = self._post(body)
        except urllib.error.HTTPError as error:
            if error.code == 400 and label == "mp3":
                # A deployment that only decodes WAV must not lose the phrase:
                # retry once with the untouched PCM, and say so in the journal.
                self._audio.last_label = "pcm-retry"
                payload = self._post(self._multipart(
                    pcm_to_wav(pcm), "audio.wav", "audio/wav"))
            else:
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

    def _post(self, body: bytes) -> bytes:
        request = urllib.request.Request(
            self._url,
            data=body,
            headers={
                "Content-Type": "multipart/form-data; boundary=%s" % _BOUNDARY,
                **(({"Authorization": "Bearer %s" % self._api_key}
                    if self._api_key else {})),
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            return response.read()

    def _multipart(self, audio: bytes, filename: str = "audio.wav",
                   content_type: str = "audio/wav") -> bytes:
        parts = [
            b"--%s\r\n" % _BOUNDARY.encode(),
            b'Content-Disposition: form-data; name="file"; filename="%s"\r\n'
            % filename.encode(),
            b"Content-Type: %s\r\n\r\n" % content_type.encode(),
            audio,
            b"\r\n",
            b"--%s\r\n" % _BOUNDARY.encode(),
            b'Content-Disposition: form-data; name="model"\r\n\r\n',
            self._model.encode(),
            b"\r\n",
            b"--%s--\r\n" % _BOUNDARY.encode(),
        ]
        return b"".join(parts)


_BOUNDARY = "waylandwhisper" + uuid.uuid4().hex[:16]
