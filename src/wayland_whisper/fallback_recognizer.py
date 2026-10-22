"""Automatic fallback: the remote engine, with a local engine behind it.

The remote path (the AN GPU) is the fast and accurate one, but a dictation
must survive the case where it is not reachable: the laptop is offline, AN
is down, the gateway refuses, the network hiccups.

The rules:

- The remote engine is tried for every phrase. AN may come back, so the
  client must not get stuck on the local path.
- The local engine is built lazily, on the first remote failure only.
  Loading faster-whisper costs about 3 GB of RAM and seconds of start time;
  a healthy remote path must not pay for it.
- When both engines fail, the error names both - a lost phrase must be
  visible in the journal, not look like silence.
"""

from __future__ import annotations


class FallbackRecognizer:
    """Try the remote engine, fall back to the local one when it fails."""

    def __init__(self, remote, local_factory) -> None:
        self._remote = remote
        self._local_factory = local_factory
        self._local = None
        self._local_failures = 0
        self._last_engine = None

    def transcribe(self, pcm: bytes) -> str:
        if not pcm:
            return ""
        try:
            text = self._remote.transcribe(pcm)
        except Exception as remote_error:
            local = self._ensure_local()
            if local is None:
                self._last_engine = "remote"
                raise RuntimeError(
                    "transcription failed: remote (%s) and no local engine configured"
                    % remote_error
                ) from remote_error
            try:
                text = local.transcribe(pcm)
            except Exception as local_error:
                self._last_engine = "local"
                raise RuntimeError(
                    "transcription failed: remote (%s) and local (%s)"
                    % (remote_error, local_error)
                ) from local_error
            self._last_engine = "local"
            return text
        self._last_engine = "remote"
        return text

    def last_engine(self):
        """Which engine did the last request: "remote", "local" or None."""
        return self._last_engine

    def _ensure_local(self):
        """Build the local engine once, on the first remote failure."""
        if self._local is None:
            self._local = self._local_factory()
        return self._local

    @property
    def local_engine(self):
        """The local engine, or None while it has not been needed."""
        return self._local
