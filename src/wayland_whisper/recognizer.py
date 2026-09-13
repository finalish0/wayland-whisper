"""Local Whisper transcription."""

from __future__ import annotations

import numpy as np
from pywhispercpp.model import Model


class WhisperRecognizer:
    """Keep one whisper.cpp model in memory for consecutive phrases."""

    def __init__(self, model_path: str, language: str) -> None:
        self._model = Model(model=model_path, redirect_whispercpp_logs_to=False)
        self._language = language

    def transcribe(self, pcm: bytes) -> str:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        segments = self._model.transcribe(samples, language=self._language)
        return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
