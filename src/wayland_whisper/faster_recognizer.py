"""faster-whisper (CTranslate2) transcription backend."""

from __future__ import annotations

import os

# The decode output depends on numpy's OpenBLAS threading (measured
# 2026-10-10); pin it before numpy is imported if this module loads first.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np


class FasterWhisperRecognizer:
    """Keep one CTranslate2 Whisper model in memory for consecutive phrases.

    The decode settings mirror the benchmarked configuration of the
    2026-10-10 CPU comparison (faster-whisper small, INT8, beam 5, German,
    OPENBLAS_NUM_THREADS=1); changing them would invalidate that measurement.
    """

    def __init__(
        self,
        model_path: str,
        language: str,
        compute_type: str = "int8",
        threads: int = 4,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise RuntimeError(
                "faster-whisper is not installed; install the 'faster' extra "
                "(pip install -e '.[faster]')"
            ) from error
        self._model = WhisperModel(
            model_path,
            device="cpu",
            compute_type=compute_type,
            cpu_threads=threads,
            num_workers=1,
            local_files_only=True,
        )
        self._language = language

    def transcribe(self, pcm: bytes) -> str:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        segments, _info = self._model.transcribe(
            samples,
            language=self._language,
            task="transcribe",
            beam_size=5,
            best_of=5,
            condition_on_previous_text=False,
            word_timestamps=False,
            vad_filter=False,
        )
        return " ".join(
            segment.text.strip() for segment in segments if segment.text.strip()
        ).strip()
