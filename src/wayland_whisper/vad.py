"""Small ONNX Silero voice activity wrapper and phrase collector."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort


FRAME_SAMPLES = 512
SAMPLE_RATE = 16000


class SileroDetector:
    """Evaluate 32 ms PCM frames with the Silero ONNX state machine."""

    def __init__(self, model_path: str) -> None:
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._rate = np.array(SAMPLE_RATE, dtype=np.int64)

    def probability(self, pcm: bytes) -> float:
        if len(pcm) != FRAME_SAMPLES * 2:
            raise ValueError("Silero needs exactly one 512-sample PCM frame")
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        outputs = self._session.run(
            None,
            {"input": samples[np.newaxis, :], "state": self._state, "sr": self._rate},
        )
        self._state = outputs[1]
        return float(outputs[0][0][0])


@dataclass
class PhraseCollector:
    """Join voice frames into phrases while retaining a small leading buffer."""

    threshold: float = 0.50
    activation_frames: int = 6
    silence_frames: int = 22
    lead_frames: int = 8
    _lead: deque[bytes] = field(init=False)
    _parts: list[bytes] = field(default_factory=list, init=False)
    _voiced_frames: int = field(default=0, init=False)
    _quiet_frames: int = field(default=0, init=False)
    _recording: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._lead = deque(maxlen=self.lead_frames)

    def add(self, pcm: bytes, probability: float) -> bytes | None:
        """Accept one frame and return a completed phrase when silence settles."""
        voiced = probability >= self.threshold
        if not self._recording:
            self._lead.append(pcm)
            self._voiced_frames = self._voiced_frames + 1 if voiced else 0
            if self._voiced_frames < self.activation_frames:
                return None
            self._recording = True
            self._parts = list(self._lead)
            self._quiet_frames = 0
            return None

        self._parts.append(pcm)
        self._quiet_frames = 0 if voiced else self._quiet_frames + 1
        if self._quiet_frames < self.silence_frames:
            return None
        return self.finish()

    def finish(self) -> bytes | None:
        """Return a pending phrase and reset the collector for the next one."""
        phrase = b"".join(self._parts) if self._recording else None
        self._parts.clear()
        self._lead.clear()
        self._voiced_frames = 0
        self._quiet_frames = 0
        self._recording = False
        return phrase
