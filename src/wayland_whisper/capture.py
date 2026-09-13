"""Read fixed-size raw PCM frames from parec."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator

from .vad import FRAME_SAMPLES, SAMPLE_RATE


def pcm_frames() -> Iterator[bytes]:
    """Yield 16 kHz, mono, signed-16-bit frames until the recorder stops."""
    command = [
        "parec",
        "--raw",
        "--format=s16le",
        "--rate=%d" % SAMPLE_RATE,
        "--channels=1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    if process.stdout is None:
        process.terminate()
        raise RuntimeError("parec did not provide stdout")
    frame_size = FRAME_SAMPLES * 2
    try:
        while frame := process.stdout.read(frame_size):
            if len(frame) != frame_size:
                break
            yield frame
    finally:
        process.terminate()
        process.wait()
