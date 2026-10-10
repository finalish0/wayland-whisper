import argparse
import wave

import pytest

from wayland_whisper.cli import build_recognizer, dump_phrase_wav
from wayland_whisper.vad import SAMPLE_RATE


def test_dump_phrase_wav_writes_valid_wav(tmp_path) -> None:
    pcm = b"\x00\x01" * 512
    directory = str(tmp_path / "dumps")
    path = dump_phrase_wav(directory, pcm)
    assert path.startswith(directory)
    with wave.open(path, "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == SAMPLE_RATE
        assert handle.readframes(handle.getnframes()) == pcm


def test_cpp_engine_requires_whisper_model() -> None:
    arguments = argparse.Namespace(engine="whispercpp", whisper_model="", language="de")
    with pytest.raises(RuntimeError):
        build_recognizer(arguments)


def test_faster_engine_requires_model_directory() -> None:
    arguments = argparse.Namespace(
        engine="fasterwhisper",
        faster_whisper_model="",
        language="de",
        faster_whisper_compute="int8",
        faster_whisper_threads=4,
    )
    with pytest.raises(RuntimeError):
        build_recognizer(arguments)
