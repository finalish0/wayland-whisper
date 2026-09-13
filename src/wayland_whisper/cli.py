"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .capture import pcm_frames
from .output import type_text
from .recognizer import WhisperRecognizer
from .vad import PhraseCollector, SileroDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wayland-whisper")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="capture, transcribe, and type speech")
    run.add_argument("--whisper-model", required=True)
    run.add_argument("--vad-model", required=True)
    run.add_argument("--language", default="de")
    run.add_argument("--threshold", type=float, default=0.50)
    run.add_argument("--output", choices=("wtype", "stdout"), default="wtype")
    return parser


def run(arguments: argparse.Namespace) -> int:
    detector = SileroDetector(arguments.vad_model)
    collector = PhraseCollector(threshold=arguments.threshold)
    recognizer = WhisperRecognizer(arguments.whisper_model, arguments.language)
    print("wayland-whisper: listening", file=sys.stderr)
    for frame in pcm_frames():
        phrase = collector.add(frame, detector.probability(frame))
        if phrase is None:
            continue
        text = recognizer.transcribe(phrase)
        if not text:
            continue
        if arguments.output == "stdout":
            print(text, flush=True)
        else:
            type_text(text)
    return 0


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        return run(arguments)
    except KeyboardInterrupt:
        return 130
    except RuntimeError as error:
        print("wayland-whisper: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
