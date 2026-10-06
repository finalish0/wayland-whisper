"""Command-line entry point."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

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
    run.add_argument(
        "--suspend-on-start",
        action="store_true",
        help="start with the microphone closed; SIGCONT ('resume') opens it",
    )
    run.add_argument("--cookie", default="", help="file the daemon writes its PID to")
    resume = commands.add_parser("resume", help="open the microphone of a suspended daemon")
    resume.add_argument("--cookie", default="", help="PID file written by 'run --cookie'")
    return parser


def _resume_pid(cookie: str) -> int:
    if not cookie:
        print("wayland-whisper: --cookie is required", file=sys.stderr)
        return 1
    try:
        with open(cookie, encoding="utf-8") as handle:
            pid = int(handle.read().strip())
        os.kill(pid, signal.SIGCONT)
    except (OSError, ValueError) as error:
        print("wayland-whisper: %s" % error, file=sys.stderr)
        return 1
    return 0


def run(arguments: argparse.Namespace) -> int:
    detector = SileroDetector(arguments.vad_model)
    collector = PhraseCollector(threshold=arguments.threshold)
    recognizer = WhisperRecognizer(arguments.whisper_model, arguments.language)

    state = {"suspended": bool(arguments.suspend_on_start), "frames": None}

    def on_suspend(_signum: int, _frame: object) -> None:
        state["suspended"] = True

    def on_resume(_signum: int, _frame: object) -> None:
        state["suspended"] = False

    signal.signal(signal.SIGUSR1, on_suspend)
    signal.signal(signal.SIGTSTP, on_suspend)
    signal.signal(signal.SIGCONT, on_resume)

    if arguments.cookie:
        with open(arguments.cookie, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))

    def close_microphone() -> None:
        """Deliver a half-collected phrase, then close the microphone."""
        if state["frames"] is None:
            return
        try:
            phrase = collector.finish()
            if phrase:
                try:
                    text = recognizer.transcribe(phrase)
                except (RuntimeError, subprocess.SubprocessError) as error:
                    print("wayland-whisper: %s" % error, file=sys.stderr)
                    text = ""
                if text:
                    if arguments.output == "stdout":
                        print(text, flush=True)
                    else:
                        type_text(text)
        finally:
            state["frames"].close()
            state["frames"] = None
            print("wayland-whisper: paused (microphone closed)", file=sys.stderr)

    print(
        "wayland-whisper: ready (microphone closed, waiting for resume)"
        if state["suspended"]
        else "wayland-whisper: listening",
        file=sys.stderr,
    )

    while True:
        if state["suspended"]:
            # A suspend request can arrive while a phrase is being transcribed
            # or typed; that work has already finished here. Deliver the
            # phrase, then close the microphone -- otherwise it would stay
            # open forever and the pad "send" key could not stop dictation.
            close_microphone()
            time.sleep(0.02)
            continue

        frames = state["frames"]
        if frames is None:
            collector.reset()
            detector.reset()
            frames = pcm_frames()
            state["frames"] = frames
            print("wayland-whisper: listening", file=sys.stderr)

        try:
            frame = next(frames)
        except StopIteration:
            state["frames"] = None
            return 0
        except (OSError, RuntimeError) as error:
            state["frames"] = None
            print("wayland-whisper: %s" % error, file=sys.stderr)
            return 1

        if state["suspended"]:
            close_microphone()
            continue

        phrase = collector.add(frame, detector.probability(frame))
        if phrase is None:
            continue
        try:
            text = recognizer.transcribe(phrase)
        except (RuntimeError, subprocess.SubprocessError) as error:
            print("wayland-whisper: %s" % error, file=sys.stderr)
            continue
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
        if arguments.command == "resume":
            return _resume_pid(arguments.cookie)
        return run(arguments)
    except KeyboardInterrupt:
        return 130
    except RuntimeError as error:
        print("wayland-whisper: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
