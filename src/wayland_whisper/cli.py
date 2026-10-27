"""Command-line entry point."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import wave
from datetime import datetime

# Pin numpy's OpenBLAS to one thread before numpy is imported: the decode
# output of the fasterwhisper engine was measured to depend on it (2026-10-10
# benchmark transcripts require it), while the whispercpp engine is
# unaffected. Must run before numpy/OpenBLAS initialize.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from . import __version__
from .capture import pcm_frames
from .output import continued_text, type_text
from .recognizer import WhisperRecognizer
from .vad import SAMPLE_RATE, PhraseCollector, SileroDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wayland-whisper")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="capture, transcribe, and type speech")
    run.add_argument(
        "--engine",
        choices=("whispercpp", "fasterwhisper", "remote"),
        default="whispercpp",
        help="transcription backend (default: whispercpp)",
    )
    run.add_argument("--whisper-model", default="", help="GGML/GGUF model for the whispercpp engine")
    run.add_argument(
        "--remote-url",
        default="http://192.168.188.51:8888/v1/audio/transcriptions",
        help="transcription endpoint of the remote (AN) gateway",
    )
    run.add_argument(
        "--remote-model",
        default="whisper-large-v3-turbo",
        help="model name for the remote transcription route",
    )
    run.add_argument(
        "--remote-api-key",
        default="",
        help="bearer token for the remote gateway (default: $WHISPER_REMOTE_API_KEY)",
    )
    run.add_argument(
        "--remote-timeout",
        type=float,
        default=120.0,
        help="seconds to wait for the remote transcription (default: 120)",
    )
    run.add_argument(
        "--remote-audio",
        choices=("auto", "pcm", "mp3"),
        default="auto",
        help="upload format: 'auto' sends WAV inside the gateway's /24 and "
             "MP3 from everywhere else, 'pcm' always WAV, 'mp3' always MP3",
    )
    run.add_argument(
        "--remote-mp3-bitrate",
        type=int,
        default=48000,
        help="MP3 bit rate in bit/s for the compressed upload (default: 48000)",
    )
    run.add_argument(
        "--fallback-model",
        default="",
        help="local CTranslate2 model used when the remote engine fails "
             "(empty: no fallback)",
    )
    run.add_argument(
        "--fallback-compute",
        default="int8",
        help="CTranslate2 compute type for the fallback model (default: int8)",
    )
    run.add_argument(
        "--fallback-threads",
        type=int,
        default=4,
        help="CPU threads for the fallback model (default: 4)",
    )
    run.add_argument(
        "--faster-whisper-model",
        default="",
        help="CTranslate2 model directory for the fasterwhisper engine",
    )
    run.add_argument(
        "--faster-whisper-compute",
        default="int8",
        help="CTranslate2 compute type for the fasterwhisper engine (default: int8)",
    )
    run.add_argument(
        "--faster-whisper-threads",
        type=int,
        default=4,
        help="CPU threads for the fasterwhisper engine (default: 4)",
    )
    run.add_argument("--vad-model", required=True)
    run.add_argument("--language", default="de")
    run.add_argument("--threshold", type=float, default=0.50)
    run.add_argument(
        "--silence-frames",
        type=int,
        default=None,
        help="frames of silence that complete a phrase (default: 22 = ~704 ms at 32 ms/frame)",
    )
    run.add_argument("--output", choices=("wtype", "stdout"), default="wtype")
    run.add_argument(
        "--suspend-on-start",
        action="store_true",
        help="start with the microphone closed; SIGCONT ('resume') opens it",
    )
    run.add_argument("--cookie", default="", help="file the daemon writes its PID to")
    run.add_argument(
        "--phrase-dump",
        default="",
        help="directory to save every finished phrase to as a WAV (debugging/A-B tests)",
    )
    resume = commands.add_parser("resume", help="open the microphone of a suspended daemon")
    resume.add_argument("--cookie", default="", help="PID file written by 'run --cookie'")
    return parser


def _local_fallback_factory(arguments: argparse.Namespace):
    """Build the local engine for the fallback path (called once, lazily).

    The decode settings mirror the benchmarked configuration of the
    2026-10-10 CPU comparison; changing them would invalidate that
    measurement.
    """

    def factory():
        from .faster_recognizer import FasterWhisperRecognizer

        return FasterWhisperRecognizer(
            arguments.fallback_model,
            arguments.language,
            arguments.fallback_compute,
            arguments.fallback_threads,
        )

    return factory


def log_inference(arguments: argparse.Namespace, recognizer, phrase: bytes,
                  text: str, dumped: str = "") -> None:
    """Write the per-phrase timing line to stderr.

    The engine named here is the one that actually ran: with a fallback it
    can be the local one even though "remote" is configured. The journal
    must not claim the remote engine when the laptop did the work.
    """
    engine = arguments.engine
    reporter = getattr(recognizer, "last_engine", None)
    if callable(reporter):
        engine = reporter() or engine
    extra = dumped
    audio_reporter = getattr(recognizer, "last_audio", None)
    if callable(audio_reporter):
        label = audio_reporter()
        if label:
            # Which payload went out: pcm (LAN), mp3 (tunnel) or a fallback.
            extra = " audio=%s%s" % (label, extra)
    print(
        "wayland-whisper: inference engine=%s elapsed=%.2fs audio=%.2fs chars=%d%s"
        % (engine, arguments.elapsed, len(phrase) / 2 / SAMPLE_RATE, len(text), extra),
        file=sys.stderr,
    )


def build_recognizer(arguments: argparse.Namespace):
    """Create the configured transcription backend (model stays in memory)."""
    if arguments.engine == "remote":
        from .remote_recognizer import RemoteWhisperRecognizer

        remote = RemoteWhisperRecognizer(
            arguments.remote_url,
            arguments.remote_model,
            arguments.language,
            arguments.remote_api_key or None,
            arguments.remote_timeout,
            audio=getattr(arguments, "remote_audio", "pcm"),
            mp3_bit_rate=getattr(arguments, "remote_mp3_bitrate", 48000),
        )
        if not arguments.fallback_model:
            return remote
        # The local engine is built lazily, on the first remote failure only.
        from .fallback_recognizer import FallbackRecognizer

        return FallbackRecognizer(remote, _local_fallback_factory(arguments))
    if arguments.engine == "fasterwhisper":
        if not arguments.faster_whisper_model:
            raise RuntimeError("--faster-whisper-model is required with --engine fasterwhisper")
        from .faster_recognizer import FasterWhisperRecognizer

        return FasterWhisperRecognizer(
            arguments.faster_whisper_model,
            arguments.language,
            arguments.faster_whisper_compute,
            arguments.faster_whisper_threads,
        )
    if not arguments.whisper_model:
        raise RuntimeError("--whisper-model is required with --engine whispercpp")
    return WhisperRecognizer(arguments.whisper_model, arguments.language)


def dump_phrase_wav(directory: str, pcm: bytes) -> str:
    """Write one finished phrase to a 16 kHz mono WAV and return the path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(
        directory, "phrase-%s.wav" % datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm)
    return path


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
    collector_arguments = {"threshold": arguments.threshold}
    if arguments.silence_frames is not None:
        collector_arguments["silence_frames"] = arguments.silence_frames
    collector = PhraseCollector(**collector_arguments)
    recognizer = build_recognizer(arguments)
    print("wayland-whisper: engine=%s" % arguments.engine, file=sys.stderr)

    state = {"suspended": bool(arguments.suspend_on_start), "frames": None, "typed": False}

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

    def emit(text: str) -> None:
        """Deliver one phrase, separating it from the previous one."""
        text = continued_text(text, state["typed"])
        if arguments.output == "stdout":
            print(text, flush=True)
        else:
            type_text(text)
        state["typed"] = True

    def deliver(phrase: bytes) -> None:
        """Transcribe one finished phrase, log its timing, and deliver it."""
        dumped = ""
        if arguments.phrase_dump:
            dumped = " dump=%s" % os.path.basename(
                dump_phrase_wav(arguments.phrase_dump, phrase)
            )
        started = time.perf_counter()
        try:
            text = recognizer.transcribe(phrase)
        except (RuntimeError, subprocess.SubprocessError) as error:
            print("wayland-whisper: %s" % error, file=sys.stderr)
            return
        elapsed = time.perf_counter() - started
        arguments.elapsed = elapsed
        log_inference(arguments, recognizer, phrase, text, dumped)
        if text:
            emit(text)

    def close_microphone() -> None:
        """Deliver a half-collected phrase, then close the microphone."""
        if state["frames"] is None:
            return
        try:
            phrase = collector.finish()
            if phrase:
                deliver(phrase)
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
            state["typed"] = False
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
        deliver(phrase)
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
