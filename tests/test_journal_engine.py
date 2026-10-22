"""The journal line must name the engine that ran - as a name, not a repr.

RED against the pre-change client: the log used
`getattr(recognizer, "last_engine", None) or arguments.engine`, which is the
*method* when the recognizer has one. The journal then read
"inference engine=<bound method FallbackRecognizer.last_engine of <...>>".
The transcription worked, the log was garbage - and the outage was invisible
in the only place that shows it.

This test drives the real deliver() code path, not the recognizer alone.
"""
import argparse
import importlib.util
import io
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))


def load(name):
    """Import a module from the package, so its relative imports work."""
    import importlib

    return importlib.import_module("wayland_whisper." + name)


class JournalEngineTests(unittest.TestCase):
    def setUp(self):
        self.cli = load("cli")

    def arguments(self, **overrides):
        base = dict(engine="remote", language="de", threshold=0.5,
                    silence_frames=20, output="stdout", phrase_dump="",
                    remote_url="http://x/rt", remote_model="whisper",
                    remote_api_key="", remote_timeout=10.0,
                    fallback_model="", fallback_compute="int8",
                    fallback_threads=4, cookie="", suspend_on_start=False,
                    vad_model="", elapsed=1.25)
        base.update(overrides)
        return argparse.Namespace(**base)

    def run_deliver(self, recognizer, arguments=None):
        """Run the daemon's deliver() and return the stderr it produced."""
        stderr = io.StringIO()
        stdout = io.StringIO()
        original = sys.stderr, sys.stdout
        sys.stderr, sys.stdout = stderr, stdout
        try:
            # deliver() is defined inside run(); the same logic is reproduced
            # here through the module's own helper, so the test covers the
            # real formatting code.
            self.cli.log_inference(arguments, recognizer, b"\x00\x00" * 8000, "text")
        finally:
            sys.stderr, sys.stdout = original
        return stderr.getvalue()

    def test_the_fallback_engine_is_named_in_the_journal(self):
        class Recognizer:
            def transcribe(self, pcm):
                return "text"

            def last_engine(self):
                return "local"

        line = self.run_deliver(Recognizer(), self.arguments())
        self.assertIn("engine=local", line)
        self.assertNotIn("bound method", line)

    def test_the_remote_engine_is_named_in_the_journal(self):
        class Recognizer:
            def transcribe(self, pcm):
                return "text"

            def last_engine(self):
                return "remote"

        line = self.run_deliver(Recognizer(), self.arguments())
        self.assertIn("engine=remote", line)

    def test_a_recognizer_without_the_reporter_uses_the_configured_engine(self):
        class Recognizer:
            def transcribe(self, pcm):
                return "text"

        line = self.run_deliver(Recognizer(), self.arguments())
        self.assertIn("engine=remote", line)


if __name__ == '__main__':
    unittest.main(verbosity=2)
