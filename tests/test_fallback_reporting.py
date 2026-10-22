"""The journal must say which engine actually transcribed the phrase.

RED against the pre-change client: the log line always prints the
configured engine ("remote"), even when the fallback did the work. During
an outage the journal would then claim the remote engine while the text
came from the laptop - exactly the wrong thing to read afterwards.

The line must also name the fallback when it fires, so a slow or degraded
dictation is visible without guessing.
"""
import argparse
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "wayland_whisper"
MODULE = ROOT / "fallback_recognizer.py"


def load():
    spec = importlib.util.spec_from_file_location("fallback_recognizer", MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EngineReportingTests(unittest.TestCase):
    def setUp(self):
        self.module = load()

    def test_a_successful_remote_request_reports_the_remote_engine(self):
        remote = self._remote(failures=0)
        recognizer = self.module.FallbackRecognizer(remote, self._factory())
        recognizer.transcribe(b"pcm")
        self.assertEqual(recognizer.last_engine(), "remote")

    def test_a_fallback_reports_that_the_local_engine_ran(self):
        remote = self._remote(failures=1)
        recognizer = self.module.FallbackRecognizer(remote, self._factory())
        recognizer.transcribe(b"pcm")
        self.assertEqual(recognizer.last_engine(), "local")

    def test_the_engine_of_a_previous_request_is_not_reported_as_the_current_one(self):
        remote = self._remote(failures=1)
        recognizer = self.module.FallbackRecognizer(remote, self._factory())
        recognizer.transcribe(b"pcm")
        self.assertEqual(recognizer.last_engine(), "local")
        # The next request goes remote again and must say so.
        recognizer.transcribe(b"pcm")
        self.assertEqual(recognizer.last_engine(), "remote")

    def test_before_the_first_request_no_engine_ran(self):
        remote = self._remote(failures=0)
        recognizer = self.module.FallbackRecognizer(remote, self._factory())
        self.assertIsNone(recognizer.last_engine())

    def _remote(self, failures):
        class Remote:
            def __init__(self):
                self.failures = failures
                self.calls = 0

            def transcribe(self, pcm):
                self.calls += 1
                if self.calls <= self.failures:
                    raise RuntimeError("remote transcription failed")
                return "remote text"

        return Remote()

    def _factory(self):
        class Local:
            def transcribe(self, pcm):
                return "local text"

        return lambda: Local()


if __name__ == '__main__':
    unittest.main(verbosity=2)
