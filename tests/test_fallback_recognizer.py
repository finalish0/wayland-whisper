"""A failed remote transcription must fall back to the local engine.

RED against the pre-change client: the remote engine raises on a failed
request and the phrase is lost. If the laptop is offline, or AN is down, or
the gateway refuses, the dictation has to keep working - that is the whole
point of having a local engine in the first place.

The rules this pins down:
- The local engine is only loaded when it is actually needed (a remote
  failure). Loading faster-whisper costs ~3 GB of RAM and seconds of start
  time, so a healthy remote path must not pay for it.
- After a remote failure the local engine is tried for that phrase.
- A later phrase tries the remote engine again: AN may be back. The client
  must not stay stuck on the local path.
- A local failure is reported, not silently swallowed: an empty transcription
  must be distinguishable from silence.
"""
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


class FakeRemote:
    def __init__(self, failures=0):
        self.failures = failures
        self.calls = 0

    def transcribe(self, pcm):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("remote transcription failed: connection refused")
        return "remote text"


class FakeLocal:
    def __init__(self, text="local text"):
        self.text = text
        self.calls = 0

    def transcribe(self, pcm):
        self.calls += 1
        return self.text


class FakeLocalFactory:
    """Builds the local engine lazily, and counts the builds."""

    def __init__(self, text="local text"):
        self.text = text
        self.built = 0
        self.calls = 0

    def __call__(self):
        self.built += 1
        return FakeLocal(self.text)


class FallbackRecognizerTests(unittest.TestCase):
    def setUp(self):
        self.module = load()

    def test_a_healthy_remote_never_builds_the_local_engine(self):
        remote = FakeRemote()
        factory = FakeLocalFactory()
        recognizer = self.module.FallbackRecognizer(remote, factory)
        self.assertEqual(recognizer.transcribe(b"pcm"), "remote text")
        self.assertEqual(factory.built, 0, "the local engine was loaded for nothing")
        self.assertEqual(factory.calls, 0)

    def test_a_failed_remote_request_falls_back_to_the_local_engine(self):
        remote = FakeRemote(failures=1)
        factory = FakeLocalFactory()
        recognizer = self.module.FallbackRecognizer(remote, factory)
        self.assertEqual(recognizer.transcribe(b"pcm"), "local text")
        self.assertEqual(factory.built, 1)

    def test_a_later_request_tries_the_remote_engine_again(self):
        remote = FakeRemote(failures=1)
        factory = FakeLocalFactory()
        recognizer = self.module.FallbackRecognizer(remote, factory)
        self.assertEqual(recognizer.transcribe(b"pcm"), "local text")
        # The second request must go remote again: AN may be back.
        self.assertEqual(recognizer.transcribe(b"pcm"), "remote text")
        self.assertEqual(remote.calls, 2)
        # The local engine is built once and kept.
        self.assertEqual(factory.built, 1)

    def test_the_failure_is_reported_when_both_engines_fail(self):
        class BrokenLocal:
            def transcribe(self, pcm):
                raise RuntimeError("local engine failed")

        remote = FakeRemote(failures=99)
        recognizer = self.module.FallbackRecognizer(remote, lambda: BrokenLocal())
        with self.assertRaises(RuntimeError) as caught:
            recognizer.transcribe(b"pcm")
        self.assertIn("local", str(caught.exception).lower())

    def test_the_error_names_both_engines(self):
        class BrokenLocal:
            def transcribe(self, pcm):
                raise RuntimeError("local engine failed")

        remote = FakeRemote(failures=99)
        recognizer = self.module.FallbackRecognizer(remote, lambda: BrokenLocal())
        with self.assertRaises(RuntimeError) as caught:
            recognizer.transcribe(b"pcm")
        message = str(caught.exception)
        self.assertIn("remote", message.lower())
        self.assertIn("local", message.lower())

    def test_empty_pcm_never_reaches_an_engine(self):
        remote = FakeRemote()
        factory = FakeLocalFactory()
        recognizer = self.module.FallbackRecognizer(remote, factory)
        self.assertEqual(recognizer.transcribe(b""), "")
        self.assertEqual(remote.calls, 0)
        self.assertEqual(factory.built, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
