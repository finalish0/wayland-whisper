"""The bearer token must come from the environment, never from the unit file.

The systemd drop-in names a file, not the token: the unit is readable by
anyone who can read the user systemd directory, and `systemctl show` prints
the environment of a running unit.
"""

import importlib.util
import os
import pathlib
import tempfile
import unittest

MODULE = pathlib.Path(__file__).resolve().parents[1] / "src" / "wayland_whisper" / "remote_recognizer.py"


def load():
    spec = importlib.util.spec_from_file_location("remote_recognizer", MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KeyFromEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.module = load()
        self.saved = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.saved)

    def test_the_key_is_read_from_the_variable(self):
        os.environ["WHISPER_REMOTE_API_KEY"] = "secret-from-variable"
        os.environ.pop("WHISPER_REMOTE_API_KEY_FILE", None)
        recognizer = self.module.RemoteWhisperRecognizer("http://x/rt")
        self.assertEqual(recognizer._api_key, "secret-from-variable")

    def test_the_key_is_read_from_the_named_file(self):
        os.environ.pop("WHISPER_REMOTE_API_KEY", None)
        with tempfile.NamedTemporaryFile("w", suffix=".key", delete=False) as handle:
            handle.write("secret-from-file\n")
            path = handle.name
        os.environ["WHISPER_REMOTE_API_KEY_FILE"] = path
        try:
            recognizer = self.module.RemoteWhisperRecognizer("http://x/rt")
            self.assertEqual(recognizer._api_key, "secret-from-file")
        finally:
            os.unlink(path)

    def test_the_variable_wins_over_the_file(self):
        os.environ["WHISPER_REMOTE_API_KEY"] = "from-variable"
        with tempfile.NamedTemporaryFile("w", suffix=".key", delete=False) as handle:
            handle.write("from-file")
            path = handle.name
        os.environ["WHISPER_REMOTE_API_KEY_FILE"] = path
        try:
            recognizer = self.module.RemoteWhisperRecognizer("http://x/rt")
            self.assertEqual(recognizer._api_key, "from-variable")
        finally:
            os.unlink(path)

    def test_without_any_key_the_recognizer_still_works(self):
        # The gateway may be reachable without a token; the recognizer must
        # not refuse to start because no key is configured.
        os.environ.pop("WHISPER_REMOTE_API_KEY", None)
        os.environ.pop("WHISPER_REMOTE_API_KEY_FILE", None)
        recognizer = self.module.RemoteWhisperRecognizer("http://x/rt")
        self.assertEqual(recognizer._api_key, "")


if __name__ == '__main__':
    unittest.main(verbosity=2)
