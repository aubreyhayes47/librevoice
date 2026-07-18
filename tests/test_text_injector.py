"""Regression tests for latency-sensitive desktop output behavior."""

import os
import subprocess
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("PYSTRAY_BACKEND", "dummy")

from daemon import TextInjector


class TextInjectorTests(unittest.TestCase):
    @patch("daemon.subprocess.run")
    def test_wl_copy_does_not_capture_forked_child_pipes(self, run):
        run.return_value = Mock(returncode=0)
        injector = TextInjector()
        injector._wl_clipboard_available = True

        injector._copy_to_clipboard("complete text")

        _, kwargs = run.call_args
        self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)
        self.assertEqual(kwargs["timeout"], 3)
        self.assertNotIn("capture_output", kwargs)

    @patch("daemon.Path.exists", return_value=True)
    @patch("daemon.subprocess.run")
    def test_existing_ydotool_socket_does_not_restart_daemon(self, run, exists):
        injector = TextInjector()

        self.assertTrue(injector._ensure_ydotool_backend())
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
