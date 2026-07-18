"""Regression tests for the small, hardware-independent hotkey boundary."""

import os
import unittest

import numpy as np

# Importing pystray normally chooses an X backend.  The hotkey code has no UI
# dependency, so use its dummy backend to keep this test headless and portable.
os.environ.setdefault("PYSTRAY_BACKEND", "dummy")

from daemon import AudioRecorder, HotkeyListener


class Event:
    def __init__(self, event_type, code, value):
        self.type = event_type
        self.code = code
        self.value = value


class Device:
    def __init__(self, events):
        self._events = events

    def read_loop(self):
        yield from self._events


class HotkeyListenerTests(unittest.TestCase):
    def test_blocking_reader_emits_one_press_and_release(self):
        listener = HotkeyListener("KEY_RIGHTCTRL", "hold")
        received = []
        listener.set_callback(received.append)
        listener._running = True

        listener._listen_device(
            Device(
                [
                    Event(1, listener.key_code, 1),  # key down
                    Event(1, listener.key_code, 2),  # autorepeat: ignore
                    Event(1, listener.key_code, 0),  # key up
                ]
            )
        )

        self.assertEqual(received, [True, False])

    def test_warm_recorder_prepends_preroll_to_utterance(self):
        recorder = AudioRecorder(sample_rate=1000, pre_roll_ms=400)
        recorder._stream = object()  # Avoid opening real audio hardware.
        recorder._pre_roll_frames.extend(
            [np.ones((100, 1), dtype=np.float32), np.ones((100, 1), dtype=np.float32)]
        )

        self.assertTrue(recorder.start())
        recorder._frames.append(np.ones((100, 1), dtype=np.float32))
        audio = recorder.stop()

        self.assertEqual(len(audio), 300)


if __name__ == "__main__":
    unittest.main()
