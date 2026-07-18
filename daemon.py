#!/usr/bin/env python3
"""LibreVoice - System-wide push-to-talk voice dictation daemon.

Hold Right Ctrl → speak → release → text appears at your cursor.
Uses OpenVINO WhisperPipeline on GPU for fast transcription.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

import evdev
import numpy as np
import notify2
import pystray
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG_PATH = Path.home() / ".config" / "librevoice" / "config.json"
DEFAULT_CONFIG = {
    "hotkey": "KEY_RIGHTCTRL",
    "mode": "hold",  # hold = record while held, toggle = press on/off
    "model_path": str(
        Path.home()
        / "Desktop"
        / "librevoice"
        / "models"
        / "whisper-large-v3-turbo-fp16"
    ),
    "device": "GPU",
    "fallback_devices": ["CPU"],
    "language": "en",
    "max_duration_sec": 30,
    "sample_rate": 16000,
    "typing_delay_ms": 12,
    "clipboard": True,
    "notifications": True,
    "log_level": "INFO",
}


def load_config():
    """Load config, creating default if missing."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            user_cfg = json.load(f)
        cfg = {**DEFAULT_CONFIG, **user_cfg}
    else:
        cfg = DEFAULT_CONFIG.copy()
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
    return cfg


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

import logging

logger = logging.getLogger("librevoice")


def setup_logging(level_str="INFO"):
    level = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
        )
    )
    logger.addHandler(handler)


# ---------------------------------------------------------------------------
# Audio recorder
# ---------------------------------------------------------------------------


class AudioRecorder:
    """Records audio from the microphone using sounddevice."""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self._stream = None
        self._frames = []
        self._recording = False
        self._lock = threading.Lock()
        self._available = None
        self._error_msg = None

    def check_mic(self):
        """Check if a microphone is available."""
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            default_input = sd.default.device[0]
            if default_input is None:
                self._available = False
                self._error_msg = "No default input device found"
                return False
            dev_info = sd.query_devices(default_input)
            if dev_info["max_input_channels"] < 1:
                self._available = False
                self._error_msg = "Default device has no input channels"
                return False
            self._available = True
            self._error_msg = None
            return True
        except Exception as e:
            self._available = False
            self._error_msg = str(e)
            return False

    def start(self):
        """Start recording audio."""
        if self._recording:
            return True

        if self._available is None:
            self.check_mic()
        if not self._available:
            logger.warning(f"Mic unavailable: {self._error_msg}")
            return False

        try:
            import sounddevice as sd

            self._frames = []
            self._recording = True

            def callback(indata, frames, time_info, status):
                if status:
                    logger.warning(f"Audio status: {status}")
                with self._lock:
                    self._frames.append(indata.copy())

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=callback,
                blocksize=int(self.sample_rate * 0.02),  # 20ms blocks
            )
            self._stream.start()
            logger.info("Recording started")
            return True
        except Exception as e:
            self._recording = False
            self._available = False
            self._error_msg = str(e)
            logger.error(f"Failed to start recording: {e}")
            return False

    def stop(self):
        """Stop recording and return the audio as a numpy array."""
        if not self._recording:
            return None

        self._recording = False
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
        except Exception as e:
            logger.error(f"Error stopping stream: {e}")

        with self._lock:
            if not self._frames:
                logger.warning("No audio captured")
                return None
            audio = np.concatenate(self._frames, axis=0).flatten()
            self._frames = []

        logger.info(f"Captured {len(audio) / self.sample_rate:.2f}s of audio")
        return audio

    @property
    def is_recording(self):
        return self._recording


# ---------------------------------------------------------------------------
# Transcriber
# ---------------------------------------------------------------------------


class Transcriber:
    """Transcribes audio using OpenVINO WhisperPipeline."""

    def __init__(self, model_path, device="GPU", fallback_devices=None):
        self.model_path = model_path
        self.device = device
        self.fallback_devices = fallback_devices or ["CPU"]
        self._pipe = None
        self._loaded_device = None
        self._loading = False

    def _load(self, device):
        """Load the model on a specific device."""
        import openvino_genai as ogai

        logger.info(f"Loading model on {device}...")
        start = time.time()
        try:
            pipe = ogai.WhisperPipeline(self.model_path, device)
            elapsed = time.time() - start
            logger.info(f"Model loaded on {device} in {elapsed:.2f}s")
            return pipe
        except Exception as e:
            logger.error(f"Failed to load on {device}: {e}")
            return None

    def ensure_loaded(self):
        """Ensure the model is loaded, trying fallback devices."""
        if self._pipe is not None:
            return True

        self._loading = True
        try:
            # Try primary device first
            pipe = self._load(self.device)
            if pipe:
                self._pipe = pipe
                self._loaded_device = self.device
                return True

            # Try fallback devices
            for device in self.fallback_devices:
                if device == self.device:
                    continue
                pipe = self._load(device)
                if pipe:
                    self._pipe = pipe
                    self._loaded_device = device
                    logger.info(f"Using fallback device: {device}")
                    return True

            logger.error("Failed to load model on any device")
            return False
        finally:
            self._loading = False

    @property
    def is_loading(self):
        return self._loading

    @property
    def is_loaded(self):
        return self._pipe is not None

    @property
    def active_device(self):
        return self._loaded_device

    def transcribe(self, audio, language="en"):
        """Transcribe audio and return the text."""
        if not self.ensure_loaded():
            return None, "Model not loaded"

        try:
            start = time.time()
            result = self._pipe.generate(audio, return_timestamps=True)
            elapsed = time.time() - start

            text = ""
            if hasattr(result, "chunks") and result.chunks:
                text = "".join(chunk.text for chunk in result.chunks)
            elif isinstance(result, str):
                text = result
            else:
                text = str(result)

            logger.info(f"Transcribed in {elapsed:.2f}s: {text[:80]}...")
            return text.strip(), None
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None, str(e)


# ---------------------------------------------------------------------------
# Text injector
# ---------------------------------------------------------------------------


class TextInjector:
    """Injects text at the cursor position using ydotool (Wayland) or xdotool (X11)."""

    def __init__(self, clipboard=True, typing_delay_ms=12):
        self.clipboard = clipboard
        self.typing_delay_ms = typing_delay_ms
        self._session_type = os.environ.get("XDG_SESSION_TYPE", "x11")
        self._wayland = self._session_type == "wayland"
        self._tools_checked = False
        self._ydotool_available = False
        self._xdotool_available = False
        self._wl_clipboard_available = False

    def _check_tools(self):
        """Check which injection tools are available."""
        if self._tools_checked:
            return

        # Check ydotool
        try:
            result = subprocess.run(
                ["which", "ydotool"], capture_output=True, text=True
            )
            self._ydotool_available = result.returncode == 0
        except Exception:
            self._ydotool_available = False

        # Check xdotool
        try:
            result = subprocess.run(
                ["which", "xdotool"], capture_output=True, text=True
            )
            self._xdotool_available = result.returncode == 0
        except Exception:
            self._xdotool_available = False

        # Check wl-clipboard
        try:
            result = subprocess.run(
                ["which", "wl-copy"], capture_output=True, text=True
            )
            self._wl_clipboard_available = result.returncode == 0
        except Exception:
            self._wl_clipboard_available = False

        self._tools_checked = True
        logger.info(
            f"Tools: ydotool={self._ydotool_available}, "
            f"xdotool={self._xdotool_available}, "
            f"wl-copy={self._wl_clipboard_available}, "
            f"session={self._session_type}"
        )

    def inject(self, text):
        """Inject text at the cursor position."""
        if not text or not text.strip():
            return False

        self._check_tools()

        # Always copy to clipboard if requested
        if self.clipboard:
            self._copy_to_clipboard(text)

        # Try to type the text
        if self._wayland:
            return self._inject_wayland(text)
        else:
            return self._inject_x11(text)

    def _copy_to_clipboard(self, text):
        """Copy text to system clipboard."""
        try:
            if self._wl_clipboard_available:
                subprocess.run(
                    ["wl-copy"],
                    input=text.encode(),
                    capture_output=True,
                    timeout=15,  # Increased timeout
                )
                logger.debug("Copied to clipboard via wl-copy")
            else:
                # Fallback to xclip
                subprocess.run(
                    ["xclip", "-selection", "clipboard"],
                    input=text.encode(),
                    capture_output=True,
                    timeout=15,
                )
                logger.debug("Copied to clipboard via xclip")
        except subprocess.TimeoutExpired:
            logger.warning("Clipboard copy timed out")
        except Exception as e:
            logger.warning(f"Clipboard copy failed: {e}")

    def _inject_wayland(self, text):
        """Inject text on Wayland using ydotool."""
        if not self._ydotool_available:
            logger.warning("ydotool not available for Wayland injection")
            return False

        try:
            result = subprocess.run(
                ["ydotool", "type", "--delay", str(self.typing_delay_ms), "--", text],
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode(errors="replace").strip()
                logger.error(f"ydotool failed (rc={result.returncode}): {stderr}")
                return False
            logger.debug("Typed via ydotool")
            return True
        except FileNotFoundError:
            logger.error("ydotool binary not found")
            self._ydotool_available = False
            return False
        except subprocess.TimeoutExpired:
            logger.error("ydotool type timed out")
            return False
        except Exception as e:
            logger.error(f"ydotool injection failed: {e}")
            return False

    def _inject_x11(self, text):
        """Inject text on X11 using xdotool."""
        if not self._xdotool_available:
            logger.warning("xdotool not available for X11 injection")
            return False

        try:
            subprocess.run(
                ["xdotool", "type", "--delay", str(self.typing_delay_ms), "--", text],
                capture_output=True,
                timeout=10,
            )
            logger.debug("Typed via xdotool")
            return True
        except Exception as e:
            logger.error(f"xdotool injection failed: {e}")
            return False


# ---------------------------------------------------------------------------
# Tray icon
# ---------------------------------------------------------------------------


class TrayIcon:
    """System tray icon showing daemon status."""

    def __init__(self):
        self._icon = None
        self._running = False
        self._state = "idle"  # idle, recording, transcribing, loading, error
        self._error_msg = ""

    def _create_image(self, state):
        """Create an icon image for the given state."""
        colors = {
            "idle": "#808080",  # Gray
            "recording": "#00C853",  # Green
            "transcribing": "#FFD600",  # Yellow
            "loading": "#2196F3",  # Blue
            "error": "#FF1744",  # Red
        }
        color = colors.get(state, "#808080")

        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Draw microphone icon
        draw.rounded_rectangle([20, 8, 44, 36], radius=12, fill=color)
        draw.rectangle([28, 36, 36, 44], fill=color)
        draw.arc([16, 40, 48, 64], 0, 180, fill=color, width=3)
        draw.line([32, 52, 32, 60], fill=color, width=3)

        return img

    def start(self):
        """Start the tray icon."""
        self._running = True

        def on_exit(icon, item):
            self._running = False
            icon.stop()

        self._icon = pystray.Icon(
            "librevoice",
            self._create_image("idle"),
            "LibreVoice - Idle",
            menu=pystray.Menu(
                pystray.MenuItem("Quit", on_exit),
            ),
        )

        # Run icon in a thread
        threading.Thread(target=self._icon.run, daemon=True).start()
        logger.info("Tray icon started")

    def stop(self):
        """Stop the tray icon."""
        self._running = False
        if self._icon:
            self._icon.stop()

    def set_state(self, state, error_msg=""):
        """Update the tray icon state."""
        if not self._icon:
            return

        self._state = state
        self._error_msg = error_msg

        labels = {
            "idle": "LibreVoice - Ready (hold Right Ctrl)",
            "recording": "LibreVoice - Recording...",
            "transcribing": "LibreVoice - Transcribing...",
            "loading": "LibreVoice - Loading model...",
            "error": f"LibreVoice - Error: {error_msg[:40]}",
        }

        try:
            self._icon.icon = self._create_image(state)
            self._icon.title = labels.get(state, "LibreVoice")
        except Exception as e:
            logger.error(f"Failed to update tray icon: {e}")


# ---------------------------------------------------------------------------
# Notification helper
# ---------------------------------------------------------------------------


def notify(summary, message="", urgency=notify2.URGENCY_NORMAL):
    """Send a desktop notification."""
    try:
        if not notify2.is_initted():
            notify2.init("LibreVoice")
        n = notify2.Notification(summary, message, "audio-input-microphone")
        n.set_urgency(urgency)
        n.show()
    except Exception as e:
        logger.warning(f"Notification failed: {e}")


# ---------------------------------------------------------------------------
# Hotkey listener
# ---------------------------------------------------------------------------


class HotkeyListener:
    """Listens for a global hotkey using evdev."""

    def __init__(self, key_name="KEY_RIGHTCTRL", mode="hold"):
        self.key_name = key_name
        self.key_code = getattr(evdev.ecodes, key_name, None)
        self.mode = mode  # "hold" or "toggle"
        self._callback = None
        self._running = False
        self._key_pressed = False
        self._toggle_state = False
        self._devices = []

    def set_callback(self, callback):
        """Set the callback for key events: callback(is_pressed)."""
        self._callback = callback

    def _find_keyboard_devices(self):
        """Find input devices that have our target key in their capabilities."""
        devices = []
        if self.key_code is None:
            logger.error(f"Unknown key name: {self.key_name}")
            return devices
        try:
            for path in evdev.list_devices():
                try:
                    device = evdev.InputDevice(path)
                    caps = device.capabilities()
                    key_caps = caps.get(evdev.ecodes.EV_KEY, [])
                    if (
                        self.key_code in key_caps
                        and "ydotool" not in device.name.lower()
                    ):
                        devices.append(device)
                        logger.debug(f"Found {self.key_name} on {device.name} ({path})")
                except (PermissionError, OSError):
                    continue
        except Exception as e:
            logger.error(f"Failed to enumerate devices: {e}")
        return devices

    def start(self):
        """Start listening for the hotkey."""
        self._running = True
        self._devices = self._find_keyboard_devices()

        if not self._devices:
            logger.error("No keyboard devices found! Are you in the 'input' group?")
            return False

        logger.info(f"Listening for {self.key_name} on {len(self._devices)} devices")

        # Start listener thread
        threading.Thread(target=self._listen, daemon=True).start()
        return True

    def stop(self):
        """Stop listening."""
        self._running = False
        for device in self._devices:
            try:
                device.close()
            except Exception:
                pass

    def _listen(self):
        """Main listening loop."""
        import select

        while self._running:
            try:
                fds = {d.fd: d for d in self._devices if d.fd >= 0}
                if not fds:
                    logger.error("No valid device file descriptors")
                    time.sleep(1)
                    self._devices = self._find_keyboard_devices()
                    continue

                # Wait for events
                r, _, _ = select.select(fds, [], [], 0.1)

                for fd in r:
                    device = fds.get(fd)
                    if not device:
                        continue

                    try:
                        for event in device.read():
                            if event.type == evdev.ecodes.EV_KEY:
                                if event.code == self.key_code:
                                    if event.value == evdev.ecodes.KEY_DOWN:
                                        logger.info("Hotkey pressed: %s", self.key_name)
                                        self._handle_key(True)
                                    elif event.value == evdev.ecodes.KEY_UP:
                                        logger.info(
                                            "Hotkey released: %s", self.key_name
                                        )
                                        self._handle_key(False)
                    except (OSError, IOError):
                        # Device disconnected, try to re-enumerate
                        self._devices = self._find_keyboard_devices()
                        break
            except Exception as e:
                logger.error(f"Listen error: {e}")
                time.sleep(0.5)

    def _handle_key(self, is_pressed):
        """Handle a key press/release event."""
        if self.mode == "hold":
            if is_pressed and not self._key_pressed:
                self._key_pressed = True
                if self._callback:
                    self._callback(True)
            elif not is_pressed and self._key_pressed:
                self._key_pressed = False
                if self._callback:
                    self._callback(False)
        else:  # toggle
            if is_pressed:
                self._toggle_state = not self._toggle_state
                if self._callback:
                    self._callback(self._toggle_state)


# ---------------------------------------------------------------------------
# Socket trigger listener (for external triggers like GNOME shortcuts)
# ---------------------------------------------------------------------------


class SocketTriggerListener:
    """Listens for push-to-talk triggers via Unix socket."""

    def __init__(self, socket_path="/tmp/librevoice-trigger.sock"):
        self.socket_path = socket_path
        self._callback = None
        self._running = False
        self._server = None
        self._thread = None
        self._toggle_state = False  # Track state for toggle mode

    def set_callback(self, callback):
        """Set callback for trigger events: callback(is_pressed)."""
        self._callback = callback

    def start(self):
        """Start listening for triggers."""
        # Clean up any existing socket
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass

        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self.socket_path)
        self._server.listen(1)
        os.chmod(self.socket_path, 0o666)  # Allow any user to connect

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"Socket trigger listener started on {self.socket_path}")
        return True

    def stop(self):
        """Stop listening."""
        self._running = False
        if self._server:
            try:
                self._server.close()
            except Exception:
                pass
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run(self):
        """Main listening loop."""
        while self._running:
            try:
                self._server.settimeout(0.5)
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Socket accept error: {e}")
                break

            try:
                data = conn.recv(16).decode().strip()
                conn.close()
                # Toggle mode: each "press" flips the state
                if data == "press":
                    self._toggle_state = not self._toggle_state
                    if self._callback:
                        self._callback(self._toggle_state)
                elif data == "release" and self._callback:
                    # Explicit release (for hold mode compatibility)
                    self._callback(False)
            except Exception as e:
                logger.error(f"Socket trigger error: {e}")


# ---------------------------------------------------------------------------
# Main daemon
# ---------------------------------------------------------------------------


class LibreVoiceDaemon:
    """Main daemon orchestrating all components."""

    def __init__(self):
        self.config = load_config()
        setup_logging(self.config["log_level"])

        self.recorder = AudioRecorder(self.config["sample_rate"])
        self.transcriber = Transcriber(
            self.config["model_path"],
            self.config["device"],
            self.config["fallback_devices"],
        )
        self.injector = TextInjector(
            self.config["clipboard"],
            self.config["typing_delay_ms"],
        )
        self.tray = TrayIcon()
        self.listener = HotkeyListener(
            self.config["hotkey"],
            self.config["mode"],
        )
        self.socket_listener = SocketTriggerListener()

        self._is_transcribing = False
        self._lock = threading.Lock()
        self._max_duration_timer = None

    def _on_hotkey(self, is_pressed):
        """Handle hotkey press/release."""
        with self._lock:
            if is_pressed:
                self._start_recording()
            else:
                self._stop_recording_and_transcribe()

    def _start_recording(self):
        """Start recording audio."""
        if self._is_transcribing:
            logger.info("Still transcribing previous audio, skipping")
            return

        if not self.recorder.check_mic():
            msg = self.recorder._error_msg or "Microphone not available"
            logger.error(f"Cannot record: {msg}")
            self.tray.set_state("error", msg)
            if self.config["notifications"]:
                notify("LibreVoice Error", msg, notify2.URGENCY_CRITICAL)
            # Auto-clear error after 3s
            threading.Timer(3.0, lambda: self.tray.set_state("idle")).start()
            return

        if self.recorder.start():
            self.tray.set_state("recording")
            max_sec = self.config.get("max_duration_sec", 30)
            if max_sec > 0:
                self._max_duration_timer = threading.Timer(
                    max_sec, self._on_max_duration
                )
                self._max_duration_timer.daemon = True
                self._max_duration_timer.start()
        else:
            self.tray.set_state("error", "Failed to start recording")

    def _on_max_duration(self):
        """Auto-stop recording if max duration exceeded."""
        logger.info("Max recording duration reached, stopping")
        self._stop_recording_and_transcribe()

    def _stop_recording_and_transcribe(self):
        """Stop recording and transcribe the audio."""
        if self._max_duration_timer:
            self._max_duration_timer.cancel()
            self._max_duration_timer = None
        audio = self.recorder.stop()
        if audio is None or len(audio) < self.config["sample_rate"] * 0.3:
            logger.info("Audio too short, skipping")
            self.tray.set_state("idle")
            return

        self.tray.set_state("transcribing")
        self._is_transcribing = True

        def do_transcribe():
            try:
                # Ensure model is loaded
                if not self.transcriber.ensure_loaded():
                    self.tray.set_state("error", "Failed to load model")
                    if self.config["notifications"]:
                        notify(
                            "LibreVoice Error",
                            "Failed to load Whisper model",
                            notify2.URGENCY_CRITICAL,
                        )
                    return

                # Transcribe
                text, error = self.transcriber.transcribe(
                    audio, self.config["language"]
                )
                if error:
                    self.tray.set_state("error", error)
                    if self.config["notifications"]:
                        notify(
                            "LibreVoice Error",
                            f"Transcription failed: {error}",
                            notify2.URGENCY_CRITICAL,
                        )
                    return

                if not text:
                    logger.info("No speech detected")
                    self.tray.set_state("idle")
                    return

                # Inject text
                success = self.injector.inject(text)
                if success:
                    logger.info(f"Injected: {text[:80]}...")
                    self.tray.set_state("idle")
                else:
                    self.tray.set_state("error", "Failed to inject text")
                    # Still copy to clipboard as fallback
                    if self.config["clipboard"]:
                        logger.info("Text copied to clipboard as fallback")
            except Exception as e:
                logger.error(f"Transcription error: {e}\n{traceback.format_exc()}")
                self.tray.set_state("error", str(e))
            finally:
                self._is_transcribing = False
                # Return to idle after a short delay
                threading.Timer(2.0, lambda: self.tray.set_state("idle")).start()

        threading.Thread(target=do_transcribe, daemon=True).start()

    def run(self):
        """Run the daemon."""
        logger.info("LibreVoice daemon starting...")

        # Start socket trigger listener (for GNOME shortcuts)
        self.socket_listener.set_callback(self._on_hotkey)
        self.socket_listener.start()

        # Check mic
        if not self.recorder.check_mic():
            msg = self.recorder._error_msg or "Microphone not available"
            logger.warning(f"Mic check failed: {msg}")
            if self.config["notifications"]:
                notify(
                    "LibreVoice",
                    f"Mic not available: {msg}. Will retry when key is pressed.",
                    notify2.URGENCY_LOW,
                )

        # Start tray icon
        self.tray.start()

        # Install callback before starting the listener thread.
        self.listener.set_callback(self._on_hotkey)

        # Start hotkey listener
        if not self.listener.start():
            logger.error("Failed to start hotkey listener")
            self.tray.set_state("error", "No keyboard devices found")
            if self.config["notifications"]:
                notify(
                    "LibreVoice Error",
                    "No keyboard devices. Are you in the 'input' group?",
                    notify2.URGENCY_CRITICAL,
                )
            return

        # Pre-load model in background
        def preload():
            self.tray.set_state("loading")
            if self.transcriber.ensure_loaded():
                self.tray.set_state("idle")
            else:
                self.tray.set_state("error", "Model load failed")

        threading.Thread(target=preload, daemon=True).start()

        # Set up signal handlers
        def shutdown(signum, frame):
            logger.info("Shutting down...")
            self.listener.stop()
            self.socket_listener.stop()
            self.tray.stop()
            sys.exit(0)

        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)

        logger.info("LibreVoice daemon running. Hold Right Ctrl to dictate.")
        if self.config["notifications"]:
            notify("LibreVoice", "Daemon started. Hold Right Ctrl to dictate.")

        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            shutdown(None, None)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    daemon = LibreVoiceDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
