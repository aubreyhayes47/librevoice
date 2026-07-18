# LibreVoice

System-wide, local push-to-talk dictation for Linux. Hold **Right Ctrl**, speak,
and release it; LibreVoice transcribes the utterance with OpenVINO Whisper and
types the result into the focused application.

The working path is intentionally small:

```text
evdev hotkey -> warm microphone + pre-roll -> OpenVINO Whisper -> ydotool
```

LibreVoice has been verified on Ubuntu GNOME Wayland with Firefox and VS Code.

## Requirements

- Python 3.12 and `python3-venv`
- `ydotool` for Wayland text injection
- `wl-clipboard` for the clipboard fallback
- PortAudio (`libportaudio2`) for microphone capture
- An OpenVINO GenAI-compatible Whisper model export
- Membership in the `input` group so `evdev` can read the global hotkey

On Ubuntu:

```bash
sudo apt install python3-venv ydotool wl-clipboard libportaudio2
sudo usermod -aG input "$USER"
```

Log out and back in after adding the group.

## Setup

Create the environment and install the tested Python dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Place the model at:

```text
models/whisper-large-v3-turbo-fp16/
```

Alternatively, set `model_path` in `~/.config/librevoice/config.json` to any
compatible local OpenVINO Whisper export.

Install and start the per-user services:

```bash
./install-service.sh
```

The installer creates a stable `~/.local/bin/librevoice-daemon` symlink, so the
checkout can live anywhere. It enables both LibreVoice and `ydotoold`; no manual
`run.sh` invocation is needed after login.

## Usage

1. Focus a text field.
2. Hold **Right Ctrl**.
3. Speak naturally.
4. Release **Right Ctrl** and wait for the text to appear.

The tray colours show the runtime state:

- Gray: ready
- Green: recording
- Yellow: processing speech
- Blue: loading the model
- Red: error

GNOME's legacy XEmbed tray bridge does not expose pystray's window title as a
hover tooltip. The icon colour and journal are the reliable status indicators
on that desktop; dictation itself does not depend on the tray UI.

## Configuration

LibreVoice creates `~/.config/librevoice/config.json` on first run. User values
override these defaults:

```json
{
  "hotkey": "KEY_RIGHTCTRL",
  "mode": "hold",
  "model_path": "/path/to/librevoice/models/whisper-large-v3-turbo-fp16",
  "device": "GPU",
  "fallback_devices": ["CPU"],
  "language": "en",
  "max_duration_sec": 30,
  "sample_rate": 16000,
  "pre_roll_ms": 400,
  "typing_delay_ms": 2,
  "clipboard": true,
  "notifications": true,
  "log_level": "INFO"
}
```

The warm microphone keeps only `pre_roll_ms` of rolling audio in memory. This
prevents the first word from being clipped; audio is not written to disk.

Keep `typing_delay_ms` at `1` or higher for reliable Wayland injection. If the
GPU model cannot load, LibreVoice tries each configured fallback device.

## Service and troubleshooting

Check current status and follow the live processing log:

```bash
systemctl --user status librevoice.service ydotoold.service
journalctl --user -u librevoice.service -f
```

A successful utterance logs, in order:

```text
Hotkey pressed
Recording started with 400 ms pre-roll
Captured ...s of audio
Transcribed in ...s
Injected: ...
```

If the hotkey is not logged, verify `groups` contains `input`. If transcription
succeeds but text does not appear, check `ydotoold.service`; LibreVoice will
restart it automatically when its socket is missing.

## Development

Run the headless regression suite and syntax checks with:

```bash
PYSTRAY_BACKEND=dummy .venv/bin/python3 -m unittest discover -s tests -v
.venv/bin/python3 -m py_compile daemon.py librevoice benchmark.py
```

The tests cover Linux key transition values, microphone pre-roll, nonblocking
clipboard behavior, and ydotoold socket reuse.
