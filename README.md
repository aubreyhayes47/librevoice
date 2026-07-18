# LibreVoice

System-wide push-to-talk voice dictation for Linux. By default, LibreVoice records while you hold the **right Ctrl** key, transcribes the audio with an OpenVINO Whisper model, and inserts the text at the current cursor.

## Ubuntu GNOME Wayland setup

LibreVoice works on GNOME Wayland by listening to keyboard events with `evdev` and injecting text with `ydotool`.

1. Install the system tools:

   ```bash
   sudo apt install ydotool wl-clipboard python3-venv
   ```

2. Allow your user to read keyboard input events:

   ```bash
   sudo usermod -aG input "$USER"
   ```

   Log out and back in after changing groups.

3. Start the `ydotool` daemon. On many Ubuntu systems this is available as a user service:

   ```bash
   systemctl --user enable --now ydotoold.service
   ```

   If that service is unavailable, start `ydotoold` using the packaging instructions for your Ubuntu release.

4. Run LibreVoice:

   ```bash
   ./run.sh
   ```

5. Hold **right Ctrl**, speak, then release **right Ctrl**. The transcribed text is also copied to the clipboard when `clipboard` is enabled.

## Configuration

The daemon creates `~/.config/librevoice/config.json` on first run. The defaults include:

```json
{
  "hotkey": "KEY_RIGHTCTRL",
  "mode": "hold",
  "socket_path": "/tmp/librevoice-trigger.sock",
  "clipboard": true
}
```

Useful options:

- `hotkey`: an `evdev.ecodes` key name, such as `KEY_RIGHTCTRL`.
- `mode`: `hold` records while the key is held; `toggle` starts and stops on repeated presses.
- `socket_path`: Unix socket path used by external trigger commands.

## GNOME custom shortcut fallback

GNOME custom keyboard shortcuts usually trigger commands on key press only, not key release, so they are not ideal for true hold-to-talk. If direct `evdev` hotkey listening is unavailable, set `"mode": "toggle"` and create a GNOME custom shortcut that runs:

```bash
/path/to/librevoice trigger toggle
```

Press the shortcut once to start recording and again to stop.

For integrations that can send both press and release events, use:

```bash
/path/to/librevoice trigger press
/path/to/librevoice trigger release
```
