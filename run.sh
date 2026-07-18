#!/bin/bash
# LibreVoice daemon wrapper - activates venv and runs the daemon
set -e

# Resolve to the actual directory this script lives in
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

# Do not activate the virtual environment in this shell.  Calling its Python
# directly is both simpler and avoids accidentally picking up system packages.
#
# A user systemd service gets its desktop-session variables from the unit.  In
# particular, do not invent DISPLAY/WAYLAND_DISPLAY here: doing so can connect
# pystray to the wrong display and makes injection fail on an X11 session.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export PYTHONUNBUFFERED=1

# Run the daemon
exec "${VENV_DIR}/bin/python3" "${SCRIPT_DIR}/daemon.py" "$@"
