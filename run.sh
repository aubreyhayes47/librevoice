#!/bin/bash
# LibreVoice daemon wrapper - activates venv and runs the daemon
set -e

# Resolve to the actual directory this script lives in
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

# Activate the virtual environment
source "${VENV_DIR}/bin/activate"

# Set environment variables
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export DISPLAY="${DISPLAY:-:0}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
export XDG_SESSION_TYPE="${XDG_SESSION_TYPE:-wayland}"

# Run the daemon
exec python3 "${SCRIPT_DIR}/daemon.py" "$@"
