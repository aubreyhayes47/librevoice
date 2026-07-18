#!/usr/bin/env bash
# Install and start LibreVoice as a per-user service.  No sudo required.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
BIN_DIR="${HOME}/.local/bin"

mkdir -p "$UNIT_DIR" "$BIN_DIR"
# The stable per-user entrypoint keeps the systemd unit independent of where
# the repository is cloned; run.sh resolves the symlink back to this checkout.
ln -sfn "$SCRIPT_DIR/run.sh" "$BIN_DIR/librevoice-daemon"
cp "$SCRIPT_DIR/systemd/ydotoold.service" "$UNIT_DIR/ydotoold.service"
cp "$SCRIPT_DIR/systemd/librevoice.service" "$UNIT_DIR/librevoice.service"
systemctl --user daemon-reload
systemctl --user enable ydotoold.service librevoice.service
# `enable --now` does not reload an already-running service.  Restart both so
# rerunning this installer always applies the current project files.
systemctl --user restart ydotoold.service
systemctl --user restart librevoice.service
systemctl --user status librevoice.service --no-pager
