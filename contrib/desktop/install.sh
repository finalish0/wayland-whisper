#!/bin/sh
# Install wayland-whisper as a user service plus Sway toggle (no root needed).
#
# Prerequisites: the repo's .venv exists (python -m venv .venv && .venv/bin/pip install -e .)
# and the model files are in place (defaults below).
set -eu

REPO=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
BIN="${BIN:-$HOME/.local/bin}"

install -Dm755 "$REPO/contrib/desktop/wayland-whisper-toggle" "$BIN/wayland-whisper-toggle"
install -Dm644 "$REPO/contrib/desktop/wayland-whisper.service" \
  "$HOME/.config/systemd/user/wayland-whisper.service"
systemctl --user daemon-reload

cat <<EOF
installed:
  $BIN/wayland-whisper-toggle
  ~/.config/systemd/user/wayland-whisper.service

add to ~/.config/sway/config (once):
  bindsym \$mod+n exec --no-startup-id $BIN/wayland-whisper-toggle master
  include ~/.config/sway/wayland-whisper.conf

then:
  swaymsg reload && $BIN/wayland-whisper-toggle on
EOF
