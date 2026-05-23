#!/usr/bin/env bash

set -euo pipefail

HOST="${1:-Server-38-CHY}"
LOCAL_PROXY_PORT="${LOCAL_PROXY_PORT:-7897}"
REMOTE_PROXY_PORT="${REMOTE_PROXY_PORT:-7897}"
SOCK_DIR="${HOME}/.ssh/agent"
mkdir -p "$SOCK_DIR"
SOCK_HASH="$(printf '%s' "${USER}@${HOST}:${LOCAL_PROXY_PORT}:${REMOTE_PROXY_PORT}" | shasum | cut -c1-12)"
SOCK_PATH="${SOCK_DIR%/}/cm-${SOCK_HASH}.sock"
REMOTE_SETUP_BLOCK="# >>> codex-remote-proxy >>>"
REMOTE_END_BLOCK="# <<< codex-remote-proxy <<<"

cleanup() {
  ssh -S "$SOCK_PATH" -O exit "$HOST" >/dev/null 2>&1 || true
  rm -f "$SOCK_PATH"
}

trap cleanup EXIT

if ! command -v ssh >/dev/null 2>&1; then
  echo "ssh is required but not installed." >&2
  exit 1
fi

if ! command -v tar >/dev/null 2>&1; then
  echo "tar is required but not installed." >&2
  exit 1
fi

if [ ! -d "$HOME/.codex" ]; then
  echo "~/.codex does not exist on this machine." >&2
  exit 1
fi

echo "Opening SSH control connection to ${HOST}..."
ssh -M -S "$SOCK_PATH" -o ControlPersist=yes -fN "$HOST"

echo "Checking remote home directory..."
ssh -S "$SOCK_PATH" "$HOST" 'echo "Remote host: $(hostname)"; echo "Remote user: $(whoami)"; printf "Remote home: "; printf "%s\n" "$HOME"'

echo "Backing up remote ~/.codex if needed and syncing local ~/.codex..."
ssh -S "$SOCK_PATH" "$HOST" 'if [ -d "$HOME/.codex" ]; then mv "$HOME/.codex" "$HOME/.codex.bak.$(date +%Y%m%d%H%M%S)"; fi'
tar czf - -C "$HOME" .codex | ssh -S "$SOCK_PATH" "$HOST" 'tar xzf - -C "$HOME"'

echo "Writing remote proxy exports into shell startup files..."
ssh -S "$SOCK_PATH" "$HOST" \
  "REMOTE_PROXY_PORT='$REMOTE_PROXY_PORT' REMOTE_SETUP_BLOCK='$REMOTE_SETUP_BLOCK' REMOTE_END_BLOCK='$REMOTE_END_BLOCK' bash -s" <<'EOF'
set -euo pipefail

update_rc() {
  local rc_file="$1"
  [ -f "$rc_file" ] || : >"$rc_file"
  python3 - "$rc_file" "$REMOTE_SETUP_BLOCK" "$REMOTE_END_BLOCK" "$REMOTE_PROXY_PORT" <<'PY'
from pathlib import Path
import sys

rc_path = Path(sys.argv[1])
start = sys.argv[2]
end = sys.argv[3]
port = sys.argv[4]
block = "\n".join(
    [
        start,
        f'export http_proxy="http://127.0.0.1:{port}"',
        f'export https_proxy="http://127.0.0.1:{port}"',
        f'export HTTP_PROXY="http://127.0.0.1:{port}"',
        f'export HTTPS_PROXY="http://127.0.0.1:{port}"',
        f'export ALL_PROXY="http://127.0.0.1:{port}"',
        f'export all_proxy="http://127.0.0.1:{port}"',
        end,
        "",
    ]
)
text = rc_path.read_text() if rc_path.exists() else ""
while start in text and end in text:
    start_idx = text.index(start)
    end_idx = text.index(end, start_idx) + len(end)
    if end_idx < len(text) and text[end_idx] == "\n":
      end_idx += 1
    text = text[:start_idx] + text[end_idx:]
text = text.rstrip() + "\n\n" + block
rc_path.write_text(text)
PY
}

update_rc "$HOME/.bashrc"
update_rc "$HOME/.zshrc"
EOF

echo "Ensuring remote port forwarding is active..."
ssh -S "$SOCK_PATH" -O forward -R "${REMOTE_PROXY_PORT}:127.0.0.1:${LOCAL_PROXY_PORT}" "$HOST"

echo "Smoke testing the remote proxy..."
ssh -S "$SOCK_PATH" "$HOST" \
  "http_proxy='http://127.0.0.1:${REMOTE_PROXY_PORT}' https_proxy='http://127.0.0.1:${REMOTE_PROXY_PORT}' curl -I --max-time 15 https://api.openai.com >/tmp/codex_remote_curl.out && sed -n '1,5p' /tmp/codex_remote_curl.out"

cat <<MSG

Remote Codex setup is ready for ${HOST}.

What changed:
  - Synced local ~/.codex to the remote home directory
  - Added proxy exports to ~/.bashrc and ~/.zshrc on the remote machine
  - Started reverse SSH forwarding: remote 127.0.0.1:${REMOTE_PROXY_PORT} -> local 127.0.0.1:${LOCAL_PROXY_PORT}

Next step:
  - Reconnect your remote IDE session, then open Codex there.

This tunnel stays up while the SSH control connection is alive.
MSG
