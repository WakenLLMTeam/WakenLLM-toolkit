#!/usr/bin/env bash
# 启动 ComfyUI HTTP API（默认 127.0.0.1:8188）
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR/ComfyUI"
if [[ ! -x .venv/bin/python ]]; then
  echo "请先运行: $DIR/install_comfyui.sh" >&2
  exit 1
fi
exec .venv/bin/python main.py --listen 127.0.0.1 --port 8188 "$@"
