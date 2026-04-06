#!/usr/bin/env bash
# 在 comfy_ppt_agent 目录下安装 ComfyUI 与本 agent 的 Python 依赖（两个 venv 分离）
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "==> ComfyUI venv: $DIR/ComfyUI/.venv"
python3 -m venv ComfyUI/.venv
ComfyUI/.venv/bin/pip install -U pip wheel
ComfyUI/.venv/bin/pip install -r ComfyUI/requirements.txt

echo "==> comfy_ppt_agent venv: $DIR/.venv"
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt

echo ""
echo "请将 SD1.5 权重放入: ComfyUI/models/checkpoints/"
echo "  默认文件名: v1-5-pruned-emaonly.safetensors"
echo "  下载: https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main"
echo ""
echo "启动服务: ./run_comfy_server.sh"
echo "生成 PPT:  .venv/bin/python comfy_ppt_agent.py"
