#!/usr/bin/env python3
"""
Standalone FLUX text-to-image runner (Hugging Face Diffusers).

默认模型: black-forest-labs/FLUX.1-schnell（Apache-2.0，一般无需门禁；首次会下载约数 GB 权重）。

可选 FLUX.1-dev：需在 https://huggingface.co/black-forest-labs/FLUX.1-dev 同意条款，
并设置环境变量 HF_TOKEN（或 huggingface-cli login）。

依赖（建议在项目 venv 中）:
  pip install torch diffusers transformers accelerate sentencepiece protobuf safetensors

用法示例:
  export HF_TOKEN=hf_xxx   # 仅 dev / 门禁模型需要
  python flux_image_runner.py --prompt "a red sports car on a mountain road" -o out.png
  python flux_image_runner.py --variant dev --prompt "..." --steps 28 --guidance 3.5
"""

from __future__ import annotations

import argparse
import os
import sys


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pick_dtype(device: str):
    import torch

    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def main() -> int:
    parser = argparse.ArgumentParser(description="FLUX 文生图（Diffusers FluxPipeline）")
    parser.add_argument(
        "--model",
        default=None,
        help="Hugging Face 模型 ID（覆盖 --variant）",
    )
    parser.add_argument(
        "--variant",
        choices=("schnell", "dev"),
        default="schnell",
        help="schnell: 快速、Apache-2.0；dev: 质量更高、需 HF 门禁与条款",
    )
    parser.add_argument("--prompt", "-p", required=True, help="正向提示词")
    parser.add_argument("--output", "-o", default="flux_out.png", help="输出 PNG 路径")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=None, help="推理步数（默认：schnell=4, dev=28）")
    parser.add_argument(
        "--guidance",
        type=float,
        default=None,
        help="guidance_scale（默认：schnell=0.0, dev=3.5）",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-cpu-offload",
        action="store_true",
        help="禁用 accelerate CPU offload（显存足够时可略快）",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Hugging Face 缓存目录（默认使用 HF_HOME / 系统默认）",
    )
    args = parser.parse_args()

    model_id = args.model
    if not model_id:
        model_id = (
            "black-forest-labs/FLUX.1-schnell"
            if args.variant == "schnell"
            else "black-forest-labs/FLUX.1-dev"
        )

    if args.variant == "dev" and not os.environ.get("HF_TOKEN"):
        print(
            "提示: FLUX.1-dev 为门禁模型，请先在网页同意条款并设置 HF_TOKEN，或改用 --variant schnell。",
            file=sys.stderr,
        )

    steps = args.steps
    guidance = args.guidance
    if steps is None:
        steps = 4 if "schnell" in model_id.lower() else 28
    if guidance is None:
        guidance = 0.0 if "schnell" in model_id.lower() else 3.5

    import torch
    from diffusers import FluxPipeline

    device = _pick_device()
    dtype = _pick_dtype(device)
    if device == "cpu":
        print("警告: FLUX 在 CPU 上极慢且易内存不足；建议使用 CUDA GPU 或 Apple Silicon（MPS）。", file=sys.stderr)

    token = os.environ.get("HF_TOKEN")
    load_kw = {
        "torch_dtype": dtype,
        "token": token,
    }
    if args.cache_dir:
        load_kw["cache_dir"] = args.cache_dir

    print(f"加载模型 {model_id} | device={device} | dtype={dtype} ...", flush=True)
    pipe = FluxPipeline.from_pretrained(model_id, **load_kw)

    if not args.no_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    gen = torch.Generator(device=device)
    if args.seed is not None:
        gen.manual_seed(args.seed)

    print(f"生成: {args.width}x{args.height}, steps={steps}, guidance={guidance} ...", flush=True)
    out = pipe(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=gen,
    )
    image = out.images[0]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    image.save(args.output)
    print(f"已保存: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
