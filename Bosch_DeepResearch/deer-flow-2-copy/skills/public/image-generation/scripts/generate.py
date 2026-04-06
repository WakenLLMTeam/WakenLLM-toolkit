import json
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image


def validate_image(image_path: str) -> bool:
    """
    Validate if an image file can be opened and is not corrupted.

    Args:
        image_path: Path to the image file

    Returns:
        True if the image is valid and can be opened, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img.load()
        return True
    except Exception as e:
        print(f"Warning: Image '{image_path}' is invalid or corrupted: {e}")
        return False


def parse_prompt(prompt_file: str) -> str:
    """
    Parse prompt from JSON or text file.

    Args:
        prompt_file: Path to prompt file (JSON or plain text)

    Returns:
        Prompt string
    """
    with open(prompt_file, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        data = json.loads(content)
        if isinstance(data, dict):
            if "prompt" in data:
                return data["prompt"]
            if "characters" in data and data["characters"]:
                return data.get("prompt", json.dumps(data["characters"][0], ensure_ascii=False))
            return json.dumps(data, ensure_ascii=False)
        return str(data)
    except json.JSONDecodeError:
        return content


def get_dimensions_from_aspect_ratio(aspect_ratio: str, base_size: int = 1024) -> tuple[int, int]:
    """
    Convert aspect ratio to pixel dimensions.

    Args:
        aspect_ratio: Format like "16:9", "4:3", "1:1", "2:3", etc.
        base_size: Base dimension (default 1024)

    Returns:
        Tuple of (width, height) both divisible by 64 for SDXL
    """
    try:
        w_str, h_str = aspect_ratio.split(":")
        w_ratio, h_ratio = float(w_str), float(h_str)
    except (ValueError, AttributeError):
        w_ratio, h_ratio = 16, 9

    if w_ratio >= h_ratio:
        width = base_size
        height = int(base_size * h_ratio / w_ratio)
    else:
        height = base_size
        width = int(base_size * w_ratio / h_ratio)

    width = (width // 64) * 64
    height = (height // 64) * 64

    return width, height


def generate_image(
    prompt_file: str,
    reference_images: list[str],
    output_file: str,
    aspect_ratio: str = "16:9",
) -> str:
    """
    Generate image using Stable Diffusion XL (local, free).

    Args:
        prompt_file: Path to JSON or text prompt file
        reference_images: List of reference image paths (optional)
        output_file: Output image file path
        aspect_ratio: Aspect ratio like "16:9", "4:3", etc.

    Returns:
        Status message
    """
    prompt = parse_prompt(prompt_file)

    if not prompt or not prompt.strip():
        raise ValueError("Prompt is empty after parsing")

    valid_reference_images = []
    for ref_img in reference_images:
        if validate_image(ref_img):
            valid_reference_images.append(ref_img)
        else:
            print(f"Skipping invalid reference image: {ref_img}")

    if len(valid_reference_images) < len(reference_images):
        print(
            f"Note: {len(reference_images) - len(valid_reference_images)} reference image(s) were skipped due to validation failure."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading Stable Diffusion XL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
        variant="fp16" if device == "cuda" else None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    if device == "cpu":
        pipe.enable_sequential_cpu_offload()
        num_steps = 15
    else:
        num_steps = 20

    width, height = get_dimensions_from_aspect_ratio(aspect_ratio)
    print(f"Generating image {width}x{height} from prompt: {prompt[:80]}...")

    try:
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_steps,
                guidance_scale=7.5,
            ).images[0]

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_file, quality=95)

        return f"Successfully generated image to {output_file}"
    finally:
        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate images using Gemini API")
    parser.add_argument(
        "--prompt-file",
        required=True,
        help="Absolute path to JSON prompt file",
    )
    parser.add_argument(
        "--reference-images",
        nargs="*",
        default=[],
        help="Absolute paths to reference images (space-separated)",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Output path for generated image",
    )
    parser.add_argument(
        "--aspect-ratio",
        required=False,
        default="16:9",
        help="Aspect ratio of the generated image",
    )

    args = parser.parse_args()

    try:
        print(
            generate_image(
                args.prompt_file,
                args.reference_images,
                args.output_file,
                args.aspect_ratio,
            )
        )
    except Exception as e:
        print(f"Error while generating image: {e}")
