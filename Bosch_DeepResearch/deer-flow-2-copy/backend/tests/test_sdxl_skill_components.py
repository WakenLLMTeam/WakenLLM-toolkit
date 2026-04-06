"""Component tests for Stable Diffusion XL integration in `image-generation` skill.

- **Always run**: import smoke, `parse_prompt`, `get_dimensions_from_aspect_ratio`, CLI `--help` path.
- **Optional heavy test**: set `RUN_SDXL_INFERENCE=1` to load SDXL and generate one small image (minutes + disk).

This intentionally avoids the full agent / PPT orchestration workflow.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Skill script lives outside backend package; add to path for direct imports.
_SKILLS_ROOT = Path(__file__).resolve().parents[2] / "skills" / "public" / "image-generation" / "scripts"
assert _SKILLS_ROOT.name == "scripts"


@pytest.fixture(scope="module")
def generate_module():
    """Import `generate.py` from the skill (requires torch + diffusers in backend venv)."""
    if not _SKILLS_ROOT.is_dir():
        pytest.skip("image-generation scripts directory missing")
    sys.path.insert(0, str(_SKILLS_ROOT))
    try:
        import generate as gen  # type: ignore[import-not-found]

        return gen
    except ImportError as e:
        pytest.skip(f"Cannot import skill generate.py (deps?): {e}")


def test_sdxl_dependencies_import() -> None:
    """Smoke: torch + diffusers available (same stack as the skill)."""
    import torch
    from diffusers import StableDiffusionXLPipeline

    assert torch.__version__
    assert StableDiffusionXLPipeline is not None


def test_get_dimensions_from_aspect_ratio(generate_module) -> None:
    gen = generate_module
    w, h = gen.get_dimensions_from_aspect_ratio("16:9", base_size=1024)
    assert w % 64 == 0 and h % 64 == 0
    assert w == 1024
    assert h == 576

    w2, h2 = gen.get_dimensions_from_aspect_ratio("1:1", base_size=1024)
    assert (w2, h2) == (1024, 1024)


def test_parse_prompt_json_and_plain(tmp_path: Path, generate_module) -> None:
    gen = generate_module
    j = tmp_path / "p.json"
    j.write_text(json.dumps({"prompt": "a red circle on white"}), encoding="utf-8")
    assert gen.parse_prompt(str(j)) == "a red circle on white"

    t = tmp_path / "p.txt"
    t.write_text("plain text prompt", encoding="utf-8")
    assert gen.parse_prompt(str(t)) == "plain text prompt"


@pytest.mark.skipif(
    os.environ.get("RUN_SDXL_INFERENCE") != "1",
    reason="Set RUN_SDXL_INFERENCE=1 to run SDXL load + one image (slow, large download possible).",
)
def test_sdxl_generate_image_end_to_end(tmp_path: Path, generate_module) -> None:
    """Full stack: load SDXL pipeline and write one image. Only when explicitly enabled."""
    gen = generate_module
    prompt_file = tmp_path / "prompt.json"
    prompt_file.write_text(json.dumps({"prompt": "a simple blue gradient abstract background"}), encoding="utf-8")
    out = tmp_path / "out.jpg"

    msg = gen.generate_image(
        str(prompt_file),
        [],
        str(out),
        aspect_ratio="16:9",
    )
    assert "Successfully" in msg
    assert out.is_file()
    assert out.stat().st_size > 1000
    assert gen.validate_image(str(out))


def test_skill_cli_invocation_helpful_error_without_torch_subprocess() -> None:
    """Subprocess: run script with missing args → argparse error, not import crash."""
    script = _SKILLS_ROOT / "generate.py"
    if not script.is_file():
        pytest.skip("generate.py missing")
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(_SKILLS_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    # argparse exits 2 when required args missing
    assert proc.returncode != 0
    assert "--prompt-file" in proc.stderr or "prompt-file" in proc.stderr or "required" in proc.stderr.lower()
