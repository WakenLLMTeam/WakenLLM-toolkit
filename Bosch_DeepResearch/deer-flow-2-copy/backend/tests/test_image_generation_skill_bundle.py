"""Regression tests for `skills/public/image-generation` and `skills/public/ppt-generation`.

These tests do **not** run the full PPT workflow: they only verify discovery, frontmatter,
parsing, container paths, and bytecode compilation — same pattern as `test_pptx_skill_bundle.py`.

**Intended AI slide-deck pipeline (document contract):**

1. ``image-generation`` — generate each slide image (ppt-generation skill defers to it).
2. ``ppt-generation`` — plan, orchestrate sequential images, then **compose** the final
   ``.pptx`` via ``ppt-generation/scripts/generate.py`` (documented in that skill's SKILL.md).

The ``pptx`` skill (covered by ``test_pptx_skill_bundle.py``) is for general read/edit/unpack
workflows on ``.pptx`` files; it is **not** a mandatory third step after ppt-generation for
this image-to-deck flow. Those tests stay in a separate module on purpose.
"""

from __future__ import annotations

import compileall
import sys
from pathlib import Path

import pytest

from deerflow.skills.loader import get_skills_root_path, load_skills
from deerflow.skills.parser import parse_skill_file
from deerflow.skills.validation import _validate_skill_frontmatter

IMAGE_GEN_REL = Path("public/image-generation")
PPT_GEN_REL = Path("public/ppt-generation")


def _skill_dir(rel: Path) -> Path:
    return get_skills_root_path() / rel


@pytest.fixture(scope="module")
def image_gen_dir() -> Path:
    p = _skill_dir(IMAGE_GEN_REL)
    if not (p / "SKILL.md").is_file():
        pytest.skip("image-generation skill missing")
    return p


@pytest.fixture(scope="module")
def ppt_gen_dir() -> Path:
    p = _skill_dir(PPT_GEN_REL)
    if not (p / "SKILL.md").is_file():
        pytest.skip("ppt-generation skill missing")
    return p


def test_image_generation_skill_files_present(image_gen_dir: Path) -> None:
    assert (image_gen_dir / "SKILL.md").is_file()
    assert (image_gen_dir / "scripts" / "generate.py").is_file()


def test_image_generation_skill_documents_sandbox_paths(image_gen_dir: Path) -> None:
    text = (image_gen_dir / "SKILL.md").read_text(encoding="utf-8")
    assert "/mnt/skills/public/image-generation" in text
    assert "/mnt/user-data/workspace" in text
    assert "/mnt/user-data/outputs" in text


def test_image_generation_frontmatter_validates(image_gen_dir: Path) -> None:
    ok, message, name = _validate_skill_frontmatter(image_gen_dir)
    assert ok is True, message
    assert name == "image-generation"


def test_image_generation_loads_with_container_path(image_gen_dir: Path) -> None:
    skills = load_skills(skills_path=get_skills_root_path(), use_config=False, enabled_only=False)
    by_name = {s.name: s for s in skills}
    assert "image-generation" in by_name
    s = by_name["image-generation"]
    assert s.category == "public"
    assert s.skill_dir.resolve() == image_gen_dir.resolve()
    assert s.get_container_file_path() == "/mnt/skills/public/image-generation/SKILL.md"


def test_image_generation_parses_from_disk(image_gen_dir: Path) -> None:
    skill_file = image_gen_dir / "SKILL.md"
    parsed = parse_skill_file(skill_file, "public", relative_path=Path("image-generation"))
    assert parsed is not None
    assert parsed.name == "image-generation"


def test_image_generation_scripts_compile(image_gen_dir: Path) -> None:
    scripts = image_gen_dir / "scripts"
    ok = compileall.compile_dir(str(scripts), quiet=1, ddir=str(scripts))
    assert ok is True, "compileall failed for image-generation/scripts"


def test_ppt_generation_skill_files_present(ppt_gen_dir: Path) -> None:
    assert (ppt_gen_dir / "SKILL.md").is_file()
    assert (ppt_gen_dir / "scripts" / "generate.py").is_file()


def test_ppt_generation_skill_documents_full_pipeline(ppt_gen_dir: Path) -> None:
    """SKILL.md must document: depends on image-generation; final .pptx via ppt-generation's compose script."""
    text = (ppt_gen_dir / "SKILL.md").read_text(encoding="utf-8")
    assert "image-generation" in text
    assert "/mnt/skills/public/ppt-generation" in text or "ppt-generation" in text
    assert "ppt-generation/scripts/generate.py" in text
    assert "/mnt/skills/public/image-generation" in text


def test_ppt_generation_frontmatter_validates(ppt_gen_dir: Path) -> None:
    ok, message, name = _validate_skill_frontmatter(ppt_gen_dir)
    assert ok is True, message
    assert name == "ppt-generation"


def test_ppt_generation_loads_with_container_path(ppt_gen_dir: Path) -> None:
    skills = load_skills(skills_path=get_skills_root_path(), use_config=False, enabled_only=False)
    by_name = {s.name: s for s in skills}
    assert "ppt-generation" in by_name
    s = by_name["ppt-generation"]
    assert s.get_container_file_path() == "/mnt/skills/public/ppt-generation/SKILL.md"


def test_ppt_generation_scripts_compile(ppt_gen_dir: Path) -> None:
    scripts = ppt_gen_dir / "scripts"
    ok = compileall.compile_dir(str(scripts), quiet=1, ddir=str(scripts))
    assert ok is True, "compileall failed for ppt-generation/scripts"
