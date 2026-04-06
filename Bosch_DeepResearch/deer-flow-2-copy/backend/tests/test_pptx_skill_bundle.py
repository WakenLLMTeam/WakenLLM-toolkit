"""Regression tests for the bundled Anthropic-style `pptx` skill under skills/public/pptx.

How this is used in DeerFlow (not executed by these tests — documented for operators):

1. **Discovery**: `load_skills()` walks `skills/public/` and `skills/custom/`, parses each
   `SKILL.md` frontmatter, and merges enable flags from `extensions_config.json` (empty
   `skills` object means all public skills default to enabled).

2. **Prompt**: The lead agent’s system prompt lists enabled skills with name, description,
   and the container path to `SKILL.md` (default `/mnt/skills/public/pptx/SKILL.md`). The
   model is expected to read that file when the user works with `.pptx` files.

3. **Execution**: In the local sandbox, the same tree is mounted at `/mnt/skills/public/pptx`.
   Shell commands in the skill docs assume `cd /mnt/skills/public/pptx` before
   `python scripts/...`. User outputs should go under `/mnt/user-data/workspace` (or
   uploads/outputs). Runtime deps (markitdown, Pillow, npm pptxgenjs, LibreOffice, poppler)
   must exist in the environment — this suite only checks layout and load/parse/compile.
"""

from __future__ import annotations

import compileall
import subprocess
import sys
from pathlib import Path

import pytest

from deerflow.skills.loader import get_skills_root_path, load_skills
from deerflow.skills.parser import parse_skill_file
from deerflow.skills.validation import _validate_skill_frontmatter

PPTX_SKILL_REL = Path("public/pptx")


def _pptx_skill_dir() -> Path:
    root = get_skills_root_path()
    return root / PPTX_SKILL_REL


@pytest.fixture(scope="module")
def pptx_dir() -> Path:
    path = _pptx_skill_dir()
    if not (path / "SKILL.md").is_file():
        pytest.skip("Bundled pptx skill missing (skills/public/pptx/SKILL.md)")
    return path


def test_pptx_skill_files_present(pptx_dir: Path) -> None:
    """Core docs and helper scripts from the upstream bundle are on disk."""
    assert (pptx_dir / "SKILL.md").is_file()
    assert (pptx_dir / "LICENSE.txt").is_file()
    assert (pptx_dir / "editing.md").is_file()
    assert (pptx_dir / "pptxgenjs.md").is_file()
    for rel in (
        "scripts/thumbnail.py",
        "scripts/clean.py",
        "scripts/add_slide.py",
        "scripts/office/unpack.py",
        "scripts/office/pack.py",
        "scripts/office/soffice.py",
        "scripts/office/validate.py",
    ):
        assert (pptx_dir / rel).is_file(), f"missing {rel}"


def test_pptx_skill_markdown_documents_deerflow_paths(pptx_dir: Path) -> None:
    """Our adaptation should point agents at the sandbox mount and workspace."""
    text = (pptx_dir / "SKILL.md").read_text(encoding="utf-8")
    assert "DeerFlow (sandbox paths)" in text
    assert "/mnt/skills/public/pptx" in text
    assert "/mnt/user-data/workspace" in text


def test_pptx_skill_frontmatter_validates(pptx_dir: Path) -> None:
    ok, message, name = _validate_skill_frontmatter(pptx_dir)
    assert ok is True, message
    assert message == "Skill is valid!"
    assert name == "pptx"


def test_pptx_skill_loads_with_expected_container_paths(pptx_dir: Path) -> None:
    skills = load_skills(skills_path=get_skills_root_path(), use_config=False, enabled_only=False)
    by_name = {s.name: s for s in skills}
    assert "pptx" in by_name
    skill = by_name["pptx"]
    assert skill.category == "public"
    assert skill.skill_dir.resolve() == pptx_dir.resolve()
    assert skill.get_container_path() == "/mnt/skills/public/pptx"
    assert skill.get_container_file_path() == "/mnt/skills/public/pptx/SKILL.md"
    assert ".pptx" in skill.description


def test_pptx_skill_parses_from_disk(pptx_dir: Path) -> None:
    skill_file = pptx_dir / "SKILL.md"
    # Same relative_path the loader uses for public/pptx/SKILL.md
    parsed = parse_skill_file(skill_file, "public", relative_path=Path("pptx"))
    assert parsed is not None
    assert parsed.name == "pptx"
    assert parsed.category == "public"
    assert parsed.license is not None
    assert "proprietary" in parsed.license.lower() or "license" in parsed.license.lower()


def test_pptx_scripts_bytecode_compile(pptx_dir: Path) -> None:
    """Syntax-check bundled Python without importing third-party deps at import time."""
    scripts_dir = pptx_dir / "scripts"
    ok = compileall.compile_dir(str(scripts_dir), quiet=1, ddir=str(scripts_dir))
    assert ok is True, "compileall failed for skills/public/pptx/scripts"


def test_pptx_thumbnail_cli_exits_when_file_missing(pptx_dir: Path) -> None:
    """Smoke-test running the thumbnail entrypoint from the skill scripts directory."""
    scripts = pptx_dir / "scripts"
    code = """
import runpy
import sys
sys.argv = ["thumbnail.py", "__definitely_missing__.pptx"]
try:
    runpy.run_path(%r, run_name="__main__")
except SystemExit as e:
    sys.exit(e.code)
""" % str(scripts / "thumbnail.py")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(scripts),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 0
