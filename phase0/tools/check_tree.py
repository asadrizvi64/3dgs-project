"""Enforce the Phase 0 file manifest.

Exit 1 if a manifest file is missing, or if an unlisted file exists under csrc/, splat2d/
or tests/. Python bytecode caches (__pycache__/, *.pyc) are generated, not source, and
are ignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MANIFEST = (
    "csrc/macrotile/splat2d_common.h",
    "csrc/macrotile/cuda_shim.h",
    "csrc/macrotile/launch.h",
    "csrc/macrotile/splat2d_fwd.cu",
    "csrc/macrotile/splat2d_bwd_pixel.cu",
    "csrc/macrotile/splat2d_bwd_splat.cu",
    "csrc/macrotile/splat2d_cpu.cpp",
    "csrc/macrotile/bindings.cpp",
    "splat2d/__init__.py",
    "splat2d/ops.py",
    "tests/gradcheck/conftest.py",
    "tests/gradcheck/test_gradcheck.py",
    "tests/gradcheck/test_opcount.py",
    "tools/check_tree.py",
    "tools/mock_build.py",
    "pyproject.toml",
)
STRICT_DIRS = ("csrc", "splat2d", "tests")


def is_generated(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix == ".pyc"


def main() -> int:
    listed = set(MANIFEST)
    missing = [rel for rel in MANIFEST if not (ROOT / rel).is_file()]
    extra: list[str] = []
    for top in STRICT_DIRS:
        for path in sorted((ROOT / top).rglob("*")):
            if not path.is_file() or is_generated(path):
                continue
            rel = path.relative_to(ROOT).as_posix()
            if rel not in listed:
                extra.append(rel)
    for rel in missing:
        print(f"MISSING  {rel}")
    for rel in extra:
        print(f"UNLISTED {rel}")
    if missing or extra:
        return 1
    print(f"check_tree: OK ({len(MANIFEST)} manifest files, no unlisted files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
