from __future__ import annotations

import ast
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    ROOT / "app.py",
    ROOT / "requirements.txt",
    ROOT / "core" / "data_merger_core.py",
    ROOT / "sections" / "potato.py",
    ROOT / "sections" / "pizza.py",
    ROOT / "sections" / "borek.py",
    ROOT / "sections" / "smallcake.py",
    ROOT / "sections" / "pyrocam.py",
    ROOT / "sections" / "bread.py",
    ROOT / "sections" / "data_merger.py",
    ROOT / "sections" / "teflon_block.py",
    ROOT / "sections" / "cookie.py",
    ROOT / "sections" / "flour_disk.py",
    ROOT / "ui" / "home.py",
    ROOT / "ui" / "layout.py",
    ROOT / "ui" / "navigation.py",
]


def main() -> int:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.exists()]
    if missing:
        print("Missing files:")
        for item in missing:
            print(f"  - {item}")
        return 1

    python_files = sorted(ROOT.rglob("*.py"))
    failures = []
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:
            failures.append((path, exc))

    if failures:
        print("Syntax failures:")
        for path, exc in failures:
            print(f"  - {path.relative_to(ROOT)}: {exc}")
        return 1

    print(f"OK: {len(python_files)} Python files parsed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
