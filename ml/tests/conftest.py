"""Make `ml/src` importable so these tests exercise the REAL preprocessing functions."""

import sys
from pathlib import Path

_ROOTS = ("data", "features", "evaluation")

for _name in _ROOTS:
    _candidates = [Path("/mlsrc") / _name, Path(__file__).resolve().parents[1] / "src" / _name]
    for _path in _candidates:
        if _path.is_dir():
            sys.path.insert(0, str(_path))
            break
    else:  # pragma: no cover - configuration error, and it must be loud rather than a silent skip
        raise RuntimeError(
            f"ml/src/{_name} not found in any of {[str(p) for p in _candidates]}. These tests exist "
            "to check the REAL modules; skipping them quietly would recreate the defect they were "
            "written to remove."
        )
