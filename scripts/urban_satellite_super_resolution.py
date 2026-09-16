"""Convenience launcher for the integrated urban-satellite research CLI."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.pop(1) if len(sys.path) > 1 and sys.path[1] == str(Path(__file__).parent) else None

from urban_satellite_super_resolution.cli import main


if __name__ == "__main__":
    main()
