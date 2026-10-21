"""
Road Network & Spatial Partitioning: Wrapper for Spatial Grid Blocks (`block_id`).
Redirects execution to `scripts.cross_validation.build_spatial_blocks`.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from scripts.cross_validation.build_spatial_blocks import main

if __name__ == "__main__":
    main()
