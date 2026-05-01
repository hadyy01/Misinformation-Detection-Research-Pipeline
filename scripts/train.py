from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a misinformation detection experiment from config.")
    parser.add_argument("--config", required=True, help="Path to a YAML or JSON config file.")
    args = parser.parse_args()

    from false_information_flagging.pipeline import run_experiment

    result = run_experiment(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
