from __future__ import annotations

import argparse
import json
from pathlib import Path

from torchtitan.config import ConfigManager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    config = ConfigManager().parse_args(request["argv"])
    args.output.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
