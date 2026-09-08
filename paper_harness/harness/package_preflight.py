from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .runtime import _audit_runtime_configs, _expand
from .runtime_preflight import run as run_preflight


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m harness.package_preflight PAYLOAD OUTPUT")
    payload = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    resolved = json.loads((payload / "campaign/resolved_campaign.json").read_text())
    base_env = dict(os.environ)
    base_env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [
                    str(payload / "harness_repo"),
                    str(payload / "autoparallel"),
                    str(payload / "torchtitan"),
                ]
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": str(resolved["world_size"]),
            "LOCAL_WORLD_SIZE": str(resolved["mast"]["nproc_per_node"]),
        }
    )
    base_env.update(
        {
            key: _expand(str(value), payload=payload, output=output)
            for key, value in resolved["mast"].get("environment", {}).items()
        }
    )
    run_preflight(payload, output)
    _audit_runtime_configs(payload, resolved, output, base_env)


if __name__ == "__main__":
    main()
