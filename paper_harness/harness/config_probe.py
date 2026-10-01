from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from .tt_config import build


def serialized(value: Any) -> Any:
    """Serialize a TorchTitan config tree, recording each config's class."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result: dict[str, Any] = {
            "_class": f"{type(value).__module__}.{type(value).__qualname__}"
        }
        for field in dataclasses.fields(value):
            if not field.name.startswith("_"):
                result[field.name] = serialized(getattr(value, field.name))
        return result
    if isinstance(value, (list, tuple)):
        return [serialized(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialized(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    config = build(request["spec"])
    args.output.write_text(
        json.dumps(serialized(config), indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
