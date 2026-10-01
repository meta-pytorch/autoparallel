"""Build the TorchTitan config for one campaign phase and arm.

The runtime launches ``python -m torchtitan.train --module harness.tt_config
--config resolved``. ``resolved()`` reads ``HARNESS_TT_CONFIG``, a JSON object
with the workload factory ``module``/``config`` and the campaign's dotted
``settings``. Settings are applied with ``dataclasses.replace`` from the leaves
up, so every config's ``__post_init__`` validation runs on the final values.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
import types
import typing
from typing import Any

ENVIRONMENT_KEY = "HARNESS_TT_CONFIG"


def resolved():
    return build(json.loads(os.environ[ENVIRONMENT_KEY]))


def build(spec: dict[str, Any]):
    module = importlib.import_module(spec["module"])
    return apply_settings(getattr(module, spec["config"])(), spec["settings"])


def apply_settings(config, settings: dict[str, Any]):
    tree: dict[str, Any] = {}
    for path, value in settings.items():
        pieces = path.split(".")
        if any(not piece for piece in pieces):
            raise ValueError(f"invalid TorchTitan setting path {path!r}")
        node = tree
        for piece in pieces[:-1]:
            node = node.setdefault(piece, {})
            if not isinstance(node, dict):
                raise ValueError(f"TorchTitan setting {path!r} crosses a value")
        if pieces[-1] in node:
            raise ValueError(f"TorchTitan setting {path!r} overlaps another setting")
        node[pieces[-1]] = _Value(value)
    return _replace(config, tree, prefix="")


@dataclasses.dataclass(frozen=True)
class _Value:
    value: Any


def _replace(config, tree: dict[str, Any], *, prefix: str):
    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        raise ValueError(f"TorchTitan setting {prefix[:-1]!r} is not a config")
    fields = {field.name for field in dataclasses.fields(config) if field.init}
    hints = typing.get_type_hints(type(config))
    changes = {}
    for name, child in tree.items():
        path = prefix + name
        if name not in fields:
            raise ValueError(f"unknown TorchTitan setting {path!r}")
        if isinstance(child, _Value):
            if not _matches(child.value, hints[name]):
                raise TypeError(
                    f"TorchTitan setting {path!r} expects {hints[name]}, "
                    f"got {child.value!r}"
                )
            changes[name] = child.value
            continue
        current = getattr(config, name)
        if current is None:
            raise ValueError(f"TorchTitan setting {path!r} is unset in the recipe")
        changes[name] = _replace(current, child, prefix=f"{path}.")
    return dataclasses.replace(config, **changes)


def _matches(value: Any, annotation: Any) -> bool:
    origin = typing.get_origin(annotation)
    if annotation is type(None):
        return value is None
    if annotation is bool:
        return isinstance(value, bool)
    if annotation in (int, float, str):
        return type(value) is annotation
    if origin is typing.Literal:
        return any(
            type(value) is type(choice) and value == choice
            for choice in typing.get_args(annotation)
        )
    if origin in (typing.Union, types.UnionType):
        return any(_matches(value, choice) for choice in typing.get_args(annotation))
    if origin is list:
        (item,) = typing.get_args(annotation)
        return isinstance(value, list) and all(_matches(entry, item) for entry in value)
    raise TypeError(f"unsupported TorchTitan setting type {annotation!r}")
