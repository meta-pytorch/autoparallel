from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor


def _canonical_name(name: str) -> str:
    prefixes = ("model._orig_mod.", "_orig_mod.", "model.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name.removeprefix(prefix)
                changed = True
    return name


@torch.no_grad()
def _write_post_load_audit(model_parts: list[torch.nn.Module], output: Path) -> None:
    records = []
    for part_index, model_part in enumerate(model_parts):
        for name, parameter in model_part.named_parameters():
            local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
            cpu_local = local.detach().cpu()
            flat = cpu_local.reshape(-1)
            sample = flat[: min(flat.numel(), 8192)].contiguous()
            sample_bytes = sample.view(torch.uint8).numpy().tobytes()
            float_local = cpu_local.double()
            records.append(
                {
                    "stage": "post_load",
                    "part": part_index,
                    "name": _canonical_name(name),
                    "raw_name": name,
                    "global_shape": list(parameter.shape),
                    "local_shape": list(local.shape),
                    "dtype": str(parameter.dtype),
                    "sample_sha256": hashlib.sha256(sample_bytes).hexdigest(),
                    "sum": float(float_local.sum().item()),
                    "square_sum": float(float_local.square().sum().item()),
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def register_post_load_parameter_audit(
    _optimizers, model_parts: list[torch.nn.Module], _parallel_dims
) -> None:
    """Write one parameter audit after every model part loads its checkpoint."""
    audit_dir = os.environ.get("PARAMETER_AUDIT_DIR")
    if audit_dir is None:
        return
    if not model_parts:
        raise RuntimeError("parameter-state audit requires at least one model part")

    loaded_parts: set[int] = set()
    handles = []

    def post_load(part_index: int):
        def hook(_module, _incompatible_keys) -> None:
            loaded_parts.add(part_index)
            if len(loaded_parts) != len(model_parts):
                return
            for handle in handles:
                handle.remove()
            output = Path(audit_dir) / f"rank_{int(os.environ['RANK']):02d}.json"
            _write_post_load_audit(model_parts, output)

        return hook

    for part_index, model_part in enumerate(model_parts):
        handles.append(
            model_part.register_load_state_dict_post_hook(post_load(part_index))
        )
