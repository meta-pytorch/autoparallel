"""
Diagnose the CI _flatten cache miss.

Wraps DeviceMesh._flatten and DeviceMesh._create_flatten_mesh to log:
  - At every _flatten call: mesh identity, dim_names, root_mesh id,
    cache state, call site
  - At every _create_flatten_mesh entry: same info, plus whether the
    cache check at line 904 hit or missed

The goal is to figure out, on CI, which mesh is triggering the
`as_strided` dispatch failure: is it the user's top-level mesh (cache
should hit), or a sub-mesh / independently-constructed mesh?

Run with:
    PYTHONPATH=. python tests/diagnose_flatten_ci.py

This script reproduces the same path as the failing CI test
(test_save_all_partitioner_compile_with_ac_enabled) but with logging.
"""

import os
import traceback
from typing import Any

# Force device emulation matching CI before importing torch
from unittest.mock import patch

_PATCHES: list[Any] = [
    patch("torch.cuda.device_count", lambda: 8),
    patch("torch.cuda.get_device_name", lambda *a, **k: "NVIDIA A10G"),
    patch("torch.cuda.get_device_capability", lambda *a, **k: (8, 6)),
    patch(
        "torch.cuda.get_device_properties",
        lambda *a, **k: type(
            "Props",
            (),
            {
                "major": 8,
                "minor": 6,
                "name": "NVIDIA A10G",
                "total_memory": 24 * 1024**3,
                "multi_processor_count": 80,
            },
        )(),
    ),
]
for p in _PATCHES:
    p.start()

import torch  # noqa: E402
from torch.distributed.device_mesh import DeviceMesh  # noqa: E402
from torch.testing._internal.distributed.fake_pg import FakeStore  # noqa: E402

# --- Instrumentation ---

_log_lines: list[str] = []


def _log(msg: str) -> None:
    _log_lines.append(msg)
    print(msg, flush=True)


def _mesh_id(m) -> str:
    return f"id={id(m):#x}"


def _summarize(m) -> str:
    if m is None:
        return "None"
    dim_names = getattr(m, "_mesh_dim_names", None)
    ndim = getattr(m, "ndim", "?")
    root = m._get_root_mesh() if hasattr(m, "_get_root_mesh") else None
    is_root = root is m
    cache_keys = (
        list(root._flatten_mapping.keys())
        if root is not None and hasattr(root, "_flatten_mapping")
        else "<no _flatten_mapping>"
    )
    return (
        f"{_mesh_id(m)} ndim={ndim} dim_names={dim_names} "
        f"is_root={is_root} root_id={_mesh_id(root)} "
        f"root_cache_keys={cache_keys}"
    )


def _short_traceback() -> str:
    # Get the call site that invoked _flatten — skip the wrapper frames.
    stack = traceback.extract_stack()
    interesting = [
        f"  {f.filename}:{f.lineno} in {f.name}"
        for f in stack
        if "device_mesh" not in f.filename and "diagnose_flatten_ci" not in f.filename
    ]
    return "\n".join(interesting[-6:])


_orig_flatten = DeviceMesh._flatten
_orig_create = DeviceMesh._create_flatten_mesh

_call_count = {"flatten": 0, "create": 0, "create_miss": 0}


def _wrapped_flatten(self, mesh_dim_name=None, backend_override=None):
    _call_count["flatten"] += 1
    n = _call_count["flatten"]
    requested_name = mesh_dim_name or (
        "_".join(self._mesh_dim_names) if self._mesh_dim_names else "<no dim names>"
    )
    _log(
        f"\n[_flatten #{n}] CALL on mesh {_summarize(self)} "
        f"requested_name={requested_name!r}"
    )
    _log(f"[_flatten #{n}] call site:\n{_short_traceback()}")
    try:
        result = _orig_flatten(self, mesh_dim_name, backend_override)
        _log(f"[_flatten #{n}] OK → result {_summarize(result)}")
        return result
    except Exception as e:
        _log(f"[_flatten #{n}] RAISED: {type(e).__name__}: {e}")
        raise


def _wrapped_create(self, mesh_dim_name, backend_override=(None, None)):
    _call_count["create"] += 1
    n = _call_count["create"]
    root = self._get_root_mesh()
    cache_hit = mesh_dim_name in root._flatten_mapping
    if not cache_hit:
        _call_count["create_miss"] += 1
        _log(
            f"  [_create_flatten_mesh #{n}] *** CACHE MISS *** "
            f"requested {mesh_dim_name!r} on root {_summarize(root)}"
        )
    else:
        _log(
            f"  [_create_flatten_mesh #{n}] cache hit "
            f"{mesh_dim_name!r} on root id={id(root):#x}"
        )
    return _orig_create(self, mesh_dim_name, backend_override)


DeviceMesh._flatten = _wrapped_flatten  # type: ignore[method-assign]
DeviceMesh._create_flatten_mesh = _wrapped_create  # type: ignore[method-assign]


# --- Reproduction of the failing test path ---


def main() -> None:
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "fake", store=FakeStore(), rank=0, world_size=256
        )

    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", (32, 8), mesh_dim_names=("dp", "tp")
    )
    _log(f"\n=== USER MESH: {_summarize(mesh)} ===\n")

    from torch.distributed.fsdp import MixedPrecisionPolicy
    from torch.distributed.tensor.placement_types import Replicate, Shard

    from autoparallel._testing.models.llama3 import Transformer, TransformerModelArgs
    from autoparallel.api import AutoParallel
    from autoparallel.compile import _make_ac_joint_pass

    with torch.device("meta"):
        model = Transformer(
            TransformerModelArgs(
                dim=256,
                n_layers=2,
                n_heads=8,
                n_kv_heads=2,
                ffn_dim_multiplier=1.3,
                multiple_of=64,
                rope_theta=500000,
                vocab_size=1024,
                max_seq_len=512,
            )
        )

    vocab_size = 1024
    seqlen = 128
    batch_size = 2 * mesh.shape[0]

    with AutoParallel(
        model,
        lambda: torch.randint(0, vocab_size, (batch_size, seqlen), device="cuda"),
        mesh,
        MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
        repeated_subgraphs=True,
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_input_constraints([(Shard(0), Replicate())])
        autop.add_output_constraints([(Shard(0), Shard(2))])
        sharding_placement = autop.optimize_placement(verbose=False)

        # Log identity of every mesh referenced by the sharding solution
        seen: dict[int, Any] = {}
        _log("\n=== MESHES REFERENCED BY SHARDING SOLUTION ===")
        for node, strategy in sharding_placement.items():
            specs: list[Any] = []
            if hasattr(strategy, "output_specs"):
                output_specs = strategy.output_specs
                if isinstance(output_specs, (list, tuple)):
                    specs.extend(output_specs)
                else:
                    specs.append(output_specs)
            if hasattr(strategy, "input_specs"):
                input_specs = strategy.input_specs or []
                specs.extend(input_specs)
            for s in specs:
                if s is None:
                    continue
                m = getattr(s, "mesh", None)
                if m is None:
                    continue
                key = id(m)
                if key not in seen:
                    seen[key] = m
                    _log(f"  spec mesh: {_summarize(m)}")
        _log(f"=== TOTAL UNIQUE SPEC MESHES: {len(seen)} ===\n")

        ac_pass = _make_ac_joint_pass()
        with torch._functorch.config.patch({"joint_custom_pass": ac_pass}):
            _log("\n=== ENTERING apply_placement ===\n")
            try:
                autop.apply_placement(sharding_placement)
                _log("\n=== apply_placement SUCCEEDED ===\n")
            except Exception as e:
                _log(f"\n=== apply_placement FAILED: {type(e).__name__}: {e} ===\n")
                raise


if __name__ == "__main__":
    try:
        main()
    finally:
        _log(f"\n=== TOTAL _flatten calls: {_call_count['flatten']} ===")
        _log(f"=== TOTAL _create_flatten_mesh calls: {_call_count['create']} ===")
        _log(
            f"=== TOTAL _create_flatten_mesh CACHE MISSES: "
            f"{_call_count['create_miss']} ==="
        )
        if _call_count["create_miss"] > 1:
            _log(
                "\n*** SMOKING GUN: cache missed more than once. The first miss "
                "is the pre-warming. Any subsequent miss is a real mesh that "
                "wasn't covered by the pre-warming and will trigger as_strided "
                "dispatch inside make_fx. Search the log above for "
                "'*** CACHE MISS ***' to see which mesh."
            )
        # Also write to a file for CI artifact upload
        log_path = os.environ.get("FLATTEN_DIAG_LOG", "flatten_diagnosis.log")
        with open(log_path, "w") as f:
            f.write("\n".join(_log_lines))
        print(f"\nFull diagnostic log written to {log_path}")
