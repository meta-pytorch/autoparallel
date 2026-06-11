"""Diagnose CI failure for test_mesh_identity.py.

Wraps DeviceMesh._flatten and DeviceMesh._create_flatten_mesh to log
every call. Mirrors the exact path the failing test takes — no extra
config patches, no joint_custom_pass.

Run on CI immediately after the failure to see which mesh triggered
the as_strided dispatch and why the cache lookup missed.
"""

import os
import sys
import traceback
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

# Mirror the apply_cuda_patches in conftest (H100, capability 9.0)
from unittest.mock import patch

_PATCHES: list[Any] = [
    patch("torch.cuda.device_count", lambda: 8),
    patch("torch.cuda.get_device_name", lambda *a, **k: "H100"),
    patch("torch.cuda.get_device_capability", lambda *a, **k: (9, 0)),
    patch(
        "torch.cuda.get_device_properties",
        lambda *a, **k: type(
            "Props",
            (),
            {
                "major": 9,
                "minor": 0,
                "name": "H100",
                "total_memory": 80 * 1024**3,
                "multi_processor_count": 132,
            },
        )(),
    ),
]
for p in _PATCHES:
    p.start()


from autoparallel._testing.models.llama3 import Transformer, TransformerModelArgs
from autoparallel.api import AutoParallel

_log_lines: list[str] = []


def _log(msg: str) -> None:
    _log_lines.append(msg)
    print(msg, flush=True)


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
        else "<none>"
    )
    return (
        f"mesh_id={id(m):#x} ndim={ndim} dim_names={dim_names} "
        f"is_root={is_root} root_id={id(root):#x} root_cache={cache_keys}"
    )


def _short_traceback() -> str:
    stack = traceback.extract_stack()
    interesting = [
        f"  {f.filename}:{f.lineno} in {f.name}"
        for f in stack
        if "device_mesh" not in f.filename
        and os.path.basename(__file__) not in f.filename
    ]
    return "\n".join(interesting[-8:])


_orig_flatten = DeviceMesh._flatten
_orig_create = DeviceMesh._create_flatten_mesh

_count = {"flatten": 0, "create": 0, "miss": 0}


def _wrapped_flatten(self, mesh_dim_name=None, backend_override=None):
    _count["flatten"] += 1
    n = _count["flatten"]
    requested = mesh_dim_name or (
        "_".join(self._mesh_dim_names) if self._mesh_dim_names else "<unnamed>"
    )
    _log(f"\n[_flatten #{n}] CALL on {_summarize(self)} requested={requested!r}")
    _log(f"[_flatten #{n}] call site:\n{_short_traceback()}")
    try:
        result = _orig_flatten(self, mesh_dim_name, backend_override)
        _log(f"[_flatten #{n}] OK → {_summarize(result)}")
        return result
    except Exception as e:
        _log(f"[_flatten #{n}] RAISED: {type(e).__name__}: {e}")
        raise


def _wrapped_create(self, mesh_dim_name, backend_override=(None, None)):
    _count["create"] += 1
    n = _count["create"]
    root = self._get_root_mesh()
    cache_hit = mesh_dim_name in root._flatten_mapping
    if not cache_hit:
        _count["miss"] += 1
        _log(
            f"  [_create_flatten_mesh #{n}] *** CACHE MISS *** "
            f"name={mesh_dim_name!r} root_id={id(root):#x} "
            f"root_cache={list(root._flatten_mapping)}"
        )
    else:
        _log(
            f"  [_create_flatten_mesh #{n}] cache hit name={mesh_dim_name!r} "
            f"root_id={id(root):#x}"
        )
    return _orig_create(self, mesh_dim_name, backend_override)


DeviceMesh._flatten = _wrapped_flatten  # type: ignore[method-assign]
DeviceMesh._create_flatten_mesh = _wrapped_create  # type: ignore[method-assign]


def main() -> None:
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "fake", store=FakeStore(), rank=0, world_size=256
        )

    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", (32, 8), mesh_dim_names=("dp", "tp")
    )
    _log(f"\n=== USER MESH: {_summarize(mesh)} ===\n")

    vocab_size = 1024
    seqlen = 128
    batch_size = 2 * mesh.shape[0]

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
                vocab_size=vocab_size,
                max_seq_len=seqlen,
            )
        )

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

        seen: dict[int, Any] = {}
        _log("\n=== MESHES IN SHARDING SOLUTION ===")
        for strategy in sharding_placement.values():
            specs: list[Any] = []
            if hasattr(strategy, "output_specs"):
                o = strategy.output_specs
                specs.extend(o if isinstance(o, (list, tuple)) else [o])
            if hasattr(strategy, "input_specs"):
                specs.extend(strategy.input_specs or [])
            for s in specs:
                if s is None:
                    continue
                m = getattr(s, "mesh", None)
                if m is None:
                    continue
                if id(m) not in seen:
                    seen[id(m)] = m
                    _log(f"  spec mesh: {_summarize(m)}")
        _log(f"=== TOTAL UNIQUE SPEC MESHES: {len(seen)} ===\n")

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
        _log(f"\n=== TOTAL _flatten calls: {_count['flatten']} ===")
        _log(f"=== TOTAL _create_flatten_mesh calls: {_count['create']} ===")
        _log(f"=== TOTAL _create_flatten_mesh CACHE MISSES: {_count['miss']} ===")
        if _count["miss"] > 1:
            _log(
                "\n*** SMOKING GUN: cache missed more than once. "
                "First miss is the pre-warming. Any further miss is the "
                "duplicate mesh triggering the as_strided dispatch. "
                "Search above for '*** CACHE MISS ***'."
            )
        log_path = os.environ.get(
            "FLATTEN_DIAG_LOG", "mesh_identity_diagnosis.log"
        )
        with open(log_path, "w") as f:
            f.write("\n".join(_log_lines))
        print(f"\nFull diagnostic log written to {log_path}")
