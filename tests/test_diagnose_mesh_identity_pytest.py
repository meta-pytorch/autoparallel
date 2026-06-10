"""Diagnose CI mesh-identity failure FROM WITHIN PYTEST.

Adds extensive instrumentation around the actual failing test to expose
state at every flatten call. Crucial difference from
tests/diagnose_mesh_identity_ci.py: this runs as a pytest test, so
fixtures and prior-test pollution are in effect — matching the actual
CI failure conditions.
"""

import os
import traceback
from typing import Any

import torch
from conftest import apply_cuda_patches
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel._testing.models.llama3 import Transformer, TransformerModelArgs
from autoparallel.api import AutoParallel


def _summarize(m) -> str:
    if m is None:
        return "None"
    dim_names = getattr(m, "_mesh_dim_names", None)
    ndim = getattr(m, "ndim", "?")
    root = m._get_root_mesh() if hasattr(m, "_get_root_mesh") else None
    cache_keys = (
        list(root._flatten_mapping.keys())
        if root is not None and hasattr(root, "_flatten_mapping")
        else "<none>"
    )
    return (
        f"mesh_id={id(m):#x} ndim={ndim} dim_names={dim_names} "
        f"is_root={root is m} root_id={id(root):#x} root_cache={cache_keys}"
    )


def _short_traceback() -> str:
    stack = traceback.extract_stack()
    interesting = [
        f"  {f.filename}:{f.lineno} in {f.name}"
        for f in stack
        if "device_mesh" not in f.filename
        and "diagnose_mesh_identity_pytest" not in f.filename
    ]
    return "\n".join(interesting[-10:])


_state: dict[str, Any] = {
    "log": [],
    "count_flatten": 0,
    "count_create": 0,
    "count_miss": 0,
    "installed": False,
}


def _log(msg: str) -> None:
    _state["log"].append(msg)
    print(msg, flush=True)


def _install_hooks():
    if _state["installed"]:
        return
    _state["installed"] = True

    _orig_flatten = DeviceMesh._flatten
    _orig_create = DeviceMesh._create_flatten_mesh

    def _wrapped_flatten(self, mesh_dim_name=None, backend_override=None):
        _state["count_flatten"] += 1
        n = _state["count_flatten"]
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
        _state["count_create"] += 1
        n = _state["count_create"]
        root = self._get_root_mesh()
        cache_hit = mesh_dim_name in root._flatten_mapping
        if not cache_hit:
            _state["count_miss"] += 1
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


@apply_cuda_patches
def test_diagnose_mesh_identity_pytest(device_mesh_2d):
    """Run the same scenario as test_mesh_identity.py but with instrumentation."""
    _install_hooks()
    mesh = device_mesh_2d
    _log(f"\n=== USER MESH (from fixture): {_summarize(mesh)} ===\n")

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

    try:
        with AutoParallel(
            model,
            lambda: torch.randint(0, vocab_size, (batch_size, seqlen), device="cuda"),
            mesh,
            MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32
            ),
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
            autop.apply_placement(sharding_placement)
            _log("\n=== apply_placement SUCCEEDED ===\n")
    finally:
        _log(f"\n=== TOTAL _flatten calls: {_state['count_flatten']} ===")
        _log(f"=== TOTAL _create_flatten_mesh calls: {_state['count_create']} ===")
        _log(f"=== TOTAL _create_flatten_mesh CACHE MISSES: {_state['count_miss']} ===")
        log_path = os.environ.get(
            "MESH_IDENTITY_PYTEST_LOG", "mesh_identity_pytest.log"
        )
        with open(log_path, "w") as f:
            f.write("\n".join(_state["log"]))
        print(f"\nFull diagnostic log written to {log_path}", flush=True)
