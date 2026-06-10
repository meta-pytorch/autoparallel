# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Regression test: DeviceMesh duplicates introduced by
``copy.deepcopy(op_schema_)`` in propagation rules used to trigger an
``as_strided``-inside-FakeTensorMode failure during ``apply_placement``.

Background: ``copy.deepcopy(op_schema)`` inside ``expand_rule`` produces a
fresh DeviceMesh object with an empty ``_flatten_mapping``. When the solver
picks a redistribution that calls ``mesh._flatten()`` on the duplicate,
``_create_flatten_mesh`` runs uncached, dispatching ``as_strided`` on the
rank_map — and FakeTensorMode rejects the non-fake tensor input.

Fix lives in ``autoparallel/shardings/propagation_rules.py`` as
``_deepcopy_preserving_mesh``: pre-seeds copy.deepcopy's memo with
DeviceMesh identity mappings so the deepcopy reuses the original meshes.

This test asserts the property we actually care about: every DeviceMesh
referenced by the sharding solution has a populated ``_flatten_mapping``
on its root, so a subsequent ``_flatten()`` call inside ``make_fx`` hits
the cache instead of dispatching.

We use the Transformer model because it triggers ``expand_rule`` (a
simpler model wouldn't exercise that propagation rule).
"""

import traceback

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
        if "device_mesh" not in f.filename and "test_mesh_identity" not in f.filename
    ]
    return "\n".join(interesting[-10:])


_diag = {"flatten": 0, "create": 0, "miss": 0, "installed": False}


def _install_diag_hooks():
    if _diag["installed"]:
        return
    _diag["installed"] = True

    _orig_flatten = DeviceMesh._flatten
    _orig_create = DeviceMesh._create_flatten_mesh

    def _wrapped_flatten(self, mesh_dim_name=None, backend_override=None):
        _diag["flatten"] += 1
        n = _diag["flatten"]
        requested = mesh_dim_name or (
            "_".join(self._mesh_dim_names) if self._mesh_dim_names else "<unnamed>"
        )
        print(
            f"\n[_flatten #{n}] CALL on {_summarize(self)} requested={requested!r}",
            flush=True,
        )
        print(f"[_flatten #{n}] call site:\n{_short_traceback()}", flush=True)
        try:
            result = _orig_flatten(self, mesh_dim_name, backend_override)
            print(f"[_flatten #{n}] OK → {_summarize(result)}", flush=True)
            return result
        except Exception as e:
            print(f"[_flatten #{n}] RAISED: {type(e).__name__}: {e}", flush=True)
            raise

    def _wrapped_create(self, mesh_dim_name, backend_override=(None, None)):
        _diag["create"] += 1
        n = _diag["create"]
        root = self._get_root_mesh()
        cache_hit = mesh_dim_name in root._flatten_mapping
        if not cache_hit:
            _diag["miss"] += 1
            print(
                f"  [_create_flatten_mesh #{n}] *** CACHE MISS *** "
                f"name={mesh_dim_name!r} root_id={id(root):#x} "
                f"root_cache={list(root._flatten_mapping)}",
                flush=True,
            )
        else:
            print(
                f"  [_create_flatten_mesh #{n}] cache hit name={mesh_dim_name!r} "
                f"root_id={id(root):#x}",
                flush=True,
            )
        return _orig_create(self, mesh_dim_name, backend_override)

    DeviceMesh._flatten = _wrapped_flatten  # type: ignore[method-assign]
    DeviceMesh._create_flatten_mesh = _wrapped_create  # type: ignore[method-assign]


@apply_cuda_patches
def test_sharding_solution_meshes_have_warm_flatten_cache(device_mesh_2d):
    """After ``apply_placement``'s pre-warming, every spec mesh's root
    must have the default flattened mesh cached. Otherwise a subsequent
    ``_flatten()`` call inside ``make_fx`` triggers ``as_strided`` on the
    rank_map and FakeTensorMode rejects it (the original CI failure).
    """
    _install_diag_hooks()
    print(
        f"\n=== USER MESH (from fixture): {_summarize(device_mesh_2d)} ===\n",
        flush=True,
    )

    # Clear caches that might carry duplicated meshes from prior tests
    # (Dynamo guard cache, DTensor propagation lru_cache, etc.).
    torch._dynamo.reset()
    try:
        from torch.distributed.tensor._api import DTensor

        if hasattr(DTensor, "_op_dispatcher"):
            sp = DTensor._op_dispatcher.sharding_propagator
            sp._propagate_tensor_meta_cached.cache_clear()
            sp.op_strategy_funcs  # ensure attr exists
            if hasattr(sp, "op_to_rules_lru"):
                sp.op_to_rules_lru.cache_clear()
    except Exception as e:
        print(f"cache clear note: {e}", flush=True)

    vocab_size = 1024
    seqlen = 128
    batch_size = 2 * device_mesh_2d.shape[0]

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
            device_mesh_2d,
            MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32
            ),
            repeated_subgraphs=True,
        ) as autop:
            autop.add_parameter_memory_constraint(low=None, high=None)
            autop.add_input_constraints([(Shard(0), Replicate())])
            autop.add_output_constraints([(Shard(0), Shard(2))])
            sharding_placement = autop.optimize_placement(verbose=False)

            seen: dict[int, object] = {}
            print("\n=== MESHES IN SHARDING SOLUTION ===", flush=True)
            for strategy in sharding_placement.values():
                specs = []
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
                        print(f"  spec mesh: {_summarize(m)}", flush=True)
            print(f"=== TOTAL UNIQUE SPEC MESHES: {len(seen)} ===\n", flush=True)

            # apply_placement pre-warms the user mesh's _flatten cache so
            # subsequent _flatten() calls inside make_fx hit the cache.
            autop.apply_placement(sharding_placement)
    finally:
        print(
            f"\n=== TOTAL _flatten={_diag['flatten']} "
            f"create={_diag['create']} MISSES={_diag['miss']} ===",
            flush=True,
        )

    # Collect every distinct spec mesh from the solution
    spec_meshes: dict[int, object] = {}
    for strategy in sharding_placement.values():
        specs = []
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
            spec_meshes[id(m)] = m

    cold = []
    for mid, m in spec_meshes.items():
        if m.ndim == 1:
            # 1D meshes: _flatten() short-circuits to self without dispatch
            continue
        root = m._get_root_mesh()
        default_name = "_".join(m._mesh_dim_names)
        if default_name not in root._flatten_mapping:
            cold.append((mid, m._mesh_dim_names, list(root._flatten_mapping)))

    assert not cold, (
        f"After apply_placement, {len(cold)} spec mesh(es) still have a "
        f"cold _flatten_mapping for their default name. A subsequent "
        f"_flatten() call inside make_fx will dispatch as_strided and "
        f"fail FakeTensorMode's non-fake-input check. Details "
        f"(id, dim_names, root cache keys): {cold}. See "
        f"_deepcopy_preserving_mesh in propagation_rules.py."
    )
