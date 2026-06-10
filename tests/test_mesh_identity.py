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

import torch
from conftest import apply_cuda_patches
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel._testing.models.llama3 import Transformer, TransformerModelArgs
from autoparallel.api import AutoParallel


@apply_cuda_patches
def test_sharding_solution_meshes_have_warm_flatten_cache(device_mesh_2d):
    """After ``apply_placement``'s pre-warming, every spec mesh's root
    must have the default flattened mesh cached. Otherwise a subsequent
    ``_flatten()`` call inside ``make_fx`` triggers ``as_strided`` on the
    rank_map and FakeTensorMode rejects it (the original CI failure).
    """
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

    with AutoParallel(
        model,
        lambda: torch.randint(0, vocab_size, (batch_size, seqlen), device="cuda"),
        device_mesh_2d,
        MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
        repeated_subgraphs=True,
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_input_constraints([(Shard(0), Replicate())])
        autop.add_output_constraints([(Shard(0), Shard(2))])
        sharding_placement = autop.optimize_placement(verbose=False)
        # apply_placement pre-warms the user mesh's _flatten cache so
        # subsequent _flatten() calls inside make_fx hit the cache.
        autop.apply_placement(sharding_placement)

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
