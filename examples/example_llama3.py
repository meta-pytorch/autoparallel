# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel._testing.models.llama3 import Transformer, TransformerModelArgs
from autoparallel.api import AutoParallel
from autoparallel.auto_bucketing import configure_inductor_for_autobucketing

world_size = 64

fake_store = FakeStore()
torch.distributed.init_process_group(
    "fake", store=fake_store, rank=0, world_size=world_size
)

use_1d_mesh = False

if use_1d_mesh:
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("dp",)
    )
else:
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda",
        (world_size // 8, 8),
        mesh_dim_names=(
            "dp",
            "tp",
        ),
    )

batch_size = 2 * mesh.shape[0]
seqlen = 2048 * 4
vocab_size = 128256
use_vocab_parallel = not use_1d_mesh
device = torch.device("cuda")

model_type = "8b"
enable_asynctp = False


def model_fn():
    if model_type == "8b":
        model_args = TransformerModelArgs(
            dim=4096,
            n_layers=32,
            n_heads=32,
            n_kv_heads=8,
            ffn_dim_multiplier=1.3,
            multiple_of=1024,
            rope_theta=500000,
            vocab_size=vocab_size,
            max_seq_len=seqlen,
        )
    elif model_type == "70b":
        model_args = TransformerModelArgs(
            dim=8192,
            n_layers=80,
            n_heads=64,
            n_kv_heads=8,
            ffn_dim_multiplier=1.3,
            multiple_of=4096,
            rope_theta=500000,
            vocab_size=vocab_size,
            max_seq_len=seqlen,
        )
    else:
        raise ValueError(f"{model_type} not available")
    m = Transformer(model_args)
    return m


def input_fn():
    x = torch.randint(0, vocab_size, (batch_size, seqlen), device=device)
    return x


configure_inductor_for_autobucketing("aten")

# parallelize the model
with torch.device("meta"):
    model = model_fn()

mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)


def group_mm_nodes_with_its_gradients(nodes):
    fwd_nodes = [n for n in nodes if "nn_module_stack" in n.meta]
    bwd_nodes = [n for n in nodes if "fwd_nn_module_stack" in n.meta]
    assert len(fwd_nodes) * 2 == len(bwd_nodes)
    res = {}
    for fwd_node in fwd_nodes:
        o = []
        for bwd_node in bwd_nodes:
            if fwd_node.meta["nn_module_stack"] == bwd_node.meta["fwd_nn_module_stack"]:
                o.append(bwd_node)
        assert len(o) == 2
        res[fwd_node] = o
    return res


def force_tp_constraints(autop, mm_nodes, feat_dim=1, bwd_constraint=False):
    # out = x @ w   - S(0)R, RS(1) -> S(0)S(1)
    # g_w = g.T @ x - S(1)S(0), S(0)R -> PS(0)
    # g_x = g @ w.T - S(0)S(1), RS(0) -> S(0)P

    add_node_constraint = autop.sharding_optimizer.add_node_constraint
    fwd_bwd_groups = group_mm_nodes_with_its_gradients(mm_nodes)
    fwd_nodes = list(fwd_bwd_groups.keys())
    dim1 = 0 if feat_dim == 1 else 1
    dim2 = 1 if feat_dim == 1 else 0
    # assume there are 7 mm nodes per transformer block
    # skip last mm as it's the final projection layer
    assert (
        len(fwd_nodes) - 1
    ) % 7 == 0, f"expected 7 mm nodes per transformer block, {len(fwd_nodes) - 1}"
    for block in range(0, len(fwd_nodes) - 1, 7):
        fwd_nodes_block = fwd_nodes[block : block + 7]
        # force the first 3 mm nodes to be S(0)S(1)
        the_nodes = fwd_nodes_block[:3] + fwd_nodes_block[4:6]
        for n in the_nodes:
            add_node_constraint(n, (Shard(0), Shard(feat_dim)))
            add_node_constraint(n.all_input_nodes[0], (Shard(0), Replicate()))
            add_node_constraint(n.all_input_nodes[1], (Replicate(), Shard(1)))

            if bwd_constraint:
                bwd_nodes = fwd_bwd_groups[n]
                # first is g_w, second is g_x
                add_node_constraint(bwd_nodes[0], (Partial(), Shard(dim1)))
                add_node_constraint(bwd_nodes[1], (Shard(0), Partial()))

        # add reduction to finish TP, yielding S(0)P
        the_nodes = fwd_nodes_block[3:4] + fwd_nodes_block[6:7]
        for n in the_nodes:
            add_node_constraint(n, (Shard(0), Partial()))
            add_node_constraint(n.all_input_nodes[0], (Shard(0), Shard(feat_dim)))
            add_node_constraint(n.all_input_nodes[1], (Replicate(), Shard(0)))

            if bwd_constraint:
                bwd_nodes = fwd_bwd_groups[n]
                # first is g_w, second is g_x
                add_node_constraint(bwd_nodes[0], (Partial(), Shard(dim2)))
                add_node_constraint(bwd_nodes[1], (Shard(0), Shard(feat_dim)))


def add_tp_constraints(autop):
    mm_nodes = autop.gm.graph.find_nodes(
        op="call_function", target=torch.ops.aten.mm.default
    )
    einsum_nodes = autop.gm.graph.find_nodes(
        op="call_function", target=torch.ops.aten.einsum.default
    )
    assert (len(mm_nodes) > 0) ^ (
        len(einsum_nodes) > 0
    ), f"only one should be non-empty, got {len(mm_nodes)} and {len(einsum_nodes)}"
    feat_dim = 1 if len(mm_nodes) > 0 else 2
    tgt_nodes = mm_nodes + einsum_nodes
    force_tp_constraints(autop, tgt_nodes, feat_dim=feat_dim, bwd_constraint=True)

    if einsum_nodes:
        # add sequence parallelism if we have einsum nodes
        autop.sharding_optimizer.add_node_constraint(
            list(tgt_nodes[3].users)[0], (Shard(0), Shard(1))
        )
        autop.sharding_optimizer.add_node_constraint(
            list(list(tgt_nodes[3].users)[0].users)[0], (Shard(0), Shard(1))
        )


# parallelize the model
with AutoParallel(
    model, input_fn, mesh, mp_policy, compile=True, repeated_subgraphs=True
) as autop:
    autop.add_parameter_memory_constraint(low=None, high=None)

    x_sharding = (Shard(0),) + (Replicate(),) * (mesh.ndim - 1)
    out_sharding = x_sharding
    if use_vocab_parallel:
        # add vocab parallel constraint
        assert mesh.ndim == 2, "Only 2d mesh supported here"
        out_sharding = (Shard(0), Shard(2))

    autop.add_input_constraints([x_sharding])
    autop.add_output_constraints([out_sharding])

    enable_manual_constraint = False
    if enable_manual_constraint and not use_1d_mesh:
        add_tp_constraints(autop)

    if enable_asynctp:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        enable_symm_mem_for_group(mesh["dp"].get_group().group_name)
        enable_symm_mem_for_group(mesh["tp"].get_group().group_name)
        torch._inductor.config._micro_pipeline_tp = False
        from autoparallel.asynctp import micro_pipeline_tp_pass

        existing_post_grad_custom_post_pass = (
            torch._inductor.config.post_grad_custom_post_pass
        )

        def _pass(graph):
            if existing_post_grad_custom_post_pass is not None:
                existing_post_grad_custom_post_pass(graph)
            micro_pipeline_tp_pass(graph)

        torch._inductor.config.post_grad_custom_post_pass = _pass

    t = time.time()
    sharding_placement = autop.optimize_placement(verbose=True)
    print(f"Took {time.time() - t:.2f} s")
    parallel_mod = autop.apply_placement(sharding_placement)

# run weight init on our sharded DTensor params
parallel_mod.to_empty(device="cuda")
parallel_mod.init_weights()

# now let's run it
x = (
    torch.randint(
        0,
        vocab_size,
        (batch_size // mesh.shape[0], seqlen),
        device=torch.device("cuda"),
    ),
)
out = parallel_mod(*x)
out.backward(torch.randn_like(out))
print("All good!")
