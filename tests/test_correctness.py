# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Numerical correctness tests for AutoParallel.

Compares the output of the parallelized model against the single-GPU
unsharded reference. Uses LocalTensorMode to simulate multi-rank execution
on a single GPU, following the pattern from test_redistribute_tensor.py.

We run the joint forward+backward FX graph (parallel_gm) directly under
LocalTensorMode rather than using apply_placement(), which would give a
cleaner nn.Module with autograd-based backward. Two upstream issues prevent
the apply_placement() path today:
  1. ProcessGroup objects (from compile_on_one_rank) are not deepcopy-safe,
     breaking extract_forward_graph's deepcopy of the joint graph.
  2. AOT autograd's compiled backward rejects LocalTensor tangents because
     LocalTensor doesn't implement __coerce_same_metadata_as_tangent__.

The joint graph's placeholders are [primals..., tangents...] and its outputs
are ((fwd_outs...), (bwd_outs...)), so we construct both primals and tangents
as LocalTensors and split the output to compare forward results and gradients.

All tests use compile_on_one_rank + dynamic=True so that rank coordinates
are symbolic in the graph (not baked in for rank 0). This is necessary for
any graph that contains all-gather → compute → local-split patterns.
"""

import contextlib

import torch
import torch.utils._pytree as pytree
from torch import nn
from torch.distributed._local_tensor import (
    LocalIntNode,
    LocalTensor,
    LocalTensorMode,
    enabled_local_tensor_mode,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.distributed_c10d import ProcessGroup
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel.api import AutoParallel
from tests.conftest import apply_cuda_patches

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rank_coord(rank, mesh_shape, mesh_dim):
    """Compute a rank's coordinate on a given mesh dimension."""
    # Mesh dimensions are ordered outermost-first.  The stride of mesh_dim
    # equals the product of all dimensions after it.
    stride = 1
    for d in range(len(mesh_shape) - 1, mesh_dim, -1):
        stride *= mesh_shape[d]
    return (rank // stride) % mesh_shape[mesh_dim]


def _make_shards(global_tensor, mesh_shape, placements):
    """Partition a global tensor into per-rank shards.

    Returns dict[rank -> Tensor], ready for LocalTensor(...).
    """
    world_size = 1
    for s in mesh_shape:
        world_size *= s

    shards = {}
    for rank in range(world_size):
        shard = global_tensor
        for mesh_dim, placement in enumerate(placements):
            if isinstance(placement, Shard):
                n_chunks = mesh_shape[mesh_dim]
                coord = _rank_coord(rank, mesh_shape, mesh_dim)
                chunk_size = shard.size(placement.dim) // n_chunks
                shard = shard.narrow(placement.dim, coord * chunk_size, chunk_size)
        shards[rank] = shard.contiguous().clone()
    return shards


def _gather_shards(local_tensor, mesh_shape, placements):
    """Reassemble a full tensor from per-rank LocalTensor shards.

    For Shard(dim): concatenate along dim (using rank ordering on that mesh dim).
    For Replicate: take rank 0's copy.
    """
    if not isinstance(local_tensor, LocalTensor):
        return local_tensor

    world_size = 1
    for s in mesh_shape:
        world_size *= s

    result = local_tensor._local_tensors[0]

    # Process mesh dimensions from last to first (innermost first).
    for mesh_dim in reversed(range(len(mesh_shape))):
        placement = placements[mesh_dim]
        if isinstance(placement, Shard):
            # Group ranks by their coordinates on all OTHER mesh dims,
            # then concatenate along this mesh dim's shard axis.
            # For simplicity, if this is the only shard dim, just concat rank 0..N-1.
            # For multi-dim meshes, we need to be more careful.
            stride = 1
            for d in range(len(mesh_shape) - 1, mesh_dim, -1):
                stride *= mesh_shape[d]

            # Collect all unique "fiber" groups along this mesh dim.
            # Each fiber is a set of ranks that differ only in mesh_dim coordinate.
            fibers = {}
            for rank in range(world_size):
                key = tuple(
                    _rank_coord(rank, mesh_shape, d)
                    for d in range(len(mesh_shape))
                    if d != mesh_dim
                )
                if key not in fibers:
                    fibers[key] = []
                fibers[key].append(rank)

            # For each fiber, concatenate along the shard dim.
            new_shards = {}
            for key, ranks in fibers.items():
                # Sort by coordinate on mesh_dim.
                ranks_sorted = sorted(
                    ranks, key=lambda r: _rank_coord(r, mesh_shape, mesh_dim)
                )
                chunks = [local_tensor._local_tensors[r] for r in ranks_sorted]
                gathered = torch.cat(chunks, dim=placement.dim)
                # Store result under the first rank as representative.
                new_shards[ranks_sorted[0]] = gathered

            if len(new_shards) == 1:
                result = next(iter(new_shards.values()))
            else:
                # Rebuild LocalTensor from the gathered fibers, mapping each
                # original rank to its fiber's gathered result.
                rebuilt = {}
                for key, ranks in fibers.items():
                    ranks_sorted = sorted(
                        ranks, key=lambda r: _rank_coord(r, mesh_shape, mesh_dim)
                    )
                    val = new_shards[ranks_sorted[0]]
                    for r in ranks:
                        rebuilt[r] = val
                local_tensor = LocalTensor(rebuilt)
                result = local_tensor._local_tensors[0]

    return result


@contextlib.contextmanager
def _patch_local_tensor_for_compile_on_one_rank():
    """Monkey-patch LocalTensorMode for compile_on_one_rank graph compatibility.

    compile_on_one_rank changes how the parallel graph references process groups
    and computes rank coordinates. LocalTensorMode doesn't handle these yet
    upstream, so we patch two things:

    1. Functional collectives receive ProcessGroup objects instead of string
       group names. We make _resolve_process_group accept both.
    2. _runtime_compute_coordinate_on_dim calls dist.get_rank() which is
       always 0 under the fake PG. We override it to produce per-rank
       LocalIntNode values when under LocalTensorMode.
    """
    import torch.distributed._local_tensor._c10d as _lt_c10d

    # Patch 1: ProcessGroup passthrough in functional collectives.
    _orig_resolve = _lt_c10d._resolve_process_group

    def _resolve_pg_or_passthrough(group_name):
        if isinstance(group_name, ProcessGroup):
            return group_name
        return _orig_resolve(group_name)

    _lt_c10d._resolve_process_group = _resolve_pg_or_passthrough

    # Patch 2: per-rank coordinate computation.
    lib = torch.library.Library("device_mesh", "IMPL")

    def _coord_with_local_tensor(full_mesh, index):
        lm = enabled_local_tensor_mode()
        if lm is None:
            rank = torch.distributed.get_rank()
            mesh_t = DeviceMesh._get_mesh_tensor_from_full_mesh(full_mesh)
            mesh_coords = DeviceMesh._compute_coordinates_from_mesh(mesh_t, rank)
            if mesh_coords is None:
                raise AssertionError
            return mesh_coords[index]
        mesh_t = DeviceMesh._get_mesh_tensor_from_full_mesh(full_mesh)
        coords = {}
        for rank in lm.ranks:
            rank_coords = DeviceMesh._compute_coordinates_from_mesh(mesh_t, rank)
            if rank_coords is None:
                raise AssertionError(f"rank {rank} not in mesh")
            coords[rank] = rank_coords[index]
        return torch.SymInt(LocalIntNode(coords))

    lib.impl(
        "_runtime_compute_coordinate_on_dim",
        _coord_with_local_tensor,
        "CompositeExplicitAutograd",
    )

    try:
        yield
    finally:
        _lt_c10d._resolve_process_group = _orig_resolve


@apply_cuda_patches
def _get_parallel_graph_and_placements(model_fn, input_fn, mesh):
    """Run AutoParallel and extract the parallel graph + placement info.

    Always uses dynamic=True (required for compile_on_one_rank) and adds
    a parameter memory constraint to force parameter sharding.
    """
    with torch.device("meta"):
        meta_model = model_fn()

    x_sharding = (Shard(0),) + (Replicate(),) * (mesh.ndim - 1)

    with AutoParallel(meta_model, input_fn, mesh, dynamic=True) as autop:
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])
        autop.add_parameter_memory_constraint()
        sharding_placement = autop.optimize_placement(verbose=False)
        autop._apply_placement_common(sharding_placement)

        parallel_gm = autop.parallel_gm
        gm = autop.gm  # Original graph (keys for sharding_placement)

        param_fqns = list(autop.joint_with_descriptors.params_spec)
        buffer_fqns = list(autop.joint_with_descriptors.buffers_spec)
        fw_metadata = autop.joint_with_descriptors._aot_state.fw_metadata
        num_fwd_outputs = fw_metadata.num_forward_returns

        # Extract placements for each primal placeholder (same order as gm's
        # placeholders, which matches parallel_gm's primal placeholders).
        primal_placements = []
        num_primals = len(fw_metadata.input_info)
        orig_placeholders = [n for n in gm.graph.find_nodes(op="placeholder")][
            :num_primals
        ]
        for node in orig_placeholders:
            val = node.meta.get("val")
            if not isinstance(val, torch.Tensor):
                primal_placements.append(None)
                continue
            strategy = sharding_placement[node]
            primal_placements.append(tuple(strategy.output_specs.placements))

        num_user_inputs = num_primals - len(param_fqns) - len(buffer_fqns)

    return (
        parallel_gm,
        primal_placements,
        num_fwd_outputs,
        param_fqns,
        buffer_fqns,
        num_user_inputs,
    )


def _run_correctness_test(
    model_fn,
    input_fn,
    mesh_shape,
    atol=1e-5,
    rtol=1e-5,
):
    """Core correctness test routine.

    1. Runs the reference model forward+backward on CUDA.
    2. Runs AutoParallel tracing+optimization (with compile_on_one_rank +
       dynamic=True + parameter memory constraint) to get parallel_gm.
    3. Feeds LocalTensor primals+tangents into parallel_gm under LocalTensorMode.
    4. Compares forward outputs and gradients against reference.
    """
    world_size = 1
    for s in mesh_shape:
        world_size *= s

    # --- Setup fake distributed (for AutoParallel tracing) ---
    if not torch.distributed.is_initialized():
        fake_store = FakeStore()
        torch.distributed.init_process_group(
            "fake", store=fake_store, rank=0, world_size=256
        )

    if len(mesh_shape) == 1:
        mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda", mesh_shape, mesh_dim_names=("dp",)
        )
    else:
        mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda", mesh_shape, mesh_dim_names=("dp", "tp")
        )

    # --- Reference forward + backward on CUDA ---
    torch.manual_seed(42)
    model = model_fn().cuda()
    ref_inputs = input_fn()
    if not isinstance(ref_inputs, (tuple, list)):
        ref_inputs = (ref_inputs,)
    ref_inputs = tuple(t.detach().cuda().requires_grad_(True) for t in ref_inputs)
    ref_out = model(*ref_inputs)
    grad_output = torch.randn_like(ref_out)
    ref_out.backward(grad_output)
    ref_out_detached = ref_out.detach()
    ref_input_grads = [inp.grad.clone() for inp in ref_inputs]
    ref_param_grads = {name: p.grad.clone() for name, p in model.named_parameters()}

    # --- AutoParallel tracing ---
    import torch.distributed.config as dist_config

    def tracing_input_fn():
        return input_fn()

    with dist_config.patch(compile_on_one_rank=True):
        (
            parallel_gm,
            primal_placements,
            num_fwd_outputs,
            param_fqns,
            buffer_fqns,
            num_user_inputs,
        ) = _get_parallel_graph_and_placements(model_fn, tracing_input_fn, mesh)

    # --- Build LocalTensor primals and tangents ---
    num_params = len(param_fqns)
    num_buffers = len(buffer_fqns)
    param_dict = dict(model.named_parameters())

    local_primals = []
    # Parameters
    for i, fqn in enumerate(param_fqns):
        placements = primal_placements[i]
        full_param = param_dict[fqn].detach()
        shards = _make_shards(full_param, mesh_shape, placements)
        local_primals.append(LocalTensor(shards))

    # Buffers
    buffer_dict = dict(model.named_buffers())
    for i, fqn in enumerate(buffer_fqns):
        placements = primal_placements[num_params + i]
        full_buf = buffer_dict[fqn].detach()
        shards = _make_shards(full_buf, mesh_shape, placements)
        local_primals.append(LocalTensor(shards))

    # User inputs
    for i, inp in enumerate(ref_inputs):
        placements = primal_placements[num_params + num_buffers + i]
        shards = _make_shards(inp.detach(), mesh_shape, placements)
        local_primals.append(LocalTensor(shards))

    # Tangents: the grad_output sharded like the forward output.
    # The forward output placements match the output constraints (Shard(0) on dp, R on tp).
    output_placements = (Shard(0),) + (Replicate(),) * (len(mesh_shape) - 1)

    # Build tangent LocalTensors — one per forward output.
    # The joint graph may have multiple tangent placeholders; we look at parallel_gm.
    num_primals_total = num_params + num_buffers + num_user_inputs
    all_placeholders = [n for n in parallel_gm.graph.find_nodes(op="placeholder")]
    tangent_placeholders = all_placeholders[num_primals_total:]

    local_tangents = []
    for i, tnode in enumerate(tangent_placeholders):
        val = tnode.meta.get("val")
        if not isinstance(val, torch.Tensor):
            local_tangents.append(val)
            continue
        # Use grad_output for the first tangent (the main output gradient).
        # For additional tangents (e.g., None gradients for non-differentiable
        # outputs), use zeros with the correct local shape.
        if i == 0:
            shards = _make_shards(grad_output, mesh_shape, output_placements)
        else:
            # Tangent shape matches the local shape from meta.
            full_tangent = torch.zeros(val.shape)
            shards = {r: full_tangent.clone() for r in range(world_size)}
        local_tangents.append(LocalTensor(shards))

    # --- Run parallel_gm under LocalTensorMode ---
    with _patch_local_tensor_for_compile_on_one_rank():
        with LocalTensorMode(frozenset(range(world_size))):
            joint_result = parallel_gm(*local_primals, *local_tangents)

    # --- Parse outputs ---
    # The joint graph output is ((fwd_outs...), (bwd_outs...)).
    # Flatten to get individual tensors.
    flat_outputs = pytree.arg_tree_leaves(joint_result)
    fwd_results = flat_outputs[:num_fwd_outputs]
    bwd_results = flat_outputs[num_fwd_outputs:]

    # --- Compare forward output ---
    fwd_out = _gather_shards(fwd_results[0], mesh_shape, output_placements)
    torch.testing.assert_close(fwd_out, ref_out_detached, atol=atol, rtol=rtol)

    # --- Compare backward gradients ---
    # The backward outputs correspond to gradients wrt each primal, in order:
    # [grad_param_0, ..., grad_param_N, grad_buf_0, ..., grad_input_0, ...]
    # (Only for primals that require grad.)
    for i, fqn in enumerate(param_fqns):
        if fqn in ref_param_grads and i < len(bwd_results):
            grad_local = bwd_results[i]
            if grad_local is not None and isinstance(
                grad_local, (torch.Tensor, LocalTensor)
            ):
                placements = primal_placements[i]
                full_grad = _gather_shards(grad_local, mesh_shape, placements)
                torch.testing.assert_close(
                    full_grad,
                    ref_param_grads[fqn],
                    atol=atol,
                    rtol=rtol,
                    msg=f"Gradient mismatch for param {fqn}",
                )

    # Input gradients come after param + buffer gradients.
    input_grad_offset = num_params + num_buffers
    for i, ref_grad in enumerate(ref_input_grads):
        idx = input_grad_offset + i
        if idx < len(bwd_results):
            grad_local = bwd_results[idx]
            if grad_local is not None and isinstance(
                grad_local, (torch.Tensor, LocalTensor)
            ):
                placements = primal_placements[idx]
                full_grad = _gather_shards(grad_local, mesh_shape, placements)
                torch.testing.assert_close(
                    full_grad,
                    ref_grad,
                    atol=atol,
                    rtol=rtol,
                    msg=f"Gradient mismatch for input {i}",
                )


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


def test_correctness_linear():
    dim = 64
    batch = 8

    def model_fn():
        return nn.Linear(dim, dim, bias=False)

    def input_fn():
        return torch.randn(batch, dim, device="cuda")

    _run_correctness_test(model_fn, input_fn, mesh_shape=(4, 2))


def test_correctness_linear_with_bias():
    dim = 64
    batch = 8

    def model_fn():
        return nn.Linear(dim, dim, bias=True)

    def input_fn():
        return torch.randn(batch, dim, device="cuda")

    _run_correctness_test(model_fn, input_fn, mesh_shape=(4, 2))


def test_correctness_linear_1d_mesh():
    dim = 64
    batch = 8

    def model_fn():
        return nn.Linear(dim, dim, bias=False)

    def input_fn():
        return torch.randn(batch, dim, device="cuda")

    _run_correctness_test(model_fn, input_fn, mesh_shape=(8,))


def test_correctness_mlp():
    dim = 64
    hidden = 128
    batch = 8

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(dim, hidden, bias=False)
            self.fc2 = nn.Linear(hidden, dim, bias=False)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    def model_fn():
        return MLP()

    def input_fn():
        return torch.randn(batch, dim, device="cuda")

    _run_correctness_test(model_fn, input_fn, mesh_shape=(4, 2))


def test_correctness_attention():
    dim = 64
    nheads = 4
    seq_len = 16
    batch = 8

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.nheads = nheads
            self.wq = nn.Linear(dim, dim, bias=False)
            self.wk = nn.Linear(dim, dim, bias=False)
            self.wv = nn.Linear(dim, dim, bias=False)
            self.wo = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            q = self.wq(x).unflatten(-1, (self.nheads, -1)).transpose(1, 2)
            k = self.wk(x).unflatten(-1, (self.nheads, -1)).transpose(1, 2)
            v = self.wv(x).unflatten(-1, (self.nheads, -1)).transpose(1, 2)
            o = nn.functional.scaled_dot_product_attention(q, k, v)
            o = o.transpose(1, 2).flatten(-2)
            return self.wo(o)

    def model_fn():
        return Attention()

    def input_fn():
        return torch.randn(batch, seq_len, dim, device="cuda")

    _run_correctness_test(model_fn, input_fn, mesh_shape=(4, 2))
