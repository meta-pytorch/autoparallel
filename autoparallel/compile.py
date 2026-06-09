# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import operator
from contextlib import contextmanager
from typing import Any, Optional, Sequence, Union

import torch
import torch._functorch.config
import torch._inductor.config
from torch._inductor.compile_fx import compile_fx
from torch._inductor.custom_graph_pass import CustomPartitionerFn

from .api import _suppress_wait_tensor_side_effect
from .graph_passes.activation_checkpointing import ac_joint_pass

_INDUCTOR_OVERLAP_PATCHES = {
    "aten_distributed_optimizations.enable_overlap_scheduling": True,
    "aten_distributed_optimizations.collective_bucketing": True,
    "aten_distributed_optimizations.insert_overlap_deps": True,
    "aten_distributed_optimizations.max_compute_pre_fetch": 10,
}


class _SaveAllPartitioner(CustomPartitionerFn):
    """Reproduce the first partitioner's save/recompute decisions via tags.

    AutoParallel partitions the joint graph twice: once inside apply_placement
    via aot_compile_joint_with_descriptors (the "first" partitioner), and
    again when the user calls torch.compile(parallel_mod, backend=...).

    The first compilation tags each forward output that it decided to save
    with custom.ap_must_save in _boxed_nop_preserve_node_meta(tag_forward=True).
    Those tags propagate to the second compilation's joint graph through
    preserve_node_meta. This partitioner reads the tags and saves exactly
    those nodes — sidestepping min-cut, which would make independent
    cost-based decisions on the second joint graph and may save FSDP
    allgather outputs that should be recomputed via FSDP prefetch.

    Specifically, when ac_joint_pass runs in the second compilation it
    adds PREFER_RECOMPUTE tags to compute ops, which causes min-cut to
    recompute matmuls in backward. The backward then needs unsharded
    weights to redo those computations, force_save_collectives marks the
    allgather outputs as MUST_SAVE, and min-cut saves them. This
    partitioner avoids the chain by ignoring those tags entirely and
    saving only what the first partitioner already chose.
    """

    def __call__(
        self,
        gm: torch.fx.GraphModule,
        joint_inputs: Sequence[object],
        **kwargs: Any,
    ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
        num_fwd_outputs: int = kwargs.pop("num_fwd_outputs")  # type: ignore[assignment]
        static_lifetime_input_indices: list[int] | None = kwargs.pop(  # type: ignore[assignment]
            "static_lifetime_input_indices", None
        )
        from torch._functorch.partitioners import (
            _extract_fwd_bwd_modules,
            _is_assert_only_symbool,
            classify_nodes,
            cleanup_recompute_tags,
            default_partition,
            force_save_bw_mutation_src,
            force_save_collectives,
            force_save_effectful_ops,
            functionalize_rng_ops,
            has_recomputable_ops,
            has_recomputable_rng_ops,
            is_opaque_node,
            is_sym_node,
            must_recompute,
            raise_getitems,
            reordering_to_mimic_autograd_engine,
            thread_graphsafe_rng_from_hops,
        )

        gm.graph.eliminate_dead_code()
        gm.recompile()

        # CSE merges duplicate allgather chains: the forward's
        # MUST_RECOMPUTE allgathers and the baked-in backward copies
        # compute the same values from the same primals. Without CSE,
        # both appear in the backward (duplication).
        if torch._functorch.config.cse:
            from torch._functorch.partitioners import fx_graph_cse

            cse_graph = fx_graph_cse(gm.graph)
            gm.graph = cse_graph

        graph_has_recomputable_ops = has_recomputable_ops(gm)
        graph_has_recomputable_rng_ops = has_recomputable_rng_ops(gm)
        if graph_has_recomputable_ops:
            gm = cleanup_recompute_tags(gm, is_default_partition=False)

        # Apply PyTorch's standard save-forcing passes. None of these affect
        # our own save decision (which only consults `ap_must_save`), but
        # they normalize the graph by setting MUST_SAVE on collectives,
        # effectful ops, and backward-mutated values. We keep them as a
        # defense against future PyTorch internals that may consult these
        # tags during extraction. force_save_collectives correctly skips
        # nodes already tagged MUST_RECOMPUTE (e.g. FSDP allgathers), so
        # the FSDP recomputation contract is preserved.
        if not torch._functorch.config.unsafe_allow_optimization_of_collectives:
            force_save_collectives(gm)
        force_save_effectful_ops(gm)
        force_save_bw_mutation_src(gm)

        if static_lifetime_input_indices is None:
            static_lifetime_input_indices = []
        node_info = classify_nodes(gm, static_lifetime_input_indices, num_fwd_outputs)

        if len(node_info.required_bw_nodes) == 0:
            return default_partition(
                gm,
                joint_inputs,
                num_fwd_outputs=num_fwd_outputs,
                static_lifetime_input_indices=static_lifetime_input_indices,
                static_lifetime_input_nodes=node_info.static_lifetime_input_nodes,
            )

        saved_values = []
        saved_sym_nodes = []
        saved_opaque_nodes = []

        def _is_multi_output(node: torch.fx.Node) -> bool:
            return (
                all(user.target == operator.getitem for user in node.users)
                and len(node.users) > 0
            )

        def _maybe_save(node: torch.fx.Node) -> None:
            if is_sym_node(node):
                if not _is_assert_only_symbool(node):
                    saved_sym_nodes.append(node)
                return
            if _is_multi_output(node):
                # Multi-output ops tagged ap_must_save: save all their
                # getitem children (DCE removes unused ones later).
                if node.meta.get("custom", {}).get("ap_must_save"):
                    for user in node.users:
                        if user.target == operator.getitem:
                            saved_values.append(user)
                return
            if is_opaque_node(node):
                saved_opaque_nodes.append(node)
                return
            if must_recompute(node):
                return
            # Save nodes tagged ap_must_save by the first compilation.
            # These are the forward graph's output tensors from the first
            # partitioner — reproducing its save/recompute decisions.
            if node.op == "placeholder":
                saved_values.append(node)
            elif node.meta.get("custom", {}).get("ap_must_save"):
                saved_values.append(node)

        for node in node_info.required_fw_nodes:
            _maybe_save(node)

        # Unclaimed nodes (neither strictly forward nor backward) may be
        # needed by backward outputs — e.g. mutable ops like index_put that
        # are _must_be_in_forward. Save them so they're available as backward
        # inputs in _extract_fwd_bwd_modules.
        for node in node_info.unclaimed_nodes:
            _maybe_save(node)

        fw_module, bw_module = _extract_fwd_bwd_modules(
            gm,
            saved_values,
            saved_sym_nodes=saved_sym_nodes,
            saved_opaque_nodes=saved_opaque_nodes,
            num_fwd_outputs=num_fwd_outputs,
            static_lifetime_input_nodes=node_info.static_lifetime_input_nodes,
        )

        if graph_has_recomputable_ops and graph_has_recomputable_rng_ops:
            fw_module, bw_module = functionalize_rng_ops(
                gm, fw_module, bw_module, len(saved_sym_nodes)
            )
        bw_module = reordering_to_mimic_autograd_engine(bw_module)

        fw_module = raise_getitems(fw_module)
        bw_module = raise_getitems(bw_module)

        fw_module = thread_graphsafe_rng_from_hops(fw_module, is_backward=False)
        bw_module = thread_graphsafe_rng_from_hops(bw_module, is_backward=True)

        return fw_module, bw_module

    def uuid(self) -> Any:
        return None


@contextmanager
def _patch_partitioner_dce():
    """Patch the partitioner's DCE to allow wait_tensor to be eliminated.

    The partitioner uses its own is_not_collective callback that treats all
    _c10d_functional ops as impure, overriding _suppress_wait_tensor_side_effect.
    We patch it to let wait_tensor through so unused collectives get DCE'd.
    """
    import torch._functorch.partitioners as partitioners

    original = partitioners.is_not_collective

    def patched_is_not_collective(node):
        if node.target in (
            torch.ops._c10d_functional.wait_tensor,
            torch.ops._c10d_functional.wait_tensor.default,
        ):
            return False
        return original(node)

    partitioners.is_not_collective = patched_is_not_collective
    try:
        yield
    finally:
        partitioners.is_not_collective = original


def _make_ac_joint_pass(
    ac_stage_size_in_GiB: Optional[Union[float, str]] = "auto",
):
    def joint_pass(fx_g, joint_inputs):
        ac_joint_pass(fx_g.graph, ac_stage_size_in_GiB)
        return fx_g

    return joint_pass


def autoparallel_backend(
    *,
    enable_ac: bool = True,
    ac_stage_size_in_GiB: Optional[Union[float, str]] = "auto",
    overlap_scheduling: bool = True,
):
    """Return a torch.compile backend that wraps Inductor with AutoParallel-optimized
    passes (activation checkpointing, comm/compute overlap scheduling).

    Usage::

        parallel_module = AutoParallel(...)
        compiled = torch.compile(parallel_module, backend=autoparallel_backend())

    Args:
        enable_ac: Enable activation checkpointing joint pass.
        ac_stage_size_in_GiB: Memory budget per AC stage. "auto" uses
            sqrt(total_recomputable_memory).
        overlap_scheduling: Enable comm/compute overlap scheduling.
    """
    functorch_patches: dict[str, Any] = {}

    if enable_ac:
        functorch_patches["joint_custom_pass"] = _make_ac_joint_pass(
            ac_stage_size_in_GiB
        )

    # Inductor configs split by lifetime:
    # - overlap scheduling configs must persist to lazy backward compilation
    #   (which runs on the first .backward() call, after compile_fx returns).
    #   compile_fx's config_patches argument re-enters the patch when backward
    #   is later compiled out of scope.
    # - custom_partitioner_fn only runs during the synchronous joint→fwd/bwd
    #   partitioning inside compile_fx, so a context manager suffices.
    inductor_persistent_patches = (
        _INDUCTOR_OVERLAP_PATCHES if overlap_scheduling else None
    )
    inductor_fwd_patches: dict[str, Any] = {
        "custom_partitioner_fn": _SaveAllPartitioner(),
    }

    def backend(gm, example_inputs):
        with (
            _suppress_wait_tensor_side_effect(),
            _patch_partitioner_dce(),
            torch._functorch.config.patch(functorch_patches),
            torch._inductor.config.patch(inductor_fwd_patches),
        ):
            return compile_fx(
                gm, example_inputs, config_patches=inductor_persistent_patches
            )

    return backend
