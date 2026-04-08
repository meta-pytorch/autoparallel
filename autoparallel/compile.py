# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from typing import Optional, Union

import torch
import torch._functorch.config
import torch._inductor.config

from .graph_passes.activation_checkpointing import ac_joint_pass


def _make_ac_joint_pass(
    ac_stage_size_in_GiB: Optional[Union[float, str]] = "auto",
):
    def joint_pass(fx_g, joint_inputs):
        ac_joint_pass(fx_g.graph, ac_stage_size_in_GiB)
        return fx_g

    return joint_pass


@contextmanager
def inductor_config(
    *,
    enable_ac: bool = True,
    ac_stage_size_in_GiB: Optional[Union[float, str]] = "auto",
    overlap_scheduling: bool = True,
):
    """Configure Inductor with AutoParallel-optimized passes for torch.compile.

    Usage::

        parallel_module = AutoParallel(...)

        with inductor_config():
            compiled = torch.compile(parallel_module)

    Args:
        enable_ac: Enable activation checkpointing joint pass.
        ac_stage_size_in_GiB: Memory budget per AC stage. "auto" uses
            sqrt(total_recomputable_memory).
        overlap_scheduling: Enable comm/compute overlap scheduling.
    """
    saved_functorch = {}
    saved_inductor = {}

    if enable_ac:
        saved_functorch["joint_custom_pass"] = torch._functorch.config.joint_custom_pass
        torch._functorch.config.joint_custom_pass = _make_ac_joint_pass(
            ac_stage_size_in_GiB
        )

    if overlap_scheduling:
        saved_inductor["aten_distributed_optimizations.enable_overlap_scheduling"] = (
            torch._inductor.config.aten_distributed_optimizations.enable_overlap_scheduling
        )
        saved_inductor["aten_distributed_optimizations.collective_bucketing"] = (
            torch._inductor.config.aten_distributed_optimizations.collective_bucketing
        )
        saved_inductor["aten_distributed_optimizations.insert_overlap_deps"] = (
            torch._inductor.config.aten_distributed_optimizations.insert_overlap_deps
        )
        saved_inductor["aten_distributed_optimizations.max_compute_pre_fetch"] = (
            torch._inductor.config.aten_distributed_optimizations.max_compute_pre_fetch
        )
        torch._inductor.config.aten_distributed_optimizations.enable_overlap_scheduling = (
            True
        )
        torch._inductor.config.aten_distributed_optimizations.collective_bucketing = (
            True
        )
        torch._inductor.config.aten_distributed_optimizations.insert_overlap_deps = True
        torch._inductor.config.aten_distributed_optimizations.max_compute_pre_fetch = 10

    try:
        yield
    finally:
        if enable_ac:
            torch._functorch.config.joint_custom_pass = saved_functorch[
                "joint_custom_pass"
            ]
        if overlap_scheduling:
            torch._inductor.config.aten_distributed_optimizations.enable_overlap_scheduling = saved_inductor[
                "aten_distributed_optimizations.enable_overlap_scheduling"
            ]
            torch._inductor.config.aten_distributed_optimizations.collective_bucketing = saved_inductor[
                "aten_distributed_optimizations.collective_bucketing"
            ]
            torch._inductor.config.aten_distributed_optimizations.insert_overlap_deps = saved_inductor[
                "aten_distributed_optimizations.insert_overlap_deps"
            ]
            torch._inductor.config.aten_distributed_optimizations.max_compute_pre_fetch = saved_inductor[
                "aten_distributed_optimizations.max_compute_pre_fetch"
            ]
