# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import torch
import torch._functorch.config
import torch._inductor.config
from torch._inductor.compile_fx import compile_fx

from .graph_passes.activation_checkpointing import ac_joint_pass


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
    functorch_patches = {}
    inductor_patches = {}

    if enable_ac:
        functorch_patches["joint_custom_pass"] = _make_ac_joint_pass(
            ac_stage_size_in_GiB
        )

    if overlap_scheduling:
        inductor_patches.update(
            {
                "aten_distributed_optimizations.enable_overlap_scheduling": True,
                "aten_distributed_optimizations.collective_bucketing": True,
                "aten_distributed_optimizations.insert_overlap_deps": True,
                "aten_distributed_optimizations.max_compute_pre_fetch": 10,
            }
        )

    def backend(gm, example_inputs):
        with (
            torch._functorch.config.patch(functorch_patches),
            torch._inductor.config.patch(inductor_patches),
        ):
            return compile_fx(gm, example_inputs)

    return backend
