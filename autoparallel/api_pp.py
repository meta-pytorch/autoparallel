# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Optional

import torch
from torch._logging import trace_structured

from autoparallel.graph_passes.graph_partition import partition_joint_with_descriptors

from .api import AutoParallel
from .module_construction import make_parallel_module

logger = logging.getLogger(__name__)


def make_pp_module(
    sharded_param_dict: dict[str, torch.nn.Parameter],
    sharded_buffer_dict: dict[str, torch.Tensor],
    ref_model: torch.nn.Module,
):
    """Create an AutoParallelPPModule that inherits from the user's model class."""
    return make_parallel_module(ref_model, sharded_param_dict, sharded_buffer_dict)


class AutoParallelPP(AutoParallel):
    def apply_placement_pp(
        self, sharding_placement=None, graph_passes: list[str] = []
    ) -> dict[str, Any]:
        assert all(
            g_pass in ["split_fsdp_collectives", "split_dI_dW"]
            for g_pass in graph_passes
        ), "Only split_fsdp_collectives and split_dI_dW_graph are supported"
        sharded_param_dict, sharded_buffer_dict = self._apply_placement_common(
            sharding_placement
        )
        num_params = len(sharded_param_dict)
        num_buffers = len(sharded_buffer_dict)
        (
            fw_module,
            bw_module,
            num_params_buffers,
            num_user_outputs,
            num_mutate_inputs,
            num_fw_outs_saved_for_bw,
            num_symints_saved_for_bw,
            _indices_of_inps_to_detach,
            adjusted_flat_args,
        ) = partition_joint_with_descriptors(self.joint_with_descriptors)
        assert num_params_buffers == (
            num_params + num_buffers
        ), f"num_params_buffers: {num_params_buffers}, num_params: {num_params}, num_buffers: {num_buffers}"
        num_input_grads = (
            len(bw_module.graph.find_nodes(op="output")[0].args[0]) - num_params_buffers
        )
        logger.info(
            f"num_params_buffers: {num_params_buffers}\n"
            f"num_user_outputs: {num_user_outputs}\n"
            f"num_mutate_inputs: {num_mutate_inputs}\n"
            f"num_input_grads: {num_input_grads}\n"
            f"num_fw_outs_saved_for_bw: {num_fw_outs_saved_for_bw}\n"
            f"num_symints_saved_for_bw: {num_symints_saved_for_bw}"
        )

        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "autoparallel_pp_fwd_graph",
                "encoding": "string",
            },
            payload_fn=lambda: fw_module.print_readable(
                print_output=False, include_stride=True, include_device=True
            ),
        )
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "autoparallel_pp_bwd_graph",
                "encoding": "string",
            },
            payload_fn=lambda: bw_module.print_readable(
                print_output=False, include_stride=True, include_device=True
            ),
        )
        unshard_module: Optional[torch.fx.GraphModule] = None
        reduce_grad_module: Optional[torch.fx.GraphModule] = None
        if "split_fsdp_collectives" in graph_passes:
            assert (
                not self.reshard_after_forward
            ), "reshard_after_forward should be False to disable FSDP all_gather in the backward pass"
            from autoparallel.graph_passes.split_fsdp_collectives import (
                split_fsdp_prefetch,
                split_fsdp_reduce_scatters_epilogue,
            )

            unshard_module, fw_module = split_fsdp_prefetch(fw_module, num_params)
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "autoparallel_pp_unshard_graph",
                    "encoding": "string",
                },
                payload_fn=lambda: unshard_module.print_readable(
                    print_output=False, include_stride=True, include_device=True
                ),
            )
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "autoparallel_pp_fwd_no_fsdp_graph",
                    "encoding": "string",
                },
                payload_fn=lambda: fw_module.print_readable(
                    print_output=False, include_stride=True, include_device=True
                ),
            )
            bw_module, reduce_grad_module = split_fsdp_reduce_scatters_epilogue(
                bw_module, num_params
            )
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "autoparallel_pp_bwd_no_fsdp_graph",
                    "encoding": "string",
                },
                payload_fn=lambda: bw_module.print_readable(
                    print_output=False, include_stride=True, include_device=True
                ),
            )
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "autoparallel_pp_reduce_grad_graph",
                    "encoding": "string",
                },
                payload_fn=lambda: reduce_grad_module.print_readable(
                    print_output=False, include_stride=True, include_device=True
                ),
            )

        bw_dI_module: Optional[torch.fx.GraphModule] = None
        bw_dW_module: Optional[torch.fx.GraphModule] = None
        if "split_dI_dW" in graph_passes:
            from autoparallel.graph_passes.split_di_dw_graph import split_di_dw_graph

            bw_dI_module, bw_dW_module, num_input_grads = split_di_dw_graph(
                bw_module,
                num_weight_gradients=num_params_buffers,
            )
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "autoparallel_pp_bw_dI_graph",
                    "encoding": "string",
                },
                payload_fn=lambda: bw_dI_module.print_readable(
                    print_output=False, include_stride=True, include_device=True
                ),
            )
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "autoparallel_pp_bw_dW_graph",
                    "encoding": "string",
                },
                payload_fn=lambda: bw_dW_module.print_readable(
                    print_output=False, include_stride=True, include_device=True
                ),
            )
            if all(
                x is None
                for x in bw_dI_module.graph.find_nodes(op="output")[0].args[0][
                    :num_input_grads
                ]
            ):
                raise RuntimeError(
                    "attempted to run split dI/dW pass on a graph that has no input gradients"
                )

        graph_meta: dict[str, int] = {
            "num_mutate_inputs": num_mutate_inputs,
            "num_user_outputs": num_user_outputs,
            "num_symints_saved_for_bw": num_symints_saved_for_bw,
            "num_params": num_params,
            "num_buffers": num_buffers,
            "num_input_grads": num_input_grads,
        }

        graph_modules: dict[str, Optional[torch.fx.GraphModule]] = {
            "fw": fw_module,
            "full_bw": bw_module,
            "bw_dI": bw_dI_module,
            "bw_dW": bw_dW_module,
            "unshard": unshard_module,
            "reduce_grad": reduce_grad_module,
        }
        self.parallel_model = make_pp_module(
            sharded_param_dict,
            sharded_buffer_dict,
            self.model,
        )
        return {
            "graph_callables": graph_modules,
            "graph_meta": graph_meta,
            "sharded_param_dict": sharded_param_dict,
            "sharded_buffer_dict": sharded_buffer_dict,
        }
