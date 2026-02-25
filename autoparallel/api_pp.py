# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from types import MethodType
from typing import Any, Optional

import torch
from torch._logging import trace_structured
from torch.export.unflatten import _AttrKind

from autoparallel.graph_passes.graph_partition import partition_joint_with_descriptors

from .api import AutoParallel, _assign_attr
from .init_weights import hook_params_setters


class AutoParallelPPModule(torch.nn.Module):
    def __init__(
        self,
        sharded_param_dict: dict[str, torch.nn.Parameter],
        sharded_buffer_dict: dict[str, torch.Tensor],
        init_weights_model: torch.nn.Module,
        ref_model: torch.nn.Module,
    ):
        super().__init__()
        self._register_params_and_buffers(
            sharded_param_dict, sharded_buffer_dict, ref_model
        )

        # Right now we require a convention that the user model provides an init_weights method,
        # although we could snoop for other methods too.
        if hasattr(init_weights_model, "init_weights"):
            hook_params_setters(init_weights_model, self)

            def init_weights(_self, *args, **kwargs):
                # this is now a deep-fake-copy of orig mod, so we don't have to use reparametrize
                return init_weights_model.init_weights(*args, **kwargs)

            # assign an init_weights method onto the output mod.
            # all it does is sneakily run the original user mod's init_weights method,
            # but with our new DTensor sharded params attached to the user module.
            self.init_weights = MethodType(init_weights, self)

    def _register_params_and_buffers(
        self, sharded_param_dict, sharded_buffer_dict, ref_model
    ):

        # We construct an unflattened structure on parallel_mod,
        # e.g. _assign_attr(v, parallel_model, k="layers.0.weight") will literally
        # create empty nn.Modules recursively and then stash 'v' so it shows up in the right spot
        # We pass ref_model to preserve the original module structure (e.g., nn.ModuleDict)
        for k, v in sharded_param_dict.items():
            _assign_attr(v, self, ref_model, k, attr_kind=_AttrKind.PARAMETER)

        for k, v in sharded_buffer_dict.items():
            _assign_attr(v, self, ref_model, k, attr_kind=_AttrKind.BUFFER)

    def forward(self, *args):
        raise NotImplementedError("This is a placeholder for the pipeline model")


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
        print(
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
        self.parallel_model = AutoParallelPPModule(
            sharded_param_dict,
            sharded_buffer_dict,
            self.init_weights_model,
            self.model,
        )
        return {
            "graph_callables": graph_modules,
            "graph_meta": graph_meta,
            "sharded_param_dict": sharded_param_dict,
            "sharded_buffer_dict": sharded_buffer_dict,
        }
