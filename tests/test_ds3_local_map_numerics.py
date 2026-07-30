# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Compare 4-GPU AutoParallel DeepSeek-V3 numerics with a single-GPU model."""

import argparse
import gc
import json
import os
from typing import Any, cast

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard

from autoparallel import AutoParallel, MoEMeshRoles, build_moe_mesh
from autoparallel._testing.models.dsv3 import DeepSeekV3Model, make_dsv3_config

_WORLD_SIZE = 4
_SEQ_LEN = 2048
_LOCAL_BATCH_SIZE = 1
_SEED = 0
_OUTPUT_RTOL = 5e-3
_GRAD_RTOL = 2e-2

_CASES: dict[str, dict[str, int]] = {
    "efsdp_boundary": dict(dp_replicate=1, dp_shard=4, cp=1, tp=1, ep=2),
    "legacy_2d": dict(dp_replicate=2, dp_shard=2, cp=1, tp=1, ep=2),
    "flattened_ep": dict(dp_replicate=1, dp_shard=2, cp=2, tp=1, ep=4),
}


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.float()
    expected = expected.float()
    diff_norm = torch.linalg.vector_norm(actual - expected)
    expected_norm = torch.linalg.vector_norm(expected)
    if expected_norm == 0:
        return diff_norm.item()
    return (diff_norm / expected_norm).item()


def _gather_state_dict(model: torch.nn.Module, rank: int) -> dict[str, torch.Tensor]:
    state_dict = {}
    for name, value in model.state_dict().items():
        full_value = value.full_tensor() if isinstance(value, DTensor) else value
        if rank == 0:
            state_dict[name] = full_value.detach().cpu()
    return state_dict


def _run_reference(
    config,
    state_dict: dict[str, torch.Tensor],
    tokens: torch.Tensor,
    device: torch.device,
    reference_mesh,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    roles = MoEMeshRoles(ep_axis_names=("ep",), ep_group_name="ep")
    with torch.device("meta"):
        model = DeepSeekV3Model(
            config,
            mesh=reference_mesh,
            roles=roles,
            compute_dtype=torch.bfloat16,
        )
    model.to_empty(device=device)
    model.init_weights(buffer_device=device, seed=_SEED)
    model.load_state_dict(state_dict, strict=True)

    with reference_mesh, torch.autograd.set_multithreading_enabled(False):
        output = model(tokens)
        output.backward(torch.ones_like(output))

    gradients = {
        name: parameter.grad.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    output = output.detach().cpu()
    del model
    torch.cuda.empty_cache()
    return output, gradients


def run_numerics_test(case: str):
    rank = torch.distributed.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    degrees = _CASES[case]
    mesh, roles = build_moe_mesh(
        dp_replicate=degrees["dp_replicate"],
        dp_shard=degrees["dp_shard"],
        cp=degrees["cp"],
        tp=degrees["tp"],
        ep=degrees["ep"],
    )
    config = make_dsv3_config(num_experts=8, max_seq_len=_SEQ_LEN)
    global_batch_size = _LOCAL_BATCH_SIZE * _WORLD_SIZE

    with torch.device("meta"):
        model = DeepSeekV3Model(
            config,
            mesh=mesh,
            roles=roles,
            compute_dtype=torch.bfloat16,
        )

    sample_input = torch.empty(
        global_batch_size, _SEQ_LEN, dtype=torch.int64, device=device
    )

    def input_fn():
        return sample_input

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    with AutoParallel(
        model, input_fn, mesh, mp_policy=mp_policy, dynamic=True
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        input_placements = (Shard(0),) * mesh.ndim
        autop.add_input_constraints([input_placements])
        autop.add_output_constraints([input_placements])
        placement = autop.optimize_placement(verbose=False)
        parallel_model = autop.apply_placement(placement)

    parallel_model.to_empty(device=device)
    parallel_model.init_weights(buffer_device=device, seed=_SEED)
    reference_state = _gather_state_dict(parallel_model, rank)

    if rank == 0:
        torch.manual_seed(_SEED)
        global_tokens = torch.randint(
            0,
            config.vocab_size,
            (global_batch_size, _SEQ_LEN),
            device=device,
        )
        scatter_list = list(global_tokens.chunk(_WORLD_SIZE))
    else:
        global_tokens = None
        scatter_list = None
    local_tokens = torch.empty(
        _LOCAL_BATCH_SIZE, _SEQ_LEN, dtype=torch.int64, device=device
    )
    torch.distributed.scatter(local_tokens, scatter_list, src=0)

    reference_mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("ep",))
    if rank == 0:
        assert global_tokens is not None
        reference_output, reference_gradients = _run_reference(
            config, reference_state, global_tokens, device, reference_mesh
        )
    else:
        reference_output = None
        reference_gradients = None
    torch.distributed.barrier(device_ids=[local_rank])

    with torch.autograd.set_multithreading_enabled(False):
        output = parallel_model(local_tokens)
        output.backward(torch.ones_like(output))

    full_output = DTensor.from_local(
        output.detach(), mesh, input_placements, run_check=False
    ).full_tensor()
    output_error = None
    output_finite = False
    if rank == 0:
        assert reference_output is not None
        output_error = _relative_error(full_output.cpu(), reference_output)
        output_finite = bool(torch.isfinite(full_output).all()) and bool(
            torch.isfinite(reference_output).all()
        )

    parallel_parameters = dict(parallel_model.named_parameters())
    if rank == 0:
        assert reference_gradients is not None
        reference_names = list(reference_gradients)
    else:
        reference_names = []
    objects: list[Any] = [reference_names]
    torch.distributed.broadcast_object_list(objects, src=0, device=device)
    reference_names = cast(list[str], objects[0])
    assert set(parallel_parameters) == set(reference_names)
    gradient_errors: dict[str, float] = {}
    nonfinite_gradients: list[str] = []
    for name in parallel_parameters:
        gradient = parallel_parameters[name].grad
        assert gradient is not None, f"Missing parallel gradient for {name}"
        full_gradient = (
            gradient.full_tensor() if isinstance(gradient, DTensor) else gradient
        )
        if rank == 0:
            assert reference_gradients is not None
            error = _relative_error(full_gradient.cpu(), reference_gradients[name])
            gradient_errors[name] = error
            if (
                not torch.isfinite(full_gradient).all()
                or not torch.isfinite(reference_gradients[name]).all()
            ):
                nonfinite_gradients.append(name)

    if rank == 0:
        assert reference_gradients is not None
        assert output_error is not None
        worst_name, worst_error = max(gradient_errors.items(), key=lambda item: item[1])
        result = {
            "case": case,
            "degrees": degrees,
            "gradient_count": len(reference_gradients),
            "gradient_relative_errors": gradient_errors,
            "model": {
                "global_batch_size": global_batch_size,
                "layers": len(config.layers),
                "seed": _SEED,
                "seq_len": _SEQ_LEN,
            },
            "nonfinite_gradients": nonfinite_gradients,
            "output_finite": output_finite,
            "output_relative_error": output_error,
            "thresholds": {
                "gradient_relative_error": _GRAD_RTOL,
                "output_relative_error": _OUTPUT_RTOL,
            },
            "worst_gradient": {
                "name": worst_name,
                "relative_error": worst_error,
            },
        }
        print(f"NUMERICS_RESULT={json.dumps(result, sort_keys=True)}", flush=True)
        failures = []
        if not output_finite:
            failures.append("non-finite output")
        if nonfinite_gradients:
            failures.append(f"non-finite gradients: {', '.join(nonfinite_gradients)}")
        if output_error >= _OUTPUT_RTOL:
            failures.append(f"output relative error {output_error} >= {_OUTPUT_RTOL}")
        for name, error in gradient_errors.items():
            if error >= _GRAD_RTOL:
                failures.append(
                    f"gradient {name} relative error {error} >= {_GRAD_RTOL}"
                )
        failure = "\n".join(failures) if failures else None
    else:
        failure = None
    status = [failure]
    torch.distributed.broadcast_object_list(status, src=0, device=device)
    assert status[0] is None, status[0]

    del parallel_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.barrier(device_ids=[local_rank])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=_CASES)
    args = parser.parse_args()

    if (
        int(os.environ.get("WORLD_SIZE", "1")) != _WORLD_SIZE
        or torch.cuda.device_count() < _WORLD_SIZE
    ):
        parser.error(f"run with torchrun --standalone --nproc-per-node {_WORLD_SIZE}")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.distributed.init_process_group("nccl", device_id=device)
    torch.use_deterministic_algorithms(True)
    try:
        run_numerics_test(args.case)
    finally:
        torch.distributed.barrier(device_ids=[local_rank])
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
