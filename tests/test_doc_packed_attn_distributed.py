# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from autoparallel import AutoParallel
from autoparallel.compile import autoparallel_backend
from autoparallel.ops import doc_packed_attn


class _CrossAttention(nn.Module):
    def forward(self, q, k, v, cu_q, cu_k):
        return doc_packed_attn(q, k, v, cu_q, n_docs=3, cu_seq_k=cu_k, enable_gqa=True)


class TestDocPackedAttnDistributed(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four GPUs")
    @with_comms
    def test_compiled_cross_attention_dp_and_dp_tp(self):
        torch.manual_seed(0)
        B, S_q, S_k, H_q, H_kv, D = 8, 32, 64, 8, 4, 64
        q = torch.randn(B, S_q, H_q, D, dtype=torch.bfloat16, device=self.device_type)
        k = torch.randn(B, S_k, H_kv, D, dtype=torch.bfloat16, device=self.device_type)
        v = torch.randn_like(k)
        q_bounds = [[0, 8, 32, 32], [0, 20, 32, 32]] * (B // 2)
        k_bounds = [[0, 48, 64, 64], [0, 16, 64, 64]] * (B // 2)
        cu_q = torch.tensor(q_bounds, dtype=torch.int32, device=self.device_type)
        cu_k = torch.tensor(k_bounds, dtype=torch.int32, device=self.device_type)
        grad = torch.randn_like(q)
        reference_inputs = [t.detach().float().requires_grad_(True) for t in (q, k, v)]
        q_ref, k_ref, v_ref = reference_inputs
        rows = []
        for b in range(B):
            documents = []
            for i in range(2):
                q_doc = q_ref[b, q_bounds[b][i] : q_bounds[b][i + 1]]
                k_doc = k_ref[b, k_bounds[b][i] : k_bounds[b][i + 1]]
                v_doc = v_ref[b, k_bounds[b][i] : k_bounds[b][i + 1]]
                out = nn.functional.scaled_dot_product_attention(
                    q_doc.transpose(0, 1).unsqueeze(0),
                    k_doc.transpose(0, 1).unsqueeze(0),
                    v_doc.transpose(0, 1).unsqueeze(0),
                    enable_gqa=True,
                )
                documents.append(out.squeeze(0).transpose(0, 1))
            rows.append(torch.cat(documents))
        reference = torch.stack(rows)
        reference.backward(grad.float())
        global_inputs = (q, k, v, cu_q, cu_k)

        def input_fn():
            return tuple(
                torch.empty(
                    t.shape,
                    dtype=t.dtype,
                    device=self.device_type,
                    requires_grad=t.is_floating_point(),
                )
                for t in global_inputs
            )

        for mesh_shape in ((self.world_size,), (2, 2)):
            mesh = init_device_mesh(
                self.device_type,
                mesh_shape,
                mesh_dim_names=("dp",) if len(mesh_shape) == 1 else ("dp", "tp"),
            )
            placements = (Shard(0),) if mesh.ndim == 1 else (Shard(0), Shard(2))
            boundary_placements = (
                (Shard(0),) if mesh.ndim == 1 else (Shard(0), Replicate())
            )
            input_placements = [placements] * 3 + [boundary_placements] * 2
            with AutoParallel(
                _CrossAttention(),
                input_fn,
                mesh,
                dynamic=True,
                repeated_subgraphs=False,
                solver_time_limit_seconds=60,
            ) as autop:
                autop.add_input_constraints(input_placements)
                autop.add_output_constraints([placements])
                placement = autop.optimize_placement()
                parallel_model = autop.apply_placement(placement)

            compiled = torch.compile(
                parallel_model,
                backend=autoparallel_backend(enable_ac=False, overlap_scheduling=False),
                fullgraph=True,
            )
            local_inputs = [
                distribute_tensor(t, mesh, p).to_local().detach().requires_grad_(i < 3)
                for i, (t, p) in enumerate(
                    zip(global_inputs, input_placements, strict=True)
                )
            ]
            output = compiled(*local_inputs)
            global_output = DTensor.from_local(
                output.detach(), mesh, placements, shape=q.shape, stride=q.stride()
            ).full_tensor()
            tolerance = 4 * torch.finfo(q.dtype).eps
            relative_tolerance = 2 * torch.finfo(q.dtype).eps
            self.assertEqual(
                global_output.float(),
                reference.detach(),
                atol=tolerance,
                rtol=relative_tolerance,
            )
            output.backward(distribute_tensor(grad, mesh, placements).to_local())
            for local, ref, p in zip(
                local_inputs[:3], reference_inputs, input_placements[:3], strict=True
            ):
                self.assertIsNotNone(local.grad)
                expected = distribute_tensor(ref.grad, mesh, p).to_local()
                self.assertEqual(
                    local.grad.float(),
                    expected,
                    atol=tolerance,
                    rtol=relative_tolerance,
                )


if __name__ == "__main__":
    run_tests()
