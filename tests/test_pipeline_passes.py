from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.fx import GraphModule
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel.api import AutoParallel
from autoparallel.pipeline.passes import (
    split_fsdp_prefetch,
    split_fsdp_reduce_scatters_epilogue,
)


@pytest.fixture(scope="module", autouse=True)
def init_pg():
    world_size = 256
    fake_store = FakeStore()
    if torch.distributed.is_initialized():
        return
    torch.distributed.init_process_group(
        "fake", store=fake_store, rank=0, world_size=world_size
    )


@pytest.fixture(scope="module")
def device_mesh_2d():
    world_size = torch.distributed.get_world_size()
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda",
        (world_size // 8, 8),
        mesh_dim_names=(
            "dp",
            "tp",
        ),
    )
    return mesh


class FFN(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        bias = False
        self.linear1 = nn.Linear(dim1, dim2, bias=bias)
        self.linear2 = nn.Linear(dim2, dim1, bias=bias)

    def forward(self, x, y):
        return y + 2, self.linear2(self.linear1(x)), y + 2


def _make_model_and_input_fn(mesh, device="cuda"):
    bs = 2048 * mesh.shape[0]
    dim1 = 1024
    dim2 = 4096

    def model_fn():
        return FFN(dim1, dim2)

    def input_fn():
        return torch.randn(bs, dim1).to(device), torch.randn(bs, 1).to(device)

    return model_fn, input_fn


@patch("torch.cuda.device_count", lambda: 8)
@patch("torch.cuda.get_device_name", lambda device: "H100")
def test_fsdp_split_passes(device_mesh_2d):
    low_mem = 0
    high_mem = None
    model_fn, input_fn = _make_model_and_input_fn(device_mesh_2d)
    with torch.device("meta"):
        model = model_fn()

    with AutoParallel(model, input_fn, device_mesh_2d) as autop:
        autop.add_parameter_memory_constraint(low=low_mem, high=high_mem)
        sharding_placement = autop.optimize_placement()
        autop.apply_placement(sharding_placement)
    gm = autop.parallel_gm
    g = gm.graph

    def gen_g_inputs(g):
        phs = g.find_nodes(op="placeholder")
        ret = []
        for ph in phs:
            ft = ph.meta["val"]
            t = torch.randn(ft.shape, dtype=ft.dtype, device=ft.device)
            ret.append(t)
        return ret

    inputs = gen_g_inputs(g)
    g_pro, g_main = split_fsdp_prefetch(g)
    g_main, g_epi = split_fsdp_reduce_scatters_epilogue(g_main)

    gm_pro = GraphModule(gm, g_pro)
    gm_main = GraphModule(gm, g_main)
    gm_epi = GraphModule(gm, g_epi)

    gm(*inputs)
    gm_epi(*gm_main(*gm_pro(*inputs)))
