# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path

import pytest
import torch

from autoparallel._testing.models.dsv3 import DeepSeekV3Model


def test_dsv3_accepts_torchtitan_grouped_experts_config():
    torchtitan_root = Path(__file__).resolve().parents[2] / "torchtitan"
    if not torchtitan_root.exists():
        pytest.skip("torchtitan sibling checkout not found")
    sys.path.insert(0, str(torchtitan_root))

    try:
        from torchtitan.models.deepseek_v3 import (
            deepseekv3_configs,  # type: ignore[import-not-found]
        )
    except Exception as exc:
        pytest.skip(f"torchtitan DeepSeek-V3 config unavailable: {exc}")

    with torch.device("meta"):
        model = DeepSeekV3Model(
            deepseekv3_configs["debugmodel"](
                attn_backend="sdpa",
                moe_comm_backend="standard",
            )
        )

    moe_layer = next(layer for layer in model.layers.values() if layer.moe_enabled)
    assert moe_layer.moe.experts.use_grouped_mm
