# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch

from autoparallel.cost_models.compute_estimation import DEVICE_LIMITS


def test_h100_supports_fp8_compute_estimation():
    h100 = next(limit for limit in DEVICE_LIMITS if limit.name == "H100")

    assert h100.gemm_tflops[torch.float8_e4m3fn] == 1979
    assert h100.gemm_tflops[torch.float8_e5m2] == 1979
