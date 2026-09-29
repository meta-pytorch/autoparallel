import torch

from autoparallel.api import AutoParallel
from autoparallel.compile import autoparallel_backend


def evaluate(model, input_fn, mesh, mp_policy, local_inputs):
    with AutoParallel(model, input_fn, mesh, mp_policy=mp_policy) as autop:
        placement = autop.optimize_placement(verbose=True)
        parallel_model = autop.apply_placement(placement)

    compiled = torch.compile(
        parallel_model,
        backend=autoparallel_backend(enable_ac=False),
    )
    return compiled(*local_inputs)
