import pytest
import torch
from torch.utils._debug_mode import DebugMode

from autoparallel._testing.models.llama3 import Transformer, TransformerModelArgs

# TODO: make device generic


@pytest.fixture(scope="module")
def llama3_debug_model():
    torch.manual_seed(1999)
    model_args = TransformerModelArgs(
        dim=256, n_layers=6, n_heads=16, vocab_size=2048, rope_theta=500000
    )
    return Transformer(model_args).cuda()


def test_deterministic(llama3_debug_model):
    batch_size = 8
    seqlen = 2048
    vocab_size = llama3_debug_model.model_args.vocab_size
    torch.manual_seed(2999)
    x = torch.randint(0, vocab_size, (batch_size, seqlen), device="cuda")
    torch.manual_seed(3999)
    r1 = llama3_debug_model(x)
    torch.manual_seed(3999)
    r2 = llama3_debug_model(x)
    assert torch.equal(r1, r2)  # bitwise equal


def test_debug_mode_bitwise_equivalent(llama3_debug_model):
    batch_size = 8
    seqlen = 2048
    vocab_size = llama3_debug_model.model_args.vocab_size
    torch.manual_seed(2999)
    x = torch.randint(0, vocab_size, (batch_size, seqlen), device="cuda")
    torch.manual_seed(3999)
    r1 = llama3_debug_model(x)
    torch.manual_seed(3999)
    with DebugMode() as debug_mode:
        r2 = llama3_debug_model(x)
    print(debug_mode.debug_string())
    assert torch.equal(r1, r2)  # bitwise equal


@pytest.mark.xfail
def test_aot_eager_bitwise_equivalent(llama3_debug_model):
    batch_size = 8
    seqlen = 2048
    vocab_size = llama3_debug_model.model_args.vocab_size
    torch.manual_seed(2999)
    x = torch.randint(0, vocab_size, (batch_size, seqlen), device="cuda")
    torch.manual_seed(3999)
    r1 = llama3_debug_model(x)
    with torch.profiler.profile(with_stack=True) as prof1:
        grads1 = torch.autograd.grad(r1.sum(), llama3_debug_model.parameters())
    prof1.export_chrome_trace("/tmp/profile/prof1.json")
    torch.manual_seed(3999)
    r2 = torch.compile(backend="aot_eager")(llama3_debug_model)(x)
    with torch.profiler.profile() as prof2:
        grads2 = torch.autograd.grad(r2.sum(), llama3_debug_model.parameters())
    from torch.fx.traceback import populate_stack_traces_to_kineto_trace
    prof2.export_chrome_trace("/tmp/profile/prof2.json")
    populate_stack_traces_to_kineto_trace("/tmp/profile/prof2.json")
    assert torch.equal(r1, r2)  # bitwise equal
    for g1, g2 in zip(grads1, grads2):
        assert torch.equal(g1, g2)
