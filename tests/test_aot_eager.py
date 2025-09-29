from autoparallel._testing.models.llama3 import Transformer, TransformerModelArgs
import pytest
import torch

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
    x = torch.randint(0, vocab_size, (batch_size, seqlen), device='cuda')
    torch.manual_seed(3999)
    r1 = llama3_debug_model(x)
    torch.manual_seed(3999)
    r2 = llama3_debug_model(x)
    assert torch.equal(r1, r2)  # bitwise equal

@pytest.xfail('aot_eager bitwise equivalence WIP')
def test_aot_eager_bitwise_equivalent(llama3_debug_model):
    batch_size = 8
    seqlen = 2048
    vocab_size = llama3_debug_model.model_args.vocab_size
    torch.manual_seed(2999)
    x = torch.randint(0, vocab_size, (batch_size, seqlen), device='cuda')
    torch.manual_seed(3999)
    r1 = llama3_debug_model(x)
    torch.manual_seed(3999)
    r2 = torch.compile(backend="aot_eager")(llama3_debug_model)(x)
    assert torch.equal(r1, r2)  # bitwise equal
