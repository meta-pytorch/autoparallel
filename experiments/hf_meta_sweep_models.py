import torch
from torch import nn
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BloomConfig,
    BloomForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    MixtralConfig,
    MixtralForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    T5Config,
    T5ForConditionalGeneration,
)
from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.mixtral.modeling_mixtral import (
    MixtralDecoderLayer,
    MixtralSparseMoeBlock,
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.t5.modeling_t5 import T5Block

from experiments.hf_benchmark_models import CASES, BenchmarkCase


class DenseMixtralSparseMoeBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.hidden_dim = block.hidden_dim
        self.num_experts = block.num_experts
        self.top_k = block.top_k
        self.gate = block.gate
        self.experts = block.experts
        self.jitter_noise = block.jitter_noise

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(
            router_logits, dim=1, dtype=torch.float
        )
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        final_hidden_states = torch.zeros_like(hidden_states)
        for expert_idx, expert_layer in enumerate(self.experts):
            expert_weights = (
                routing_weights
                * (selected_experts == expert_idx).to(routing_weights.dtype)
            ).sum(dim=-1, keepdim=True)
            final_hidden_states = (
                final_hidden_states + expert_layer(hidden_states) * expert_weights
            )
        return (
            final_hidden_states.reshape(batch_size, sequence_length, hidden_dim),
            router_logits,
        )


def use_dense_mixtral_expert_evaluation(model):
    replaced = 0
    for module in model.modules():
        block = getattr(module, "block_sparse_moe", None)
        if isinstance(block, MixtralSparseMoeBlock):
            module.block_sparse_moe = DenseMixtralSparseMoeBlock(block)
            replaced += 1
    return replaced


def gpt2_config(hidden_size, layers, heads):
    return GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=hidden_size,
        n_layer=layers,
        n_head=heads,
        use_cache=False,
        return_dict=True,
    )


def t5_config(d_model, d_kv, d_ff, layers, heads):
    return T5Config(
        vocab_size=32128,
        d_model=d_model,
        d_kv=d_kv,
        d_ff=d_ff,
        num_layers=layers,
        num_decoder_layers=layers,
        num_heads=heads,
        use_cache=False,
        return_dict=True,
    )


def llama_config(hidden_size, intermediate_size, layers, heads, kv_heads):
    return LlamaConfig(
        vocab_size=128256,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
        use_cache=False,
        return_dict=True,
    )


def mistral_nemo_config():
    return MistralConfig(
        vocab_size=131072,
        hidden_size=5120,
        intermediate_size=14336,
        num_hidden_layers=40,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=1024 * 1024,
        sliding_window=None,
        rms_norm_eps=1e-5,
        rope_theta=1000000.0,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
        use_cache=False,
        return_dict=True,
    )


def qwen_config(hidden_size, intermediate_size, layers, heads, kv_heads):
    return Qwen2Config(
        vocab_size=151936,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        attention_bias=True,
        tie_word_embeddings=False,
        use_cache=False,
        return_dict=True,
    )


def pythia_config(hidden_size, intermediate_size, layers, heads):
    return GPTNeoXConfig(
        vocab_size=50304,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        max_position_embeddings=2048,
        rotary_pct=0.25,
        rotary_emb_base=10000,
        layer_norm_eps=1e-5,
        use_parallel_residual=True,
        tie_word_embeddings=False,
        use_cache=False,
        return_dict=True,
    )


def bert_config(hidden_size, intermediate_size, layers, heads):
    return BertConfig(
        vocab_size=30522,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        max_position_embeddings=512,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        return_dict=True,
    )


def bloom_config(hidden_size, layers, heads):
    return BloomConfig(
        vocab_size=250880,
        hidden_size=hidden_size,
        n_layer=layers,
        n_head=heads,
        seq_length=2048,
        use_cache=False,
        return_dict=True,
    )


def mixtral_config():
    return MixtralConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=32768,
        sliding_window=None,
        rms_norm_eps=1e-5,
        rope_theta=1000000.0,
        num_local_experts=8,
        num_experts_per_tok=2,
        output_router_logits=False,
        tie_word_embeddings=False,
        use_cache=False,
        return_dict=True,
    )


def mixtral_canary_config():
    return MixtralConfig(
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=1536,
        num_hidden_layers=6,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=1024,
        sliding_window=None,
        rms_norm_eps=1e-5,
        num_local_experts=4,
        num_experts_per_tok=2,
        output_router_logits=False,
        tie_word_embeddings=False,
        use_cache=False,
        return_dict=True,
    )


def case(name, task, sequence_length, config_factory, model_cls, block_cls):
    return BenchmarkCase(
        name,
        task,
        16,
        sequence_length,
        config_factory,
        model_cls,
        block_cls,
    )


MOE_CANARY_CASES = {
    entry.name: entry
    for entry in (
        case(
            "mixtral-canary-seq512",
            "causal-lm",
            512,
            mixtral_canary_config,
            MixtralForCausalLM,
            MixtralDecoderLayer,
        ),
    )
}


SWEEP_CASES = {
    entry.name: entry
    for entry in (
        case(
            "gpt2-small-seq1024",
            "causal-lm",
            1024,
            lambda: gpt2_config(768, 12, 12),
            GPT2LMHeadModel,
            GPT2Block,
        ),
        case(
            "gpt2-medium-seq1024",
            "causal-lm",
            1024,
            lambda: gpt2_config(1024, 24, 16),
            GPT2LMHeadModel,
            GPT2Block,
        ),
        CASES["gpt2-xl-seq1024"],
        case(
            "t5-base-seq512",
            "seq2seq",
            512,
            lambda: t5_config(768, 64, 3072, 12, 12),
            T5ForConditionalGeneration,
            T5Block,
        ),
        case(
            "t5-large-seq512",
            "seq2seq",
            512,
            lambda: t5_config(1024, 64, 4096, 24, 16),
            T5ForConditionalGeneration,
            T5Block,
        ),
        CASES["t5-3b-seq512"],
        case(
            "t5-11b-seq512",
            "seq2seq",
            512,
            lambda: t5_config(1024, 128, 65536, 24, 128),
            T5ForConditionalGeneration,
            T5Block,
        ),
        case(
            "llama3.2-1b-seq2048",
            "causal-lm",
            2048,
            lambda: llama_config(2048, 8192, 16, 32, 8),
            LlamaForCausalLM,
            LlamaDecoderLayer,
        ),
        case(
            "llama3.2-3b-seq2048",
            "causal-lm",
            2048,
            lambda: llama_config(3072, 8192, 28, 24, 8),
            LlamaForCausalLM,
            LlamaDecoderLayer,
        ),
        CASES["llama3-8b-seq2048"],
        case(
            "llama3-70b-seq2048",
            "causal-lm",
            2048,
            lambda: llama_config(8192, 28672, 80, 64, 8),
            LlamaForCausalLM,
            LlamaDecoderLayer,
        ),
        CASES["mistral-7b-seq2048"],
        case(
            "mistral-nemo-12b-seq2048",
            "causal-lm",
            2048,
            mistral_nemo_config,
            MistralForCausalLM,
            MistralDecoderLayer,
        ),
        case(
            "qwen2-1.5b-seq2048",
            "causal-lm",
            2048,
            lambda: qwen_config(1536, 8960, 28, 12, 2),
            Qwen2ForCausalLM,
            Qwen2DecoderLayer,
        ),
        case(
            "qwen2-7b-seq2048",
            "causal-lm",
            2048,
            lambda: qwen_config(3584, 18944, 28, 28, 4),
            Qwen2ForCausalLM,
            Qwen2DecoderLayer,
        ),
        case(
            "pythia-410m-seq1024",
            "causal-lm",
            1024,
            lambda: pythia_config(1024, 4096, 24, 16),
            GPTNeoXForCausalLM,
            GPTNeoXLayer,
        ),
        case(
            "pythia-1.4b-seq1024",
            "causal-lm",
            1024,
            lambda: pythia_config(2048, 8192, 24, 16),
            GPTNeoXForCausalLM,
            GPTNeoXLayer,
        ),
        CASES["pythia-6.9b-seq1024"],
        case(
            "bert-base-seq512",
            "masked-lm",
            512,
            lambda: bert_config(768, 3072, 12, 12),
            BertForMaskedLM,
            BertLayer,
        ),
        case(
            "bert-large-seq512",
            "masked-lm",
            512,
            lambda: bert_config(1024, 4096, 24, 16),
            BertForMaskedLM,
            BertLayer,
        ),
        CASES["deberta-v2-xxlarge-seq512"],
        case(
            "bloom-560m-seq1024",
            "causal-lm",
            1024,
            lambda: bloom_config(1024, 24, 16),
            BloomForCausalLM,
            BloomBlock,
        ),
        case(
            "bloom-7.1b-seq1024",
            "causal-lm",
            1024,
            lambda: bloom_config(4096, 30, 32),
            BloomForCausalLM,
            BloomBlock,
        ),
        case(
            "mixtral-8x7b-seq2048",
            "causal-lm",
            2048,
            mixtral_config,
            MixtralForCausalLM,
            MixtralDecoderLayer,
        ),
    )
}
