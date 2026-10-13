# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from transformers import (
    DebertaV2Config,
    DebertaV2ForMaskedLM,
    GPT2Config,
    GPT2LMHeadModel,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    PretrainedConfig,
    T5Config,
    T5ForConditionalGeneration,
)
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Layer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.t5.modeling_t5 import T5Block


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    task: str
    global_batch: int
    sequence_length: int
    config_factory: Callable[[], PretrainedConfig]
    model_cls: type[nn.Module]
    block_cls: type[nn.Module]


class TrainingLossModule(nn.Module):
    def __init__(self, model: nn.Module, task: str) -> None:
        super().__init__()
        self.model = model
        self.task = task

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if self.task == "causal-lm":
            input_ids, labels = inputs
            return self.model(input_ids=input_ids, labels=labels).loss
        if self.task == "seq2seq":
            input_ids, attention_mask, decoder_input_ids, labels = inputs
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
            ).loss
        if self.task == "masked-lm":
            input_ids, labels = inputs
            return self.model(input_ids=input_ids, labels=labels).loss
        raise ValueError(f"unknown task: {self.task}")


def make_inputs(
    case: BenchmarkCase, batch: int, device: str
) -> tuple[torch.Tensor, ...]:
    config = case.config_factory()
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (batch, case.sequence_length),
        device=device,
    )
    labels = torch.randint(
        0,
        config.vocab_size,
        (batch, case.sequence_length),
        device=device,
    )
    if case.task != "seq2seq":
        return input_ids, labels
    attention_mask = torch.ones(
        batch,
        case.sequence_length,
        dtype=torch.long,
        device=device,
    )
    decoder_input_ids = torch.randint(
        0,
        config.vocab_size,
        (batch, case.sequence_length),
        device=device,
    )
    return input_ids, attention_mask, decoder_input_ids, labels


def _gpt2_xl_config() -> GPT2Config:
    return GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=1600,
        n_layer=48,
        n_head=25,
        use_cache=False,
        return_dict=True,
    )


def _t5_3b_config() -> T5Config:
    return T5Config(
        vocab_size=32128,
        d_model=1024,
        d_kv=128,
        d_ff=16384,
        num_layers=24,
        num_decoder_layers=24,
        num_heads=32,
        use_cache=False,
        return_dict=True,
    )


def _llama3_8b_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
        use_cache=False,
        return_dict=True,
    )


def _mistral_7b_config() -> MistralConfig:
    return MistralConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=32768,
        sliding_window=4096,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
        use_cache=False,
        return_dict=True,
    )


def _pythia_6_9b_config() -> GPTNeoXConfig:
    return GPTNeoXConfig(
        vocab_size=50304,
        hidden_size=4096,
        intermediate_size=16384,
        num_hidden_layers=32,
        num_attention_heads=32,
        max_position_embeddings=2048,
        rotary_pct=0.25,
        rotary_emb_base=10000,
        layer_norm_eps=1e-5,
        use_parallel_residual=True,
        tie_word_embeddings=False,
        use_cache=False,
        return_dict=True,
    )


def _deberta_v2_xxlarge_config() -> DebertaV2Config:
    return DebertaV2Config(
        vocab_size=128100,
        hidden_size=1536,
        num_hidden_layers=48,
        num_attention_heads=24,
        intermediate_size=6144,
        max_position_embeddings=512,
        max_relative_positions=-1,
        position_biased_input=False,
        relative_attention=True,
        pos_att_type=["p2c", "c2p"],
        share_att_key=True,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        return_dict=True,
    )


CASES = {
    case.name: case
    for case in (
        BenchmarkCase(
            "gpt2-xl-seq1024",
            "causal-lm",
            16,
            1024,
            _gpt2_xl_config,
            GPT2LMHeadModel,
            GPT2Block,
        ),
        BenchmarkCase(
            "t5-3b-seq512",
            "seq2seq",
            16,
            512,
            _t5_3b_config,
            T5ForConditionalGeneration,
            T5Block,
        ),
        BenchmarkCase(
            "llama3-8b-seq2048",
            "causal-lm",
            16,
            2048,
            _llama3_8b_config,
            LlamaForCausalLM,
            LlamaDecoderLayer,
        ),
        BenchmarkCase(
            "mistral-7b-seq2048",
            "causal-lm",
            16,
            2048,
            _mistral_7b_config,
            MistralForCausalLM,
            MistralDecoderLayer,
        ),
        BenchmarkCase(
            "pythia-6.9b-seq1024",
            "causal-lm",
            16,
            1024,
            _pythia_6_9b_config,
            GPTNeoXForCausalLM,
            GPTNeoXLayer,
        ),
        BenchmarkCase(
            "deberta-v2-xxlarge-seq512",
            "masked-lm",
            16,
            512,
            _deberta_v2_xxlarge_config,
            DebertaV2ForMaskedLM,
            DebertaV2Layer,
        ),
    )
}


def make_model(case: BenchmarkCase) -> TrainingLossModule:
    config = case.config_factory()
    return TrainingLossModule(case.model_cls(config), case.task)
