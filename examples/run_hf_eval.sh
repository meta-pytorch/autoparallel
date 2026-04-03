#!/bin/bash
# Run ~30 HF models through example_hf.py with 1D and 2D meshes.
# Covers causal LMs (decoder-only), seq2seq (encoder-decoder), and
# masked LMs (encoder-only).
# Outputs go to /tmp/hf_eval_results/

set -e

OUTDIR=/tmp/hf_eval_results
mkdir -p "$OUTDIR"

# Each entry is "task|model". Default task is causal-lm.
ENTRIES=(
    # --- Decoder-only (causal LM) ---
    "causal-lm|openai-community/gpt2-medium"          # GPT-2, 355M
    "causal-lm|openai-community/gpt2-xl"              # GPT-2, 1.5B
    "causal-lm|EleutherAI/gpt-neo-2.7B"              # GPT-Neo, 2.7B
    "causal-lm|EleutherAI/gpt-j-6b"                  # GPT-J, 6B
    "causal-lm|EleutherAI/pythia-160m"               # GPT-NeoX, 160M
    "causal-lm|EleutherAI/pythia-6.9b"               # GPT-NeoX, 6.9B
    "causal-lm|bigscience/bloom-560m"                # BLOOM (ALiBi), 560M
    "causal-lm|bigscience/bloom-7b1"                 # BLOOM (ALiBi), 7.1B
    "causal-lm|cerebras/Cerebras-GPT-1.3B"           # Cerebras-GPT, 1.3B
    "causal-lm|Qwen/Qwen2-1.5B"                     # Qwen2 (GQA + RoPE), 1.5B
    "causal-lm|Qwen/Qwen2.5-3B"                     # Qwen2.5 (GQA + RoPE), 3B
    "causal-lm|Qwen/Qwen2.5-7B"                     # Qwen2.5 (GQA + RoPE), 7B
    "causal-lm|microsoft/phi-1"                      # Phi, 1.3B
    "causal-lm|microsoft/phi-2"                      # Phi, 2.7B
    "causal-lm|HuggingFaceTB/SmolLM2-135M"           # SmolLM2 (LLaMA-like), 135M
    "causal-lm|HuggingFaceTB/SmolLM2-1.7B"           # SmolLM2 (LLaMA-like), 1.7B
    "causal-lm|TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # LLaMA, 1.1B
    "causal-lm|stabilityai/stablelm-2-1_6b"          # StableLM2 (GQA + RoPE), 1.6B
    "causal-lm|tiiuae/Falcon3-3B-Base"               # Falcon3 (LLaMA-based), 3B
    "causal-lm|tiiuae/Falcon3-7B-Base"               # Falcon3 (LLaMA-based), 7B
    "causal-lm|mistralai/Mistral-7B-v0.1"            # Mistral (GQA, sliding window), 7B

    # --- Encoder-decoder (seq2seq) ---
    "seq2seq|google-t5/t5-small"                     # T5, 60M
    "seq2seq|google-t5/t5-base"                      # T5, 223M

    # --- Encoder-only (masked LM) ---
    "masked-lm|google-bert/bert-base-uncased"        # BERT, 110M
    "masked-lm|FacebookAI/roberta-base"              # RoBERTa, 125M
    "masked-lm|distilbert/distilbert-base-uncased"   # DistilBERT, 67M

    "causal-lm|tiiuae/falcon-7b"                     # Falcon v1, 7B

    # --- Known-failing models (regression tracking) ---
    # "causal-lm|facebook/opt-350m"                  # OPT — Dynamo graph break (layerdrop)
    "causal-lm|RWKV/rwkv-4-169m-pile"                # RWKV — broadcast shape mismatch
    # "seq2seq|facebook/bart-base"                   # BART — Dynamo graph break (layerdrop)
)

MESHES=("8" "2,4")

for entry in "${ENTRIES[@]}"; do
    task="${entry%%|*}"
    model="${entry#*|}"
    safe_name=$(echo "$model" | tr '/' '_')
    for mesh in "${MESHES[@]}"; do
        mesh_label=$(echo "$mesh" | tr ',' 'x')
        outfile="$OUTDIR/${safe_name}_${mesh_label}.txt"
        echo "=== Running $model task=$task mesh=$mesh ==="
        timeout 600 python examples/example_hf.py \
            --model "$model" \
            --mesh "$mesh" \
            --task "$task" \
            > "$outfile" 2>&1 || echo "FAILED (exit=$?)" >> "$outfile"
        tail -3 "$outfile"
        echo "---"
    done
done

echo "All runs complete. Results in $OUTDIR/"
