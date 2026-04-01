#!/bin/bash
# Run ~25 HF causal LM models through example_hf.py with 1D and 2D meshes.
# Outputs go to /tmp/hf_eval_results/

set -e

OUTDIR=/tmp/hf_eval_results
mkdir -p "$OUTDIR"

MODELS=(
    # --- One representative per architecture family ---
    "openai-community/gpt2-medium"          # GPT-2, 355M
    "openai-community/gpt2-xl"              # GPT-2, 1.5B
    "EleutherAI/gpt-neo-2.7B"              # GPT-Neo, 2.7B
    "EleutherAI/gpt-j-6b"                  # GPT-J, 6B
    "EleutherAI/pythia-160m"               # GPT-NeoX, 160M
    "EleutherAI/pythia-6.9b"               # GPT-NeoX, 6.9B
    "bigscience/bloom-560m"                # BLOOM (ALiBi), 560M
    "bigscience/bloom-7b1"                 # BLOOM (ALiBi), 7.1B
    "cerebras/Cerebras-GPT-1.3B"           # Cerebras-GPT, 1.3B
    "Qwen/Qwen2-1.5B"                     # Qwen2 (GQA + RoPE), 1.5B
    "Qwen/Qwen2.5-3B"                     # Qwen2.5 (GQA + RoPE), 3B
    "Qwen/Qwen2.5-7B"                     # Qwen2.5 (GQA + RoPE), 7B
    "microsoft/phi-1"                      # Phi, 1.3B
    "microsoft/phi-2"                      # Phi, 2.7B
    "HuggingFaceTB/SmolLM2-135M"           # SmolLM2 (LLaMA-like), 135M
    "HuggingFaceTB/SmolLM2-1.7B"           # SmolLM2 (LLaMA-like), 1.7B
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # LLaMA, 1.1B
    "stabilityai/stablelm-2-1_6b"          # StableLM2 (GQA + RoPE), 1.6B
    "tiiuae/Falcon3-3B-Base"               # Falcon3 (LLaMA-based), 3B
    "tiiuae/Falcon3-7B-Base"               # Falcon3 (LLaMA-based), 7B
    "mistralai/Mistral-7B-v0.1"            # Mistral (GQA, sliding window), 7B

    # --- Known-failing models (regression tracking) ---
    "facebook/opt-350m"                    # OPT — Dynamo graph break (layerdrop)
    "tiiuae/falcon-7b"                     # Falcon v1 — index_put TupleStrategy bug
    "google/gemma-2-2b"                    # Gemma 2 — gated (requires HF auth)
    "apple/OpenELM-1_1B"                   # OpenELM — requires trust_remote_code
    "RWKV/rwkv-4-169m-pile"                # RWKV — broadcast shape mismatch
)

MESHES=("8" "2,4")

for model in "${MODELS[@]}"; do
    safe_name=$(echo "$model" | tr '/' '_')
    for mesh in "${MESHES[@]}"; do
        mesh_label=$(echo "$mesh" | tr ',' 'x')
        outfile="$OUTDIR/${safe_name}_${mesh_label}.txt"
        echo "=== Running $model mesh=$mesh ==="
        timeout 300 python examples/example_hf.py \
            --model "$model" \
            --mesh "$mesh" \
            > "$outfile" 2>&1 || echo "FAILED (exit=$?)" >> "$outfile"
        # Print last few lines for quick status
        tail -3 "$outfile"
        echo "---"
    done
done

echo "All runs complete. Results in $OUTDIR/"
