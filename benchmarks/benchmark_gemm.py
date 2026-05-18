"""Benchmark GEMM throughput across a wide range of matrix shapes on H100.

Produces a JSON file with (M, K, N) -> {time_us, tflops, efficiency, flops, bytes} mappings.
Shapes include:
  - Systematic sweep of powers-of-2 dimensions
  - LLaMA-3 8B shapes under various sharding configs (TP=1,2,4,8)
  - Rectangular shapes that stress different tile configurations
"""

import argparse
import json
import time

import torch


def benchmark_mm(M, K, N, dtype=torch.bfloat16, num_warmup=5, num_iters=20):
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)

    for _ in range(num_warmup):
        torch.mm(A, B)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        torch.mm(A, B)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / num_iters
    elapsed_us = elapsed_ms * 1e3
    flops = 2 * M * K * N
    tflops = flops / (elapsed_ms * 1e-3) * 1e-12
    read_write_bytes = (M * K + K * N + M * N) * A.element_size()

    return {
        "time_us": elapsed_us,
        "tflops": tflops,
        "flops": flops,
        "bytes": read_write_bytes,
    }


def generate_shapes():
    shapes = set()

    # 1) Systematic sweep: M from 128..16384, K and N from 128..16384 (powers of 2)
    dims = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    for M in dims:
        for K in dims:
            for N in dims:
                shapes.add((M, K, N))

    # 2) LLaMA-3 8B shapes (hidden=4096, ffn=14336, head_dim=128, n_heads=32, n_kv_heads=8)
    # Batch tokens = seq_len * batch_size / TP. Common: seq=2048 or 4096, batch=1..4
    # Under TP sharding, dimensions get divided
    hidden = 4096
    ffn = 14336
    head_dim = 128
    n_heads = 32
    n_kv_heads = 8

    for tp in [1, 2, 4, 8]:
        for batch_tokens in [512, 1024, 2048, 4096, 8192, 16384]:
            M_val = batch_tokens

            # QKV projections: [batch_tokens, hidden] x [hidden, (n_heads/tp)*head_dim]
            qkv_N = (n_heads // tp) * head_dim
            shapes.add((M_val, hidden, qkv_N))

            # K/V projections (GQA): [batch_tokens, hidden] x [hidden, (n_kv_heads/tp)*head_dim]
            if n_kv_heads >= tp:
                kv_N = (n_kv_heads // tp) * head_dim
                shapes.add((M_val, hidden, kv_N))

            # Output projection: [batch_tokens, (n_heads/tp)*head_dim] x [(n_heads/tp)*head_dim, hidden]
            shapes.add((M_val, qkv_N, hidden))

            # FFN gate/up: [batch_tokens, hidden] x [hidden, ffn/tp]
            if ffn % tp == 0:
                ffn_tp = ffn // tp
                shapes.add((M_val, hidden, ffn_tp))
                # FFN down: [batch_tokens, ffn/tp] x [ffn/tp, hidden]
                shapes.add((M_val, ffn_tp, hidden))

            # Sequence-parallel variants: M dimension is batch_tokens/tp
            if batch_tokens >= tp:
                M_sp = batch_tokens // tp
                shapes.add((M_sp, hidden, qkv_N))
                if n_kv_heads >= tp:
                    shapes.add((M_sp, hidden, kv_N))
                shapes.add((M_sp, qkv_N, hidden))
                if ffn % tp == 0:
                    shapes.add((M_sp, hidden, ffn_tp))
                    shapes.add((M_sp, ffn_tp, hidden))

    # 3) Backward weight-gradient shapes (transposed): [K, batch_tokens] x [batch_tokens, N]
    # These are the same FLOPs but different shapes (tall-skinny x short-wide)
    for batch_tokens in [2048, 4096, 8192, 16384]:
        for tp in [1, 2, 4, 8]:
            qkv_N = (n_heads // tp) * head_dim
            # dW = X^T @ dY: [hidden, batch_tokens] x [batch_tokens, qkv_N]
            shapes.add((hidden, batch_tokens, qkv_N))
            shapes.add((qkv_N, batch_tokens, hidden))
            if ffn % tp == 0:
                ffn_tp = ffn // tp
                shapes.add((hidden, batch_tokens, ffn_tp))
                shapes.add((ffn_tp, batch_tokens, hidden))

    # 4) Non-power-of-2 dimensions (ffn=14336 is not power of 2)
    odd_dims = [14336, 7168, 3584, 1792]
    for d in odd_dims:
        for M_val in [1024, 2048, 4096, 8192, 16384]:
            shapes.add((M_val, 4096, d))
            shapes.add((M_val, d, 4096))

    # 5) Small M (batch_size=1 inference-like)
    for M_val in [1, 2, 4, 8, 16, 32, 64]:
        for K_val in [4096, 8192, 14336]:
            for N_val in [4096, 8192, 14336]:
                shapes.add((M_val, K_val, N_val))

    return sorted(shapes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=str, default="benchmarks/gemm_benchmark_h100.json"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=20)
    parser.add_argument(
        "--max-elements",
        type=int,
        default=2**32,
        help="Skip shapes where M*K + K*N > max_elements (OOM guard)",
    )
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    shapes = generate_shapes()

    # Filter shapes that would OOM
    shapes = [
        (M, K, N) for M, K, N in shapes if M * K + K * N + M * N <= args.max_elements
    ]

    print(f"Benchmarking {len(shapes)} GEMM shapes on {torch.cuda.get_device_name()}")
    print(f"dtype={args.dtype}, warmup={args.num_warmup}, iters={args.num_iters}")

    results = {}
    peak_tflops = 989.5  # H100 bf16 peak

    t0 = time.time()
    for i, (M, K, N) in enumerate(shapes):
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(shapes) - i - 1) if i > 0 else 0
            print(
                f"[{i+1}/{len(shapes)}] M={M}, K={K}, N={N} (elapsed={elapsed:.0f}s, ETA={eta:.0f}s)"
            )

        try:
            r = benchmark_mm(
                M,
                K,
                N,
                dtype=dtype,
                num_warmup=args.num_warmup,
                num_iters=args.num_iters,
            )
            r["efficiency"] = r["tflops"] / peak_tflops
            key = f"{M},{K},{N}"
            results[key] = r
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM: M={M}, K={K}, N={N}, skipping")
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\nDone. {len(results)} shapes benchmarked in {elapsed:.0f}s")
    print(f"Saving to {args.output}")

    with open(args.output, "w") as f:
        json.dump(
            {
                "device": torch.cuda.get_device_name(),
                "dtype": args.dtype,
                "peak_tflops": peak_tflops,
                "num_iters": args.num_iters,
                "results": results,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
