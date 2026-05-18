"""Fit GEMM cost model parameters from benchmark data.

This script fits the Hill-function roofline model parameters used in
compute_estimation.py. The model is:

    eff(AI) = eff_max * AI^alpha / (AI^alpha + ai_half^alpha)
    time = max(flops / (peak * eff(AI)), bytes / (bw * bw_eff)) + launch

where AI = flops / bytes is the arithmetic intensity.

Usage:
    python benchmarks/fit_gemm_model.py [--benchmark benchmarks/gemm_benchmark_h100.json]
"""

import argparse
import json

import numpy as np
from scipy.optimize import minimize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        default="benchmarks/gemm_benchmark_h100.json",
        help="Path to benchmark JSON from benchmark_gemm.py",
    )
    parser.add_argument(
        "--min-m", type=int, default=64, help="Minimum M dimension for fitting"
    )
    args = parser.parse_args()

    with open(args.benchmark) as f:
        data = json.load(f)

    peak_tflops = data["peak_tflops"]
    results = data["results"]

    # H100 SXM memory bandwidth
    H100_BW = 3.35 * (1024**4)

    Ms, Ks, Ns, times = [], [], [], []
    for key, val in results.items():
        M, K, N = map(int, key.split(","))
        Ms.append(M)
        Ks.append(K)
        Ns.append(N)
        times.append(val["time_us"])

    Ms = np.array(Ms, dtype=np.float64)
    Ks = np.array(Ks, dtype=np.float64)
    Ns = np.array(Ns, dtype=np.float64)
    times = np.array(times)
    mask = Ms >= args.min_m

    def predict(params, M, K, N):
        eff_max, ai_half, alpha, bw_eff, launch = params
        fl = 2.0 * M * K * N
        byt = (M * K + K * N + M * N) * 2
        ai = fl / byt
        eff = eff_max * ai**alpha / (ai**alpha + ai_half**alpha)
        compute_t = fl / (peak_tflops * 1e12 * np.maximum(eff, 1e-4)) * 1e6
        mem_t = np.maximum(byt / (H100_BW * bw_eff) * 1e6, launch)
        return np.maximum(compute_t, mem_t)

    def mape_fn(params):
        pred = predict(params, Ms[mask], Ks[mask], Ns[mask])
        return np.mean(np.abs(pred - times[mask]) / times[mask])

    # Multi-start optimization
    best_loss = 1e9
    best_params = None
    for eff0 in [0.7, 0.75, 0.8]:
        for ai0 in [50, 75, 100, 150]:
            for a0 in [1.0, 1.5, 2, 4, 6, 8]:
                for bw0 in [0.5, 0.7, 0.9]:
                    try:
                        res = minimize(
                            mape_fn,
                            [eff0, ai0, a0, bw0, 7],
                            bounds=[
                                (0.3, 1.0),
                                (10, 2000),
                                (0.5, 10),
                                (0.3, 1.0),
                                (3, 15),
                            ],
                            method="L-BFGS-B",
                        )
                        if res.fun < best_loss:
                            best_loss = res.fun
                            best_params = res.x
                    except Exception:
                        pass

    p = best_params
    print("=== Fitted GEMM cost model ===")
    print(f"  eff_max = {p[0]:.4f}")
    print(f"  ai_half = {p[1]:.2f}")
    print(f"  alpha   = {p[2]:.4f}")
    print(f"  bw_eff  = {p[3]:.4f}")
    print(f"  launch  = {p[4]:.2f} us")
    print(f"  MAPE (M>={args.min_m}) = {best_loss * 100:.1f}%")

    # Compare with fixed 70% model
    flops_all = 2.0 * Ms * Ks * Ns
    bytes_all = (Ms * Ks + Ks * Ns + Ms * Ns) * 2
    old_pred = np.maximum(
        flops_all / (peak_tflops * 1e12 * 0.70) * 1e6,
        np.maximum(bytes_all / (H100_BW * 0.70) * 1e6, 7.0),
    )

    print("\n=== Comparison with fixed 70% model ===")
    for label, m in [
        (f"M >= {args.min_m}", mask),
        ("M >= 1024", Ms >= 1024),
    ]:
        new = predict(p, Ms[m], Ks[m], Ns[m])
        nm = np.mean(np.abs(new - times[m]) / times[m]) * 100
        om = np.mean(np.abs(old_pred[m] - times[m]) / times[m]) * 100
        print(f"  {label:20s}: new={nm:5.1f}%  old={om:5.1f}%  delta={om - nm:+.1f}pp")


if __name__ == "__main__":
    main()
