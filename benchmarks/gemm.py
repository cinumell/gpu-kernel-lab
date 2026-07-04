from __future__ import annotations

import argparse
import statistics

import torch

from gpu_kernel_lab import LAYOUTS, available_providers, gemm, make_gemm_inputs


def _time_ms(fn, *, warmup: int = 10, repetitions: int = 50) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GEMM providers")
    parser.add_argument("--provider", action="append", choices=available_providers())
    parser.add_argument("--shape", action="append", default=[], help="M,K,N (repeatable)")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument(
        "--layout",
        action="append",
        choices=LAYOUTS,
        help="physical A/B layout (repeatable; default: rr)",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for GPU benchmarking")
    providers = args.provider or list(available_providers())
    shapes = [tuple(map(int, shape.split(","))) for shape in args.shape]
    shapes = shapes or [(512, 512, 512), (1024, 1024, 1024), (4096, 4096, 4096)]
    dtype = getattr(torch, args.dtype)
    layouts = args.layout or ["rr"]

    print(f"gpu={torch.cuda.get_device_name()} dtype={args.dtype}")
    print(
        f"{'provider':<16} {'layout':<8} {'M,K,N':<22} "
        f"{'median ms':>10} {'TFLOP/s':>10} {'p90 ms':>10}"
    )
    for m, k, n in shapes:
        for layout in layouts:
            a, b = make_gemm_inputs(m, k, n, layout=layout, dtype=dtype)
            for provider in providers:
                samples = _time_ms(lambda: gemm(a, b, provider=provider))
                median = statistics.median(samples)
                p90 = sorted(samples)[int(0.9 * (len(samples) - 1))]
                tflops = (2 * m * n * k) / (median * 1e-3) / 1e12
                print(
                    f"{provider:<16} {layout:<8} {str((m, k, n)):<22} "
                    f"{median:>10.3f} {tflops:>10.2f} {p90:>10.3f}"
                )


if __name__ == "__main__":
    main()
