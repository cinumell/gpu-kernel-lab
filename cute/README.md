# CuTE experiments

Keep CuTE kernels here as focused, standalone programs grouped by architecture:

```text
cute/sm70/   Volta-era tensor core experiments, WMMA baselines
cute/sm80/   tiled layouts, cp.async, mma.sync
cute/sm90/   TMA, WGMMA, warp specialization
```

CuTE ships with NVIDIA CUTLASS. Pin the CUTLASS commit in each experiment report
instead of vendoring it initially. A first kernel should be added only after the
CUDA tiled GEMM baseline is measured, so layout algebra has a concrete mapping
to code you already understand.
