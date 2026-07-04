# GEMM model

For `C[M,N] = A[M,K] @ B[K,N]`, the conventional operation count is `2*M*N*K`.
Effective throughput is therefore:

```text
TFLOP/s = 2*M*N*K / elapsed_seconds / 1e12
```

This number is meaningful only alongside dtype, accumulation dtype, input
layout, GPU, math mode, and shape. Small GEMMs may be latency-bound; large GEMMs
can expose compute throughput; skinny GEMMs often stress a different bottleneck.

## Correctness set

Include square, rectangular, skinny, odd, and tile-boundary shapes. Compare with
an FP32-accumulating reference and select tolerances based on dtype and K.

## Optimization questions

1. Are global memory accesses coalesced?
2. How much reuse does each loaded element receive?
3. Is the shared-memory layout bank-conflict free?
4. Are copies and compute overlapped?
5. What limits occupancy: registers, shared memory, or threads?
6. Are tensor-core instructions emitted and sufficiently fed?

