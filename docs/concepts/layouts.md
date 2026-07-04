# GEMM input layouts

Use `C[M,N] = A[M,K] @ B[K,N]` throughout. Layout describes how the logical
indices map to memory; it does not change the mathematical operation. Start with
these six cases so each experiment has a clear lesson.

| Name | A layout | B layout | What it exposes |
|---|---|---|---|
| `rr` | row-major | row-major | Simple baseline; A's K dimension and B's N dimension are contiguous |
| `rc` | row-major | column-major | Contiguous reduction loads from both A and B; changes output-column access |
| `cr` | column-major | row-major | Opposite access pressure; useful when studying cooperative tiles |
| `cc` | column-major | column-major | Both operands differ from the conventional row-major baseline |
| `padded` | row-major with `lda > K` | row-major with `ldb > N` | Leading dimensions, submatrices, pitch, and incorrect packed-memory assumptions |
| `offset` | row-major, one-element offset | row-major, one-element offset | Alignment assumptions, vectorized loads, and transaction efficiency |

Here `r` means stride-1 along the last logical dimension and `c` means stride-1
along the first. Always print strides while learning:

```python
from gpu_kernel_lab import make_gemm_inputs

a, b = make_gemm_inputs(128, 64, 96, layout="rc")
print(a.shape, a.stride())  # (128, 64), usually (64, 1)
print(b.shape, b.stride())  # (64, 96), usually (1, 64)
```

Run one shape across all layouts:

```bash
python -m benchmarks.gemm --shape 1024,1024,1024 \
  --layout rr --layout rc --layout cr --layout cc \
  --layout padded --layout offset \
  --provider torch --provider triton-naive
```

## Recommended order

1. Make `rr` correct for awkward dimensions.
2. Support `rc`, then inspect whether loads along K become simpler.
3. Add `cr` and `cc` without inserting hidden contiguous copies.
4. Add `padded` to prove the kernel respects explicit strides.
5. Add `offset` last; compare scalar and vectorized load paths.

For every case, record `shape`, `stride`, `storage_offset`, pointer alignment,
dtype, and whether an implementation materialized a copy. A fast result after an
unnoticed copy is a very persuasive little liar.

After these are stable, expand into batched/strided-batched GEMM, block-sparse
layouts, tensor-core swizzles, and Hopper TMA-compatible multidimensional tiles.
