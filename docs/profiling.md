# Profiling and instruction inspection

Use a release build with line information, then move from the broad view to the
specific question:

```bash
cmake -S cuda -B build/cuda -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build/cuda -j
ncu --set basic --target-processes all build/cuda/naive_gemm
scripts/dump_sass.sh build/cuda/naive_gemm > /tmp/naive-sm80.sass
```

Use architecture `90` on H100. Start with elapsed cycles, achieved occupancy,
DRAM/L2 throughput, shared-memory conflicts, tensor-pipe utilization, registers,
and spills. Add metric sets only to answer a question; collecting everything can
change runtime and produces noisy reports.

For Triton, set `TRITON_CACHE_DIR` to a known scratch location when studying
generated artifacts. Treat cache paths as implementation details that can change
between Triton versions, and record that version in the experiment note.

