# Roadmap

## Phase 0 — Measurement discipline

- [ ] Capture A100 and H100 software/hardware environments.
- [ ] Establish PyTorch FP32/TF32/FP16 baselines for the standard shape set.
- [ ] Add JSON/CSV benchmark output and regression thresholds.

## Phase 1 — Memory hierarchy

- [x] Add a masked, naive Triton GEMM.
- [ ] Add autotuned tiled Triton GEMM.
- [ ] Add naive and shared-memory-tiled CUDA GEMMs.
- [ ] Study coalescing, bank conflicts, vectorized loads, and occupancy.

## Phase 2 — Tensor cores

- [ ] Triton `tl.dot` across FP16, BF16, and TF32 modes.
- [ ] CUDA WMMA, then inline PTX `mma.sync` on SM80.
- [ ] Express the SM80 kernel with CuTE layouts and tiled MMA.

## Phase 3 — Hopper

- [ ] Establish an SM90 tensor-core baseline.
- [ ] Add TMA copies and compare against `cp.async`-style pipelines.
- [ ] Explore WGMMA and warp-specialized producer/consumer pipelines.

## Phase 4 — Compiler output

- [ ] Automate cubin extraction and `nvdisasm` reports.
- [ ] Relate source constructs to PTX and SASS instructions.
- [ ] Track registers, spills, shared memory, occupancy, and instruction mix.

