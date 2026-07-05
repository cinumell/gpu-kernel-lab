# GPU Kernel Lab

A learning notebook in repository form: implement GPU kernels in CUDA, CuTE,
and Triton; prove correctness; benchmark them; inspect generated PTX/SASS; and
record what changes between Volta (V100, SM70), Ampere (A100, SM80), and Hopper
(H100, SM90).

## Learning path

| Stage | Implementation | Main idea |
|---|---|---|
| 00 | PyTorch | Trusted reference and measurement baseline |
| 01 | Triton naive | Program IDs, masks, pointer arithmetic |
| 02 | Triton tiled | Blocking, reuse, occupancy, autotuning |
| 03 | CUDA naive | Threads, grids, memory coalescing |
| 04 | CUDA tiled | Shared memory, vector loads, pipelining |
| 05 | CuTE SM70/80 | Layout algebra, WMMA, `mma.sync` |
| 06 | CuTE SM90 | WGMMA, TMA, warp specialization |
| 07 | SASS study | Confirm what the compiler actually emitted |

Each experiment should answer four questions: **is it correct, how fast is it,
why, and on which GPU/software stack?**

## Repository map

```text
benchmarks/           one benchmark entry point and reusable cases
cuda/                 standalone CUDA kernels and build files
docs/                 concepts, machine records, and experiment reports
scripts/              environment capture and SASS inspection helpers
src/gpu_kernel_lab/   Python API, reference op, and Triton kernels
tests/                 correctness tests (including awkward shapes)
```

The initial benchmark matrix includes six physical input layouts: `rr`, `rc`,
`cr`, `cc`, `padded`, and `offset`. See
[docs/concepts/layouts.md](docs/concepts/layouts.md) for the memory behavior each
case is designed to expose.

## Quick start on a GPU machine

Use Python 3.10+ and a PyTorch build matching the machine's CUDA driver.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev,triton]'

python scripts/capture_env.py
pytest
python -m benchmarks.gemm --provider torch --provider triton-naive
```

Triton is currently supported on Linux GPU machines; keep dependency resolution
on those hosts rather than on a macOS editing laptop.

## Build the CUDA examples

Build on the Linux machine containing the target NVIDIA GPU, CUDA Toolkit, a
C++ compiler, and CMake. For a V100 (`sm70`):

```bash
cmake -S cuda -B build/cuda \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build/cuda -j
./build/cuda/naive_gemm
```

For an A100 (`sm80`):

```bash
cmake -S cuda -B build/cuda \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build/cuda -j
./build/cuda/naive_gemm
```

For an H100, use `-DCMAKE_CUDA_ARCHITECTURES=90`. To produce a binary containing
code for multiple architectures, quote the semicolon-separated CMake list:

```bash
cmake -S cuda -B build/cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="70;80;90"
cmake --build build/cuda -j
```

### Why `Unix Makefiles`?

It is **not required by CUDA or this project**. `-G "Unix Makefiles"` tells
CMake which build-system files to generate. It makes the configure step create a
`build/cuda/Makefile`, after which either of these commands builds the project:

```bash
cmake --build build/cuda -j  # portable and preferred
make -C build/cuda -j        # invokes the generated Makefile directly
```

On most Linux installations, `Unix Makefiles` is already CMake's default, so
`-G "Unix Makefiles"` can be omitted. Naming it explicitly makes the example
predictable. If Ninja is installed, it is also a good choice:

```bash
cmake -S cuda -B build/cuda-ninja \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build/cuda-ninja -j
```

A build directory remembers its generator. Do not switch the same directory
between Make and Ninja; use a separate build directory as shown above.

## Experiment workflow

1. Add a kernel behind the same `gemm(a, b)` contract.
2. Add correctness cases: non-square, non-power-of-two, and non-aligned shapes.
3. Run warmups before timing; report median and percentiles, not one sample.
4. Save environment metadata beside results.
5. Compare effective TFLOP/s and percent of the relevant datatype peak.
6. Inspect PTX/SASS before claiming an instruction-level optimization.
7. Write a short report from `docs/experiments/_template.md`.

Start with FP32 inputs/outputs, then add TF32, FP16/BF16 tensor cores, and only
then architecture-specific paths. This keeps numerical and performance changes
separable.

## Design rules

- PyTorch is the oracle, not automatically the performance winner.
- Benchmarks synchronize the GPU and never include compilation in timed runs.
- Every result names GPU, clocks/power policy, CUDA, driver, framework, dtype,
  layouts, dimensions, and commit.
- Architecture-specific code is explicit (`sm70`, `sm80`, `sm90`), never silently used.
- Generated binaries, dumps, and benchmark output stay out of Git; conclusions
  and small machine-readable summaries belong in Git.

## Near-term milestones

See [ROADMAP.md](ROADMAP.md). The first useful change is to implement and tune a
tiled Triton GEMM, then mirror the same blocking decisions in CUDA.
