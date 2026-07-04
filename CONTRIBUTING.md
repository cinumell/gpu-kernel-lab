# Contributing experiments

Keep changes small enough that one performance claim can be evaluated at a
time. A kernel change should normally include:

- a descriptive implementation name rather than replacing an older stage;
- correctness tests on aligned and awkward dimensions;
- a reproducible benchmark command;
- an experiment note with environment and profiler evidence;
- no committed generated binaries or large profiler captures.

Format Python with Ruff and build CUDA with an explicit architecture. Commit
messages can follow `area: result`, for example `triton: add tiled fp16 gemm`.

