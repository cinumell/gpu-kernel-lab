from __future__ import annotations

from collections.abc import Callable

import torch

Gemm = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _torch_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)


def _providers() -> dict[str, Gemm]:
    providers: dict[str, Gemm] = {"torch": _torch_gemm}
    try:
        from .triton.naive_gemm import triton_naive_gemm
    except (ImportError, RuntimeError):
        pass
    else:
        providers["triton-naive"] = triton_naive_gemm
    return providers


def available_providers() -> tuple[str, ...]:
    return tuple(_providers())


def gemm(a: torch.Tensor, b: torch.Tensor, *, provider: str = "torch") -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("gemm expects two rank-2 tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible GEMM shapes: {tuple(a.shape)} and {tuple(b.shape)}")
    try:
        implementation = _providers()[provider]
    except KeyError as error:
        choices = ", ".join(available_providers())
        raise ValueError(f"unknown or unavailable provider {provider!r}; choices: {choices}") from error
    return implementation(a, b)

