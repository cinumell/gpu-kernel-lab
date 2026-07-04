from __future__ import annotations

import torch

LAYOUTS = ("rr", "rc", "cr", "cc", "padded", "offset")


def _matrix(
    rows: int,
    columns: int,
    *,
    column_major: bool,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if column_major:
        # Transposing a fresh row-major allocation produces shape (rows, columns)
        # with strides (1, rows): a true column-major view without copying.
        return torch.empty((columns, rows), device=device, dtype=dtype).t()
    return torch.empty((rows, columns), device=device, dtype=dtype)


def make_gemm_inputs(
    m: int,
    k: int,
    n: int,
    *,
    layout: str = "rr",
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create logical A[M,K] and B[K,N] with a selected physical layout."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; choices: {', '.join(LAYOUTS)}")

    if layout in ("rr", "rc", "cr", "cc"):
        a = _matrix(m, k, column_major=layout[0] == "c", device=device, dtype=dtype)
        b = _matrix(k, n, column_major=layout[1] == "c", device=device, dtype=dtype)
    elif layout == "padded":
        # The logical matrices exclude the padding, but their leading dimensions
        # are larger than K and N. Seven is intentionally awkward for alignment.
        a = torch.empty((m, k + 7), device=device, dtype=dtype)[:, :k]
        b = torch.empty((k, n + 7), device=device, dtype=dtype)[:, :n]
    else:  # offset
        # Offset each row-major matrix by one element to break common 16-byte
        # alignment assumptions while preserving unit inner strides.
        a = torch.empty(m * k + 1, device=device, dtype=dtype)[1:].view(m, k)
        b = torch.empty(k * n + 1, device=device, dtype=dtype)[1:].view(k, n)

    a.normal_()
    b.normal_()
    return a, b

