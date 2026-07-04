"""Small, stable public API for the evolving kernel implementations."""

from .gemm import available_providers, gemm
from .layouts import LAYOUTS, make_gemm_inputs

__all__ = ["LAYOUTS", "available_providers", "gemm", "make_gemm_inputs"]
