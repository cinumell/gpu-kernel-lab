import pytest

torch = pytest.importorskip("torch")

from gpu_kernel_lab import LAYOUTS, available_providers, gemm, make_gemm_inputs


def test_torch_provider_on_cpu() -> None:
    a = torch.randn(7, 11)
    b = torch.randn(11, 5)
    torch.testing.assert_close(gemm(a, b), a @ b)


def test_rejects_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        gemm(torch.randn(2, 3), torch.randn(4, 5))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_input_layouts(layout: str) -> None:
    a, b = make_gemm_inputs(7, 11, 5, layout=layout, device="cpu", dtype=torch.float32)
    assert a.shape == (7, 11)
    assert b.shape == (11, 5)
    assert (a @ b).shape == (7, 5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape", [(32, 32, 32), (37, 29, 43), (128, 65, 96)])
def test_gpu_providers(shape: tuple[int, int, int], layout: str) -> None:
    m, k, n = shape
    a, b = make_gemm_inputs(m, k, n, layout=layout)
    expected = a @ b
    for provider in available_providers():
        actual = gemm(a, b, provider=provider)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
