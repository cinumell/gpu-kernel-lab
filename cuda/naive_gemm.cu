#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { \
  cudaError_t error = (call); \
  if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

__global__ void naive_gemm(const float* a, const float* b, float* c, int m, int n, int k) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;
  float accumulator = 0.0f;
  for (int inner = 0; inner < k; ++inner) accumulator += a[row * k + inner] * b[inner * n + col];
  c[row * n + col] = accumulator;
}

int main() {
  constexpr int m = 127, n = 131, k = 129;
  std::vector<float> a(m * k, 1.0f), b(k * n, 1.0f), c(m * n);
  float *da, *db, *dc;
  CUDA_CHECK(cudaMalloc(&da, a.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&db, b.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dc, c.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(da, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));
  dim3 block(16, 16), grid((n + 15) / 16, (m + 15) / 16);
  naive_gemm<<<grid, block>>>(da, db, dc, m, n, k);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(c.data(), dc, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
  for (float value : c) if (std::abs(value - k) > 1e-4f) return EXIT_FAILURE;
  std::printf("PASS: %dx%dx%d\n", m, k, n);
  CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
}

