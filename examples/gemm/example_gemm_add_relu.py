"""Compute ReLU(A @ B + C) in a fused TileLang kernel."""

import torch
import tilelang
import tilelang.language as T


def ref_program(a, b, c):
    # Accumulate and add C in FP32, then cast only the final result.
    return torch.relu(a.float() @ b.float() + c.float()).to(a.dtype)


@tilelang.jit
def matmul_add_relu(A, B, C, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C: T.Tensor((M, N), dtype)
    D = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        D_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        # Compute A @ B with FP32 accumulation.
        T.clear(D_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, D_local)

        # Fuse the matrix addition and ReLU before writing to global memory.
        for i, j in T.Parallel(block_M, block_N):
            D_local[i, j] = T.max(
                D_local[i, j] + T.cast(C[by * block_M + i, bx * block_N + j], accum_dtype), 0
            )

        T.copy(D_local, D[by * block_M, bx * block_N])

    return D


def main(M=1024, N=1024, K=1024):
    kernel = matmul_add_relu.compile(M=M, N=N, K=K, block_M=128, block_N=128, block_K=32)

    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.randn(M, N, device="cuda", dtype=torch.float16)

    d = kernel(a, b, c)
    ref_d = ref_program(a, b, c)

    print("d:")
    print(d)
    print("ref_d:")
    print(ref_d)

    torch.testing.assert_close(d, ref_d, rtol=1e-2, atol=1e-2)
    print("All checks passed.")

    print("CUDA Source:")
    print(kernel.get_kernel_source())

    profiler = kernel.get_profiler()
    latency = profiler.do_bench(backend="cupti")
    print(f"tilelang Latency: {latency}ms")


def run_regression_perf():
    kernel = matmul_add_relu.compile(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32)
    profiler = kernel.get_profiler()
    return profiler.do_bench(backend="cupti")


if __name__ == "__main__":
    main()
