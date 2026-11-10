import tilelang
import tilelang.language as T

# 我们从阶段 0 的 **GPU 执行模型** 开始，结合仓库里的 [quickstart.py](/Users/ylzzz/code/personal/ai/tilelang/examples/quickstart.py:19) 理解：

# ```python
# with T.Kernel(
#     T.ceildiv(N, block_N),
#     T.ceildiv(M, block_M),
#     threads=128,
# ) as (bx, by):
# ```

# 这里把输出矩阵 `C[M, N]` 划分成多个小块：

# - **Grid**：所有 Block 的集合，这里有 `ceil(N/block_N) × ceil(M/block_M)` 个 Block。
# - **Block**：负责计算一个 `block_M × block_N` 的输出块，`bx、by` 是块坐标。
# - **Thread**：每个 Block 有 128 个线程，协作搬数据和计算。
# - **Warp**：在 NVIDIA GPU 上，32 个线程组成一个 Warp，因此这里每个 Block 有 4 个 Warp。

# 例子中，`M=N=1024`，`block_M=block_N=128`，所以共有 **8 × 8 = 64 个 Block**。每个 Block 负责输出矩阵中的一块 `128 × 128` 区域。

# 先抓住这个核心：**TileLang 中描述的 tile，是一组线程共同处理的数据块；一个线程通常要处理多个元素。**

# 下一步就沿着这个例子看：为什么需要 `A_shared`、`B_shared` 和 `C_local`，它们分别存在哪里。

# @tilelang.jit(target="cuda")
# target currently can be "cuda" or "hip" or "cpu".
# if not specified, it will be inferred from the input tensors during compile time
@tilelang.jit
def matmul(A, B, block_M: int, block_N: int, block_K: int):
    M, N, K = T.const("M, N, K")
    dtype = T.float16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    # Initialize Kernel Context
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        # Enable rasterization for better L2 cache locality (Optional)
        # T.use_swizzle(panel_size=10, enable=True)

        # Clear local accumulation
        T.clear(C_local)

        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            # Copy tile of A
            # This is a sugar syntax for parallelized copy
            T.copy(A[by * block_M, ko * block_K], A_shared)

            # Copy tile of B
            T.copy(B[ko * block_K, bx * block_N], B_shared)

            # Perform a tile-level GEMM on the shared buffers
            # Currently we dispatch to the cute/hip on Nvidia/AMD GPUs
            T.gemm(A_shared, B_shared, C_local)

        # relu
        for i, j in T.Parallel(block_M, block_N):
            C_local[i, j] = T.max(C_local[i, j], 0)

        # Copy result back to global memory
        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


M = 1024
N = 1024
K = 1024
block_M = 128
block_N = 128
block_K = 32

# Define the kernel (matmul) and compile/lower it into an executable module
matmul_relu_kernel = matmul.compile(M=M, N=N, K=K, block_M=block_M, block_N=block_N, block_K=block_K)
# Test the kernel in Python with PyTorch data
import torch

# Create random input tensors on the GPU
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)

# Run the kernel
c = matmul_relu_kernel(a, b)

print(c)
# Reference multiplication using PyTorch
# ref_c = torch.relu(a @ b)

# # Validate correctness
# torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

# 禁止 PyTorch 在 FP16 GEMM 中使用低精度中间归约
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

ref_c = torch.relu(a @ b)

torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


print("Kernel output matches PyTorch reference.")

# Retrieve and inspect the generated CUDA source (optional)
cuda_source = matmul_relu_kernel.get_kernel_source()
print("Generated CUDA kernel:\n", cuda_source)

# Profile latency with kernel
profiler = matmul_relu_kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

latency = profiler.do_bench()

print(f"Latency: {latency} ms")
