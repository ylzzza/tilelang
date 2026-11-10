"""演示分块矩阵乘法 C = A @ B.T 的实现、自动调优和性能测试。

A、B、C 的形状分别为 (M, K)、(N, K)、(M, N)，K 是累加维度。
每个 GPU 线程块负责 C 的一个二维分块，并沿 K 维分批加载数据、累加乘积。
运行时可搜索候选配置，也可直接使用按设备计算能力选取的启发式配置。
"""

import argparse
import itertools
import tilelang as tl
import tilelang.language as T
from tilelang.autotuner import AutoTuner
from tilelang.carver.template import MatmulTemplate
from tilelang.carver.arch import CUDA
from tilelang.carver.arch import CDNA
from tilelang.carver.roller.rasterization import NoRasterization
import torch


def ref_program(A, B):
    """
    使用 PyTorch 计算 A 与 B 的转置的矩阵乘积，作为正确性校验和测速的基准。

    A、B 应为二维张量，形状分别为 (M, K)、(N, K)。
    结果形状为 (M, N)，计算式为 A @ B.T，使用输入张量的数据类型。
    """
    return A @ B.T


def get_configs(M, N, K, with_roller=False, topk=20):
    """
    生成分块矩阵乘法内核的候选调优配置，每个配置用一个字典表示。

    with_roller 为 True 时，通过 MatmulTemplate 的 Roller 获取最多 topk 个
    面向当前设备、适合 Tensor Core 计算的分块建议。配置包含：
      - block_M、block_N：每个线程块计算的输出分块大小。
      - block_K：每轮沿 K 维处理的长度。
      - num_stages：流水线缓冲级数，0 表示不启用自动流水线。
      - thread_num：每个线程块使用的线程数。
      - enable_rasteration：是否启用线程块的栅格化重排（swizzle）。

    with_roller 为 False 时，枚举预设参数的笛卡尔积。
    enable_rasteration 保留原有拼写，以兼容内核的同名参数。

    参数：
        M, N, K (int)：矩阵乘法维度，供 Roller 模板生成分块建议使用。
        with_roller (bool)：是否使用考虑设备架构的 Roller 建议，默认枚举预设参数。
        topk (int)：启用 Roller 时请求的最大建议数量。

    返回：
        list[dict]：包含上述调优参数的配置列表。

    异常：
        ValueError：启用 Roller，但获取的建议为 None 时抛出。
    """
    if with_roller:
        # PyTorch 的 HIP 构建对应 AMD GPU，否则使用 NVIDIA CUDA 架构描述。
        arch = CUDA("cuda") if torch.version.hip is None else CDNA("hip")
        # 这里按 FP16 输入/输出、FP32 累加建立选参模板；
        # 模板仅用于推荐参数，下方实际调优内核的数据类型单独设为 BF16。
        carve_template = MatmulTemplate(
            M=M,
            N=N,
            K=K,
            in_dtype=T.float16,
            out_dtype=T.float16,
            accum_dtype=T.float32,
        ).with_arch(arch)

        # 确认模板能生成等价计算，再请求候选调度建议。
        func = carve_template.equivalent_function()
        assert func is not None, "Function is None"
        roller_hints = carve_template.recommend_hints(topk=topk)
        if roller_hints is None:
            raise ValueError("No Roller Hints Found for TensorCore Scheduling")
        configs = []
        for hint in roller_hints:
            config = {}
            block_m, block_n = hint.block
            warp_m, warp_n = hint.warp
            # 一个线程块可沿 M、N 两维划分为多个 warp，分别负责更小的输出分块。
            block_rows, block_cols = block_m // warp_m, block_n // warp_n
            config["block_M"] = block_m
            config["block_N"] = block_n
            config["block_K"] = hint.rstep[0]
            # Roller 的级数大于 1 时才启用流水线，否则统一设为 0。
            config["num_stages"] = hint.pipeline_stage if hint.pipeline_stage > 1 else 0
            # 按本示例每个 warp 为 32 个线程的计算方式，换算线程块的总线程数。
            config["thread_num"] = block_rows * block_cols * 32
            # 此处仅提取是否启用重排；内核中用固定 panel_size=10 的 swizzle。
            config["enable_rasteration"] = hint.rasterization_plan is not NoRasterization
            configs.append(config)
    else:
        # 手动指定搜索范围：分块尺寸、流水线级数、线程数以及是否重排线程块。
        block_M = [64, 128, 256]
        block_N = [64, 128, 256]
        block_K = [32, 64]
        num_stages = [0, 1, 2, 3]
        thread_num = [128, 256]
        enable_rasterization = [True, False]
        # 枚举所有组合，共 3 × 3 × 2 × 4 × 2 × 2 = 288 个候选配置。
        _configs = list(
            itertools.product(
                block_M,
                block_N,
                block_K,
                num_stages,
                thread_num,
                enable_rasterization,
            )
        )

        configs = [
            {
                "block_M": c[0],
                "block_N": c[1],
                "block_K": c[2],
                "num_stages": c[3],
                "thread_num": c[4],
                "enable_rasteration": c[5],  # 保留原参数名，以保持向后兼容。
            }
            for c in _configs
        ]
    return configs


def get_best_config(
    M,
    N,
    K,
    with_roller: bool = False,
    profile_backend: str = "event",
):
    """编译、校验并测量候选内核，返回耗时最短的配置及其已编译内核。

    M、N、K 固定本次计算的矩阵维度，kernel 的形参则对应可搜索的调优参数。
    with_roller 决定候选配置来源，profile_backend 决定性能测量使用的后端。
    """

    def kernel(
        block_M=None,
        block_N=None,
        block_K=None,
        num_stages=None,
        thread_num=None,
        enable_rasteration=None,
    ):
        # 每组候选参数生成一个具体内核；输入/输出用 BF16，乘加结果用 FP32 累积。
        dtype = T.bfloat16
        accum_dtype = T.float32

        # prim_func 描述交给 TileLang 编译的计算，参数 A、B、C 位于设备全局内存。
        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            # 网格的 x 轴对应输出的列分块，y 轴对应行分块。
            # ceildiv 向上取整，使末尾不足一个完整分块的区域也有线程块负责。
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
                # 将 A、B 的当前分块放入线程块内共享的内存，供矩阵乘法重复使用。
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                # fragment 表示分布在线程寄存器中的累加分块，使用 FP32 减少累加误差。
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                # 输出前先转回 BF16，并经共享内存中转写回全局内存。
                C_shared = T.alloc_shared((block_M, block_N), dtype)
                # 重排线程块与输出分块的映射，以改善相邻分块对 L2 缓存的复用。
                T.use_swizzle(panel_size=10, enable=enable_rasteration)
                # K 维上的所有分块都累加到同一个 C_local，因此必须先清零。
                T.clear(C_local)
                # 沿 K 维循环；多级流水线可重叠不同迭代的数据加载与计算，0 表示关闭。
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    # 下标表示全局内存中的分块起点，复制范围由共享内存分块的形状确定。
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    # B 按 (N, K) 存储，计算时转置，完成
                    # C_local += A_shared @ B_shared.T，结果形状为 (block_M, block_N)。
                    T.gemm(
                        A_shared,
                        B_shared,
                        C_local,
                        transpose_B=True,
                    )
                # 所有 K 分块计算完成后，再转换精度并写回当前线程块负责的输出区域。
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])

        return main

    # 将候选字典中的参数传给 kernel，生成内核并比较各配置的性能。
    autotuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=get_configs(M, N, K, with_roller))
        .set_compile_args(
            # -1 指定最后一个参数 C 为输出，由运行时分配并返回；调用时只需提供 A、B。
            out_idx=[-1],
            # 自动选择当前设备对应的编译目标。
            target="auto",
        )
        .set_profile_args(
            # 根据参数的形状和类型自动生成测试张量，并与 PyTorch 参考实现比较结果。
            supply_type=tl.TensorSupplyType.Auto,
            ref_prog=ref_program,
            skip_check=False,
            backend=profile_backend,
        )
    )
    # 每个候选内核预热 3 次、测量 20 次，返回包含最优配置和内核的调优结果。
    return autotuner.run(warmup=3, rep=20)


def get_heuristic_config() -> dict:
    """按 GPU 计算能力直接选择一组预设参数，省去自动搜索的开销。"""
    # 获取当前 CUDA 设备属性。
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.cuda.current_device()
    sm_major, sm_minor = torch.cuda.get_device_capability(device)
    # 例如计算能力 (8, 0) 转为 80，便于匹配下面的预设配置。
    sm_version = sm_major * 10 + sm_minor
    print(f"CUDA device capability: {sm_version}")
    # 对计算能力 8.0 和 9.0 使用不同的 K 分块、流水线级数与线程数。
    if sm_version in {80}:
        return {"block_M": 128, "block_N": 256, "block_K": 32, "num_stages": 2, "thread_num": 128, "enable_rasteration": True}
    elif sm_version in {90}:
        return {"block_M": 128, "block_N": 256, "block_K": 64, "num_stages": 3, "thread_num": 256, "enable_rasteration": True}
    else:
        # 其他计算能力使用默认配置，并关闭流水线。
        return {"block_M": 128, "block_N": 256, "block_K": 32, "num_stages": 0, "thread_num": 128, "enable_rasteration": True}


# jit 将返回的计算描述编译为可调用内核；最后一个参数 C 是自动分配的输出。
@tl.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, num_stages, thread_num, enable_rasteration, dtype=T.float16, accum_dtype=T.float32):
    """按指定参数构建矩阵乘法内核，供启发式路径直接编译和使用。

    分块计算逻辑与自动调优路径相同；这里默认使用 FP16 输入/输出和 FP32 累加。
    block_M、block_N 决定输出分块，block_K 决定每轮累加的 K 维长度；
    num_stages、thread_num、enable_rasteration 控制流水线、线程数和线程块重排。
    """

    @T.prim_func
    def gemm_autotune(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # 每个 (bx, by) 线程块计算 C 的一个 (block_M, block_N) 分块。
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
            # A、B 分块缓存在共享内存，C_local 保存 FP32 累加结果。
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            # C_shared 的数据类型与输出一致，作为寄存器到全局内存的中转缓冲区。
            C_shared = T.alloc_shared((block_M, block_N), dtype)
            # 可选的线程块重排用于改善 L2 缓存复用，累加器在进入循环前清零。
            T.use_swizzle(panel_size=10, enable=enable_rasteration)
            T.clear(C_local)
            # 每轮加载 A 的一块行数据、B 的一块行数据，累加它们的矩阵乘积。
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                # 对 B 分块转置，使 (block_M, block_K) 与 (block_K, block_N) 相乘。
                T.gemm(
                    A_shared,
                    B_shared,
                    C_local,
                    transpose_B=True,
                )
            # K 维归约结束后转换为输出类型，再写回对应的 C 分块。
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return gemm_autotune


def main(
    M: int = 4096,
    N: int = 4096,
    K: int = 4096,
    use_autotune: bool = False,
    with_roller: bool = False,
    profile_backend: str = "event",
):
    """选择内核配置，并比较 TileLang 与 PyTorch 的耗时、数值结果及计算吞吐率。"""
    if use_autotune:
        # 搜索候选配置；result.kernel 已完成编译，可直接用于后续性能测试。
        result = get_best_config(
            M,
            N,
            K,
            with_roller=with_roller,
            profile_backend=profile_backend,
        )
        print(result.config)
        kernel = result.kernel
    else:
        # 不进行搜索，按设备计算能力选择参数，再通过 jit 编译内核。
        config = get_heuristic_config()
        kernel = matmul(M, N, K, **config)

    # 自动准备测试输入；未传入参考函数时，do_bench 测量已编译的 TileLang 内核。
    profiler = kernel.get_profiler(tensor_supply_type=tl.TensorSupplyType.Auto)
    tilelang_latency = profiler.do_bench(
        backend=profile_backend,
    )
    # 传入参考函数时，测量相同矩阵形状和数据类型下的 PyTorch 实现。
    ref_latency = profiler.do_bench(
        ref_program,
        backend=profile_backend,
    )
    # 检查输出是否在绝对误差和相对误差容限内与参考结果一致。
    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)
    print(f"TileLang latency: {tilelang_latency}")
    print(f"Ref latency: {ref_latency}")
    # GEMM 约有 2*M*N*K 次浮点运算；耗时单位为毫秒，乘 1e-9 后得到 TFLOPS。
    print(f"TileLang TFlops: {2 * M * N * K / tilelang_latency * 1e-9}")
    print(f"Ref TFlops: {2 * M * N * K / ref_latency * 1e-9}")


def run_regression_perf(M: int = 4096, N: int = 4096, K: int = 4096):
    """供性能回归测试调用：使用启发式配置和 CUPTI 测量，返回内核耗时（毫秒）。"""
    config = get_heuristic_config()
    kernel = matmul(M, N, K, **config)
    profiler = kernel.get_profiler(tensor_supply_type=tl.TensorSupplyType.Auto)
    return profiler.do_bench(backend="cupti")


if __name__ == "__main__":
    # 从命令行读取矩阵维度、选参方式和测速后端，再交给 main 执行。
    # 未指定 --use_autotune 时使用启发式配置；--with_roller 仅在自动调优时生效。
    parser = argparse.ArgumentParser(description="Autotuned MatMul Benchmark")
    parser.add_argument("--m", type=int, default=4096, help="Matrix dimension M")
    parser.add_argument("--n", type=int, default=4096, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=4096, help="Matrix dimension K")
    parser.add_argument("--use_autotune", action="store_true", default=False, help="Whether to use autotune for matmul configs")
    parser.add_argument("--with_roller", action="store_true", default=False, help="Whether to enable BitBLAS roller for search space")
    parser.add_argument("--profile_backend", type=str, default="event", help="Profiler backend")
    args = parser.parse_args()
    main(
        args.m,
        args.n,
        args.k,
        args.use_autotune,
        args.with_roller,
        args.profile_backend,
    )
