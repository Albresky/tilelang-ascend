# TileLang-Ascend 与 TileLang 对比分析

## 执行摘要

TileLang-Ascend 是 TileLang 针对华为昇腾 NPU 的专门适配版本。两者共享相同的核心架构和 TVM 基础设施，但在**编译器工具流**和**Python DSL lowering 流程**上存在显著差异，以适应 NPU 与 GPU 在硬件架构和执行模型上的本质区别。

---

## 一、编译器工具流对比

### 1.1 整体架构差异

#### TileLang (NVIDIA GPU)
```
Python DSL → TVM TIR → Pass Pipeline → CUDA/HIP Code → NVCC/HIPCC → Cubin/HSA Code Object
                                            ↓
                                    CUTLASS Templates (GPU-Optimized)
```

#### TileLang-Ascend (华为 NPU)
```
Python DSL → TVM TIR → Pass Pipeline → Ascend C Code → cann-cc → Ascend Binary
                                            ↓
                                    Ascend 内置算子库 (NPU-Optimized)
```

**核心区别**：
- **后端模板库**: GPU 版本使用 NVIDIA CUTLASS，NPU 版本使用华为 Ascend C Runtime
- **编译器**: GPU 使用 NVCC/HIPCC，NPU 使用 CANN 工具链的 cann-cc
- **目标格式**: GPU 生成 PTX/Cubin，NPU 生成 Ascend 专有二进制格式

---

### 1.2 编译Pass流程差异

#### 1.2.1 LowerAndLegalize 阶段

**TileLang (GPU版本) - 14个核心Pass**:
```python
def LowerAndLegalize(mod: IRModule, target: Target) -> IRModule:
    mod = tir.transform.BindTarget(target)(mod)
    
    # GPU特有优化
    if should_force_let_inline():
        mod = tilelang.transform.LetInline()(mod)
    
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)  # GPU验证优化
    mod = tilelang.transform.Simplify()(mod)
    
    # Layout配置 (GPU Fragment/Shared Memory)
    mod = tilelang.transform.LayoutReducer()(mod)
    mod = tilelang.transform.LayoutInference()(mod)
    
    # 降级Tile操作 (T.gemm → CUTLASS调用)
    mod = tilelang.transform.LowerTileOp()(mod)
    
    # GPU特有：L2缓存持久化映射
    mod = tilelang.transform.LowerL2Persistent()(mod)
    
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LoopVectorizeDynamic()(mod)
    
    return mod
```

**TileLang-Ascend (NPU版本) - 精简为9个Pass**:
```python
def LowerAndLegalize(mod: IRModule, target: Target) -> IRModule:
    mod = tir.transform.BindTarget(target)(mod)
    
    # NPU特有：宿主机数据处理
    mod = tilelang.transform.HostProcesser()(mod)
    
    # NPU前端合法化
    mod = tilelang.transform.FrontendLegalize()(mod)
    
    mod = tir.transform.Simplify()(mod)
    
    # NPU内存布局推断 (L0A/L0B/L0C/L1/UB)
    mod = tilelang.transform.LayoutInference()(mod)
    
    # 降级为NPU内置算子
    mod = tilelang.transform.LowerTileOp()(mod)
    
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tilelang.transform.LoopVectorizeDynamic()(mod)
    
    return mod
```

**关键差异**：

| Pass类型 | TileLang (GPU) | TileLang-Ascend (NPU) | 原因 |
|---------|---------------|----------------------|------|
| `LetInline` | ✅ 可选 | ❌ 无 | NPU无需此优化 |
| `InjectAssumes` | ✅ 加速验证器 | ❌ 无 | GPU符号证明需求 |
| `LayoutReducer` | ✅ Reducer布局配置 | ❌ 无 | NPU内存模型不同 |
| `LowerL2Persistent` | ✅ L2缓存映射 | ❌ 无 | NPU无L2缓存概念 |
| `HostProcesser` | ❌ 无 | ✅ NPU专有 | 提取宿主机Tiling信息 |
| `FrontendLegalize` | ❌ 无 | ✅ NPU专有 | 处理Swizzle等前端IR |

---

#### 1.2.2 OptimizeForTarget 阶段

**TileLang (GPU版本) - 高度复杂**:
```python
def OptimizeForTarget(mod: IRModule, target: Target) -> IRModule:
    # 基础优化
    mod = tir.transform.LowerOpaqueBlock()(mod)
    mod = tir.transform.NarrowDataType(32)(mod)
    
    # GPU特有：Warp专用化 (Hopper架构)
    if allow_warp_specialized(target=target):
        mod = tilelang.transform.WarpSpecialized()(mod)
    
    # GPU特有：软件流水线注入
    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
    
    # GPU特有：PTX异步拷贝
    if allow_tma_and_warp_specialized(target=target):
        mod = tilelang.transform.InjectPTXAsyncCopy()(mod)
        mod = tilelang.transform.InjectTMABarrier()(mod)
    
    # GPU特有：共享内存合并
    if should_enable_aggressive_merge(target=target):
        mod = tilelang.transform.MergeSharedMemoryAllocations()(mod)
    
    # GPU特有：Hopper指令降级
    if is_hopper(target):
        mod = tilelang.transform.LowerHopperIntrin()(mod)
    
    # 向量化和存储优化
    mod = tilelang.transform.VectorizeLoop()(mod)
    mod = tir.transform.StorageRewrite()(mod)
    mod = tir.transform.UnrollLoop()(mod)
    
    # 更多GPU优化 (Thread同步、Fence Proxy等)
    # ... (共约20个Pass)
    
    return mod
```

**TileLang-Ascend (NPU版本) - 精简流程**:
```python
def OptimizeForTarget(mod: IRModule, target: Target) -> IRModule:
    # 基础优化
    mod = tir.transform.LowerOpaqueBlock()(mod)
    mod = tir.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tir.transform.Simplify()(mod)
    
    # 向量化和存储优化
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize())(mod)
    mod = tir.transform.StorageRewrite()(mod)
    mod = tir.transform.UnrollLoop()(mod)
    mod = tir.transform.RenormalizeSplitPattern()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tir.transform.RemoveNoOp()(mod)
    mod = tir.transform.RewriteUnsafeSelect()(mod)
    mod = tir.transform.HoistIfThenElse()(mod)
    
    return mod
```

**关键差异**：

| 优化Pass | TileLang (GPU) | TileLang-Ascend (NPU) | 原因 |
|---------|---------------|----------------------|------|
| `WarpSpecialized` | ✅ Hopper架构 | ❌ 无 | NPU无Warp概念 |
| `InjectSoftwarePipeline` | ✅ 流水线优化 | ❌ 无 | NPU执行模型不同 |
| `InjectPTXAsyncCopy` | ✅ TMA异步拷贝 | ❌ 无 | NPU无TMA硬件 |
| `InjectTMABarrier` | ✅ TMA同步 | ❌ 无 | 同上 |
| `MergeSharedMemoryAllocations` | ✅ 共享内存合并 | ❌ 无 | NPU内存管理不同 |
| `LowerHopperIntrin` | ✅ WGMMA等指令 | ❌ 无 | NPU使用Cube/Vector Core |
| Pass总数 | ~20个 | ~13个 | GPU需更多硬件特定优化 |

---

### 1.3 代码生成差异

#### 1.3.1 目标代码生成器

**TileLang (GPU)**:
- **文件**: `src/target/codegen_cuda.cc` (继承TVM CodeGenCUDA)
- **生成语言**: CUDA C++ / HIP
- **关键特性**:
  - 直接生成 `__global__` kernel
  - 使用 CUTLASS 模板 (`tl::gemm_ss<...>()`)
  - 支持 `__shared__` 动态内存分配
  - 生成 PTX 内联汇编 (TMA, WGMMA)

**示例生成代码**:
```cpp
#include <tl_templates/cuda/gemm.h>

extern "C" __global__ void gemm_kernel(
    half_t* __restrict__ A,
    half_t* __restrict__ B,
    half_t* __restrict__ C
) {
    extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
    half_t* A_shared = (half_t*)(buf_dyn_shmem + 0);
    half_t* B_shared = (half_t*)(buf_dyn_shmem + 8192);
    
    // CUTLASS GEMM模板调用
    tl::gemm_ss<128, 128, 32, 2, 2, ...>(
        A_shared, B_shared, C_local
    );
}
```

**TileLang-Ascend (NPU)**:
- **文件**: `src/target/codegen_ascend.cc` (全新实现)
- **生成语言**: Ascend C (基于C++扩展)
- **关键特性**:
  - 生成 `CATLASS_GLOBAL` 宏标注的函数
  - 使用 Ascend C 内置算子 (`AscendC::Gemm()`)
  - 显式管理 L0A/L0B/L0C/L1/UB 缓冲区
  - 生成跨核心同步代码 (Cube ↔ Vector Core)

**示例生成代码**:
```cpp
#include "tl_templates/ascend/common.h"
#include "acl/acl.h"

using namespace Catlass;

extern "C" CATLASS_GLOBAL void gemm_kernel(
    GM_ADDR half_t* A,
    GM_ADDR half_t* B,
    GM_ADDR half_t* C
) {
    // NPU特有：显式L1/L0缓冲区声明
    __local__ half_t ascend_l1[16384];
    __local__ half_t ascend_l0a[4096];
    __local__ half_t ascend_l0b[4096];
    __local__ half_t ascend_l0c[16384];
    
    // NPU内置GEMM算子
    AscendC::Gemm(
        ascend_l0a, ascend_l0b, ascend_l0c,
        128, 256, 64, init_flag
    );
}
```

#### 1.3.2 编译命令差异

**TileLang (GPU)**:
```python
# tilelang/engine/lower.py
@tvm.register_func("tilelang_callback_cuda_compile")
def tilelang_callback_cuda_compile(code, target):
    cutlass_path = get_cutlass_include_path()
    compute_version = get_target_compute_version(target)
    
    arch = [f"-arch=sm_{compute_version}"]
    format = "cubin"
    
    ptx = nvcc.compile_cuda(
        code, format, arch,
        options=[
            "-std=c++17",
            "--use_fast_math",
            "-I" + tl_template_path,
            "-I" + cutlass_path,
        ]
    )
    return ptx
```

**TileLang-Ascend (NPU)**:
```python
# 推测：需调用CANN工具链编译器
@tvm.register_func("tilelang_callback_ascend_compile")
def tilelang_callback_ascend_compile(code, target):
    ascend_toolkit_path = get_ascend_toolkit_path()
    
    # 使用cann-cc编译Ascend C代码
    binary = cann_cc.compile_ascend(
        code,
        options=[
            "-std=c++17",
            "-I" + tl_template_path,
            "-I" + ascend_toolkit_path,
        ]
    )
    return binary
```

---

### 1.4 运行时执行差异

#### TileLang (GPU)
- **Backend适配器**:
  - `CythonKernelAdapter`: 编译为 `.so` 动态库
  - `NVRTCKernelAdapter`: 运行时编译 (NVRTC API)
- **执行流程**:
  1. 加载 `.cubin` 或 NVRTC 编译的 kernel
  2. 通过 CUDA Driver API 启动 kernel
  3. 使用 PyTorch DLPack 进行零拷贝数据交换
- **线程模型**: `<<<grid, block>>>` 显式并行

#### TileLang-Ascend (NPU)
- **Backend适配器**:
  - 基于 ACL (Ascend Computing Language) Runtime
  - 可能使用 `torch_npu` 扩展
- **执行流程**:
  1. 加载编译好的 Ascend Binary
  2. 通过 ACL API (`aclrtLaunchKernel`) 启动
  3. 使用 `torch_npu` 的 DLPack 接口
- **线程模型**: 
  - 使用 `T.Kernel(num_cores, is_npu=True)`
  - 显式指定核心数 (core_id)
  - 通过 `T.Scope("C")` 区分 Cube Core / Vector Core

---

## 二、Python DSL Lowering 流程对比

### 2.1 内存层次映射

#### 2.1.1 GPU内存层次 (TileLang)

```python
# GPU三级内存模型
T.alloc_local(shape, dtype)         # → register (寄存器)
T.alloc_shared(shape, dtype)        # → shared memory (共享内存)
T.alloc_fragment(shape, dtype)      # → wmma.matrix_a/b/accumulator (Tensor Core寄存器)
```

**映射到CUDA代码**:
```cpp
float4 local_buf[256];                      // local → register
__shared__ half_t shared_buf[16384];        // shared → __shared__
wmma::fragment<...> frag_a;                 // fragment → wmma寄存器
```

#### 2.1.2 NPU内存层次 (TileLang-Ascend)

```python
# NPU五级内存模型 (新增)
T.alloc_L0A(shape, dtype)           # → L0A Buffer (Cube Core A矩阵)
T.alloc_L0B(shape, dtype)           # → L0B Buffer (Cube Core B矩阵)
T.alloc_L0C(shape, dtype)           # → L0C Buffer (Cube Core 累加器)
T.alloc_L1(shape, dtype)            # → L1 Buffer (片上缓存)
T.alloc_ub(shape, dtype)            # → Unified Buffer (Vector Core缓存)
```

**对应关系**:
| TVM Scope | Ascend硬件 | 用途 |
|-----------|-----------|------|
| `wmma.matrix_a` | L0A | Cube Core输入矩阵A |
| `wmma.matrix_b` | L0B | Cube Core输入矩阵B |
| `wmma.accumulator` | L0C | Cube Core累加结果 |
| `shared.dyn` | L1 | 片上高速缓存 |
| `shared` | UB (Unified Buffer) | Vector Core统一缓冲区 |

**映射到Ascend C代码**:
```cpp
__local__ half_t ascend_l0a[4096];    // L0A → Cube Core矩阵A寄存器
__local__ half_t ascend_l0b[4096];    // L0B → Cube Core矩阵B寄存器
__local__ half_t ascend_l0c[16384];   // L0C → Cube Core累加器
__local__ half_t ascend_l1[65536];    // L1  → 片上L1缓存
__local__ half_t ascend_ub[32768];    // UB  → Vector Core统一缓冲区
```

---

### 2.2 Kernel Launch模型差异

#### 2.2.1 GPU执行模型 (TileLang)

```python
@T.prim_func
def gemm_kernel(A, B, C):
    # GPU: 2D Grid + 1D Block
    with T.Kernel(
        T.ceildiv(N, block_N),  # grid.x
        T.ceildiv(M, block_M),  # grid.y
        threads=128             # blockDim.x
    ) as (bx, by):
        # 自动映射到CUDA线程模型
        # bx = blockIdx.x
        # by = blockIdx.y
        # threadIdx.x ∈ [0, 128)
        
        # 线程级并行
        A_shared = T.alloc_shared((block_M, block_K), "float16")
        T.copy(A[by * block_M, :], A_shared)  # 128个线程并行拷贝
```

**降级后的CUDA代码**:
```cpp
__global__ void gemm_kernel(...) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;  // 自动生成
    
    __shared__ half_t A_shared[16384];
    
    // 自动生成的并行拷贝代码
    for (int i = tid; i < 16384; i += 128) {
        A_shared[i] = A[by * block_M * K + i];
    }
    __syncthreads();
}
```

#### 2.2.2 NPU执行模型 (TileLang-Ascend)

```python
@T.prim_func
def gemm_kernel(A, B, C):
    # NPU: 核心级并行 (无线程抽象)
    m_num = M // block_M
    n_num = N // block_N
    
    with T.Kernel(
        m_num * n_num,  # 总核心数
        is_npu=True     # NPU标志
    ) as (cid, _):      # cid = 核心ID
        # 手动计算核心坐标
        bx = cid // n_num
        by = cid % n_num
        
        # 显式指定执行作用域
        with T.Scope("C"):  # 在Cube Core上执行
            A_L1 = T.alloc_L1((block_M, K_L1), "float16")
            
            # 必须手动同步
            T.copy(A[bx * block_M, :], A_L1)
            T.barrier_all()  # 显式同步所有核心
```

**关键差异**：

| 特性 | GPU (TileLang) | NPU (TileLang-Ascend) |
|------|---------------|----------------------|
| 线程模型 | CUDA线程网格 (Grid/Block) | 核心级并行 (Core ID) |
| 并行抽象 | `threadIdx.x` 自动生成 | 无线程抽象，显式核心管理 |
| 同步机制 | `__syncthreads()` 自动插入 | `T.barrier_all()` 必须显式调用 |
| 作用域指定 | 自动推断 | `T.Scope("C")` / `T.Scope("V")` 显式指定 |
| 数据拷贝并行化 | 自动线程级并行 | 需手动向量化 (`T.add` 等) |

---

### 2.3 计算原语差异

#### 2.3.1 GPU计算原语 (TileLang)

```python
# 高级GEMM原语 (自动映射到CUTLASS)
T.gemm(A_shared, B_shared, C_local)

# 自动向量化的拷贝
T.copy(A[i, j], A_shared)  # 128线程并行拷贝

# 自动Reduce
T.clear(C_local)  # 清零累加器

# 软件流水线
for k in T.Pipelined(num_stages=3):
    T.copy(...)  # 自动插入cp_async
    T.gemm(...)  # 自动管道调度
```

**降级到TIR**:
```python
# T.gemm() → tir.call_extern("tl::gemm_ss<128, 128, 32, ...>")
tir.call_extern(
    "void",
    "tl::gemm_ss<128, 128, 32, 2, 2, 0, 0, 0, 32, 128, 0, 0>",
    A_shared.data, B_shared.data, C_local.data
)
```

#### 2.3.2 NPU计算原语 (TileLang-Ascend)

```python
# NPU专用GEMM原语 (版本化)
T.gemm_v0(A_L1, B_L1, C_L0, init=True)  # 初始化版本
T.gemm_v0(A_L1, B_L1, C_L0)             # 累加版本

# 手动向量化拷贝
T.copy(A[i, j], A_L1)  # 单核心拷贝，需手动优化

# 显式跨核心同步
T.barrier_all()  # 同步所有核心

# 跨核心通信 (Cube ↔ Vector)
T.set_cross_flag("mte1", "mte2", 0)  # 设置标志
T.wait_cross_flag("mte1", "mte2", 0) # 等待标志
```

**降级到TIR**:
```python
# T.gemm_v0() → tir.call_extern("AscendC::Gemm")
tir.call_extern(
    "void",
    "gemm_v0",
    A_L1.data, B_L1.data, C_L0.data,
    M, N, K, init_flag
)
```

**关键差异**:

| 原语类型 | GPU | NPU | 差异说明 |
|---------|-----|-----|---------|
| GEMM | `T.gemm()` | `T.gemm_v0()` | NPU需显式指定初始化 |
| 拷贝 | 自动并行 | 单核心 | NPU无自动线程并行 |
| Reduce | `T.clear()` 自动 | 手动管理 | NPU需显式清零 |
| 流水线 | `T.Pipelined()` | 手动编排 | NPU需显式flag同步 |
| 同步 | 自动插入 | `T.barrier_all()` | NPU必须手动调用 |

---

### 2.4 完整示例对比

#### 2.4.1 GPU版本 GEMM (TileLang)

```python
@tilelang.jit(out_idx=[-1])
def matmul_gpu(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def gemm(A: T.Tensor((M, K), "float16"),
             B: T.Tensor((K, N), "float16"),
             C: T.Tensor((M, N), "float16")):
        
        # 2D Grid启动
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=128
        ) as (bx, by):
            
            # 共享内存分配
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            
            # 清零累加器
            T.clear(C_local)
            
            # 软件流水线
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            
            # 写回
            T.copy(C_local, C[by * block_M, bx * block_N])
    
    return gemm
```

**特点**:
- ✅ 简洁的API，无需关心底层细节
- ✅ 自动线程并行化
- ✅ 自动流水线调度
- ✅ 自动同步管理

#### 2.4.2 NPU版本 GEMM (TileLang-Ascend)

```python
@tilelang.jit(out_idx=[-1])
def matmul_npu(M, N, K, block_M, block_N, K_L1):
    m_num = M // block_M
    n_num = N // block_N
    
    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"),
             B: T.Tensor((K, N), "float16"),
             C: T.Tensor((M, N), "float16")):
        
        # 核心级并行启动
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            # 手动计算核心坐标
            bx = cid // n_num
            by = cid % n_num
            
            # NPU特有内存分配
            A_L1 = T.alloc_L1((block_M, K_L1), "float16")
            B_L1 = T.alloc_L1((K_L1, block_N), "float16")
            C_L0 = T.alloc_L0C((block_M, block_N), "float32")
            
            # 在Cube Core上执行
            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                
                for k in T.serial(loop_k):
                    # 显式数据拷贝
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)
                    
                    # 显式同步所有核心
                    T.barrier_all()
                    
                    # NPU GEMM原语
                    if k == 0:
                        T.gemm_v0(A_L1, B_L1, C_L0, init=True)
                    else:
                        T.gemm_v0(A_L1, B_L1, C_L0)
                    
                    # 再次同步
                    T.barrier_all()
                
                # 写回全局内存
                T.copy(C_L0, C[bx * block_M, by * block_N])
    
    return main
```

**特点**:
- ❌ 需要手动管理核心坐标
- ❌ 显式同步调用
- ❌ 显式指定执行作用域
- ❌ 手动管理内存层次
- ✅ 更贴近NPU硬件模型

---

### 2.5 Lowering Pass实现差异

#### 2.5.1 Layout Inference差异

**GPU版本**:
```cpp
// src/transform/layout_inference.cc
class LayoutInferencer {
  void InferFragmentLayout(Buffer buf) {
    // 推断WMMA Fragment布局
    if (buf.scope == "wmma.matrix_a") {
      // 使用CUTLASS Layout (row_major/col_major)
      buf.layout = make_cutlass_layout(buf.shape);
    }
  }
};
```

**NPU版本**:
```cpp
// src/transform/layout_inference.cc (修改版)
class LayoutInferencer {
  void InferFragmentLayout(Buffer buf) {
    // 推断NPU L0缓冲区布局
    if (buf.scope == "wmma.matrix_a") {  // 对应L0A
      // 使用Ascend C Layout (ZN格式等)
      buf.layout = make_ascend_layout(buf.shape);
    }
  }
};
```

#### 2.5.2 Lower Tile Op差异

**GPU版本**:
```cpp
// src/transform/lower_tile_op.cc
void LowerGemm(Call op) {
  // T.gemm() → tl::gemm_ss<...>()
  String template_sig = "tl::gemm_ss<" + 
    std::to_string(M) + "," +
    std::to_string(N) + "," +
    std::to_string(K) + "," +
    "...>";
  
  return call_extern("void", template_sig, A, B, C);
}
```

**NPU版本**:
```cpp
// src/transform/lower_tile_op.cc (修改版)
void LowerGemm(Call op) {
  // T.gemm_v0() → AscendC::Gemm()
  bool init_flag = op.attrs["init"];
  
  return call_extern(
    "void", "gemm_v0",
    A, B, C, M, N, K, init_flag
  );
}
```

#### 2.5.3 NPU专有Pass

**HostProcesser Pass** (NPU独有):
```cpp
// src/transform/ascend_host.cc
class HostProcesser : public IRMutator {
  // 提取宿主机Tiling信息
  Map<Var, PrimExpr> tiling_map_;
  
  Stmt VisitStmt_(const LetStmtNode *op) {
    if (isNeedTiling(op->var, op->value)) {
      // 将Tiling参数存储到attrs
      tiling_map_.Set(op->var, op->value);
      return VisitStmt(op->body);  // 移除Let语句
    }
    return IRMutator::VisitStmt_(op);
  }
};
```

**作用**:
- NPU需要在Host端计算Tiling参数
- 将 `block_M`, `block_N` 等存储到 `func.attrs["tiling_map"]`
- GPU版本无需此Pass，因为可以在Device端计算

---

## 三、核心技术差异总结

### 3.1 硬件架构差异

| 维度 | NVIDIA GPU | 华为 NPU |
|------|-----------|---------|
| **执行单元** | SM (Streaming Multiprocessor) | Cube Core + Vector Core |
| **并行模型** | SIMT (单指令多线程) | SIMD (单指令多数据) |
| **内存层次** | Register / Shared / L2 / Global | L0A/B/C / L1 / UB / L2 / Global |
| **线程抽象** | Thread / Warp / Block | 无 (核心级并行) |
| **同步原语** | `__syncthreads()` / `__syncwarp()` | 跨核心Flag同步 |
| **专用计算单元** | Tensor Core (WMMA/MMA) | Cube Core (矩阵计算) |

### 3.2 编译器架构差异

| 组件 | TileLang (GPU) | TileLang-Ascend (NPU) |
|------|---------------|----------------------|
| **Pass数量** | ~34 个 | ~22 个 (精简) |
| **模板库** | CUTLASS | Ascend C Runtime |
| **后端编译器** | NVCC / HIPCC | cann-cc |
| **中间格式** | PTX | Ascend IR |
| **优化重点** | Warp专用化, TMA, 流水线 | 跨核心通信, 内存对齐 |

### 3.3 DSL语义差异

| 概念 | GPU语义 | NPU语义 |
|------|--------|--------|
| `T.Kernel(...)` | Grid/Block启动 | 核心数量 |
| `T.alloc_shared()` | 共享内存 | L1缓存 |
| `T.alloc_fragment()` | Tensor Core寄存器 | L0C累加器 |
| `T.gemm()` | CUTLASS模板 | Ascend内置算子 |
| `T.copy()` | 自动并行拷贝 | 单核心拷贝 |
| `T.Pipelined()` | 自动流水线 | 手动Flag同步 |
| `T.Scope()` | 不存在 | Cube/Vector Core选择 |
| `T.barrier_all()` | 自动插入 | 必须显式调用 |

---

## 四、迁移指南

### 4.1 从TileLang迁移到TileLang-Ascend

#### 步骤1: 修改Kernel启动
```python
# GPU版本
with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
    pass

# NPU版本
with T.Kernel(num_cores, is_npu=True) as (cid, _):
    bx = cid // n_num
    by = cid % n_num
```

#### 步骤2: 替换内存分配
```python
# GPU版本
A_shared = T.alloc_shared((M, K), "float16")
C_local = T.alloc_fragment((M, N), "float32")

# NPU版本
A_L1 = T.alloc_L1((M, K), "float16")
C_L0 = T.alloc_L0C((M, N), "float32")
```

#### 步骤3: 添加显式同步
```python
# GPU版本 (自动同步)
T.copy(A[...], A_shared)
T.gemm(A_shared, B_shared, C_local)

# NPU版本 (显式同步)
T.copy(A[...], A_L1)
T.barrier_all()  # 必须添加
T.gemm_v0(A_L1, B_L1, C_L0, init=True)
T.barrier_all()  # 必须添加
```

#### 步骤4: 添加执行作用域
```python
# NPU版本需要
with T.Scope("C"):  # Cube Core
    T.gemm_v0(...)

with T.Scope("V"):  # Vector Core
    T.add(...)
```

---

## 五、性能优化差异

### 5.1 GPU优化技术

1. **Warp Specialization** (Hopper)
   - Producer/Consumer Warp分离
   - TMA异步加载

2. **Software Pipelining**
   - `cp_async` 多级流水
   - 隐藏内存延迟

3. **Shared Memory Merge**
   - 生命周期分析
   - 减少Shared Memory使用

4. **Tensor Core优化**
   - WMMA/MMA指令映射
   - Fragment Layout优化

### 5.2 NPU优化技术

1. **跨核心流水线**
   - 手动Flag同步
   - Cube/Vector Core并行

2. **内存对齐**
   - L0/L1 Buffer对齐要求
   - ZN Layout优化

3. **Swizzle优化**
   - 减少L2 Bank冲突
   - 提高内存带宽利用

4. **Tile尺寸调优**
   - 受L0/L1容量限制
   - 需要考虑核心数量

---

## 六、结论

TileLang-Ascend 在保持 TileLang 核心架构的同时，针对 NPU 特性进行了深度适配：

### 相同点
✅ 共享TVM IR基础设施  
✅ 使用相同的Pass框架  
✅ 保持Python DSL风格一致  

### 不同点
❌ 编译Pass流程精简 (34→22个)  
❌ 内存层次映射完全不同  
❌ 执行模型从线程级到核心级  
❌ 需要显式同步和作用域管理  
❌ 后端生成Ascend C代码  

### 设计哲学差异
- **GPU版本**: 高度抽象，自动化，隐藏硬件细节
- **NPU版本**: 暴露硬件模型，显式控制，接近底层

这种设计反映了NPU与GPU在硬件架构上的本质差异，是对异构计算编译器设计的重要探索。

