## GEMM

### cutlass::gemm::device::SparseGemm

A `class` that performs SpMM under the 50% sparsity on Ampere defined in `cutlass/include/cutlass/gemm/device/gemm_sparse.h`.

**Template Arguments**
* `ElementA_(typename)`: element type of A matrix operand
* `LayoutA_(typename)`: layout type of A matrix operand
* `ElementB_(typename)`: element type of B matrix operand
* `LayoutB_(typename)`: Layout type of B matrix operand
* `ElementC_(typename)`: Element type of C & D matrix operand
* `LayoutC_(typename)`: Layout type of C matrix operand
* `ElementAccumulator_(typename)`: Element type for internal accumulation
  * Default: `ElementC_`.
* `OperatorClass_(typename)`: Simt or Tensor Core
* `ArchTag_(typename)`: architecture type
  * `arch::Sm70`, `arch::Sm80`
* `ThreadblockShape_(typename)`: threadblock-level tile size (`GemmShape`)
* `WarpShape_(typename)`: Warp-level tile size (`GemmShape`)
* `InstructionShape_(typename)`: ??????
* `EpilogueOutputOp_(typename)`: Epilogue output operator
* `ThreadblockSwizzle_(typename)`: See `cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>` in Tiling.
* `Stages(int)`: Number of stages used in the pipelined mainloop
* `AlignmentA(int)`: Access granularity of A matrix in units of elements
* `AlignmentB(int)`: Access granularity of B matrix in units of elements
* `SplitKSerial(bool)`: if true, kernel supports split-K with serial reduction
* `Operator_(typename)`: Operation performed by GEMM
  

**Members**
* `GemmKernel (kernel::DefaultSparseGemm)`: The gemm kernel will be launched
  * See TODO ???????
* `Arguments (struct)`: A `struct` that contains all the input arguments of the GEMM kernel
  * `Constructor`:
    * Input:
        * `problem_size_(GemmCoord)`
        * `ref_A_(TensorRef<ElementA const, LayoutA>)`
        * `ref_B_(TensorRef<ElementB const, LayoutB>)`
        * `ref_C_(TensorRef<ElementC const, LayoutC>)`
        * `ref_D_(TensorRef<ElementC, LayoutC>)`
        * `ref_E_(TensorRef<ElementE, LayoutE>)`:
        * `epilogue_`
        * `split_k_slices(int)`
* `can_implement()`: A function that verify the problem is supported by the kernel 
  * Input: `args(Arguments)`: an instance of the above `struct` `Arguments`.
  * Output: `Status`: support or not
* `get_workspace_size()`: If split-K is applied, additional space is required to store the partial sums. Otherwise, this just returns 0.
* `initialize()`: initialize the kernels. This function allocates memory space for the workspace, and set the kernel's attribute to enable larger shared memory (with `cudaFuncAttributeMaxDynamicSharedMemorySize`). 
  * Input:
    * `args(Arguments)`: the kernel's arguments
    * `workspace`: additional space for the kernel
    * `stream`: CUDA stream.
* `run()`: Run the kernel. It first get the grid size and block size, launch the kernel, and report success or not.
  * Input: stream.
* `()`: launch the `run()` function above.

**Example**
```c++
// ElementA_
using ElementInputA = cutlass::half_t;
// LayoutA_
using LayoutInputA = cutlass::layout::RowMajor;
// ElementB_
using ElementInputB = cutlass::half_t;
// LayoutB_
using LayoutInputB = cutlass::layout::RowMajor;
// ElementC_
using ElementOutput = float;
// LayoutC_
using LayoutOutput = cutlass::layout::RowMajor;
// ElementAccumulator_
using ElementAccumulator = float;
// OperatorClass_
using MMAOp = cutlass::arch::OpClassTensorOp;
// ArchTag_
using SmArch = cutlass::arch::Sm80;
// ThreadblockShape_
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 64>;  // <- threadblock tile M = 128, N = 128, K = 256
// WarpShape_
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>;  // <- warp tile M = 64, N = 64, K = 256
// InstructionShape_
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 32>;  // <- MMA Op tile M = 16, N = 8, K = 128

// Epilogue output operator
using ElementComputeEpilogue = ElementAccumulator; 
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementComputeEpilogue>;
// ThreadblockSwizzle_
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
// Stages
constexpr int NumStages = 3;

using Gemm = cutlass::gemm::device::SparseGemm<ElementInputA,
                                               LayoutInputA,
                                               ElementInputB,
                                               LayoutInputB,
                                               ElementOutput,
                                               LayoutOutput,
                                               ElementAccumulator,
                                               MMAOp,
                                               SmArch,
                                               ShapeMMAThreadBlock,
                                               ShapeMMAWarp,
                                               ShapeMMAOp,
                                               EpilogueOp,
                                               SwizzleThreadBlock,
                                               NumStages>;
```
***
### cutlass::gemm::kernel::DefaultSparseGemm

The wrapper of the GPU device kernel that perform 50% sparsity SpMM. It constructs the `Mma` and `Epilogue` types and define the kernel-level GEMM operator under `cutlass::gemm::kernel::SparseGemm` type. Besides, the `Mma` is under type `cutlass::gemm::threadblock::DefaultSparseMma`. It is defined in `cutlass/include/cutlass/gemm/kernel/default_gemm_sparse.h`.
**Template Arguments**
* `ElementA_(typename)`: Element type for A matrix operand
* `LayoutA_(typename)`: Layout type for A matrix operand
* `kAlignmentA(int)`: Access granularity of A matrix in units of elements
* `ElementB_(typename)`: Element type for B matrix operand
* `LayoutB_(typename)`: Layout type for B matrix operand
* `kAlignmentB(int)`: Access granularity of B matrix in units of elements
* `ElementC_(typename)`: Element type for C and D matrix operands
* `LayoutC_(typename)`: Layout type for C and D matrix operands
* `ElementAccumulator(typename)`: Element type for internal accumulation
* `OperatorClass(typename)`: Operator class tag
* `ArchTag(typename)`: Tag indicating architecture to tune for
* `ThreadblockShape(typename)`: Threadblock-level tile size (concept: GemmShape)
* `WarpShape(typename)`: Warp-level tile size (concept: GemmShape)
* `InstructionShape(typename)`: Warp-level tile size (concept: GemmShape)
* `EpilogueOutputOp(typename)`: Epilogue output operator
* `ThreadblockSwizzle(typename)`: Threadblock-level swizzling operator
* `Stages(int)`: Number of stages used in the pipelined mainloop
* `SplitKSerial(bool)`: If true, kernel is configured to support serial reduction in the epilogue
* `Operator(typename)`: Operation performed by GEMM

***
### cutlass::gemm::kernel::SparseGemm
This `struct` assembles all the block-level APIs to build the SpMM kernel under 50% sparsity. It is defined under `cutlass/include/cutlass/gemm/kernel/sparse_gemm.h`.

**Template Args**
* `Mma_(typename)`: Threadblock-scoped matrix multiply-accumulate 
* `Epilogue_(typename)`: Epilogue_
* `ThreadblockSwizzle_(typename)`: Threadblock swizzling function
* `SplitKSerial(bool)`: If true, code supporting split-K via serial reduction is enabled.

**Members**
* `can_implement()`: check whether the problem size matches the kernel
* `operator()`: Executes one GEMM.