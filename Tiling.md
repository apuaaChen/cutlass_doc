## Tiling

### cutlass::gemm::GemmShape<M, N, K>
A simple `struct` that defines a (M x N x K) tile size defined in `cutlass/include/cutlass/gemm/gemm.h`

**Args**
* `M(int)`: Rows of matrix product
* `N(int)`: COlumns of matrix product
* `K(int)`: Inner dimension of matrix product

**Example**
```c++
// Defines a Thread Block Tiling size (128 * 64) x (64 * 128)
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 64>;
```

**Members**
* `toCoord()`
  * Input: N/A
  * Output: A `Coord<3>` object that contains `M`, `N`, `K`.

***

### cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>
A `struct` that helps computing the offset of a thread block tile.

The problem size is `(problem_size.m(), problem_size.n(), problem_size.k())`. The thread block tile size is `(tile_size.m(), tile_size.n())`. Normally, each thread block will sweep through the K dimension to generate the final result. But there is still an option to split K to `split_k_slices` parts assigned to different thread blocks.

**Members**
* `kTile (int)`: ?????
* `get_tiled_shape()`:
  * Input： 
    * `problem_size (GemmCoord)`: problem size of the GEMM workload 
    * `tile_size (GemmCoord)`: tile size of each thread block
    * `split_k_slices (int)`: the K is partitioned to `split_k_slices` parts processed in parallel.
  * Output: A `(GemmCoord)` object contains the number of thread blocks along M, N, and K dimensions.
  * Example: Given problem size `(127, 128, 64)` and tile size `(32, 32)`, `split_k_slices=2`, the output is `(4, 4, 2)`.
* `get_grid_shape()`:
  * Input: `tiled_shape (GemmCoord)`: ???
  * Output: A `dim3` object of the thread block tile size.
* `get_tile_offset()`:
  * Input: `tiled_shape (GemmCoord)`
  * Output: A `GemmCoord` object contains `(blockIdx.x, blockIdx.y, blockIdx.z)`.
> Notably, the default kTile is 1. When it is other values, the functionality of the member functions will be affected.