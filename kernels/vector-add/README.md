### What to learn
- CUDA kernel syntax
- threads, blocks and grids
- Mapping threads to data
- Global memory access
- Memory coalescing
- Grid-stride loops
- Kernel launch overhead
- Memory bandwidth
- Basic CUDA error handling
- CUDA event timing

### Versions to implement
1. PyTorch reference
2. Triton vector addition
3. Basic CUDA vector addition
4. CUDA grid-stride loop
5. Optional vectorized version using float4

### Experiments
- Block sizes: 64, 128, 256, 512
- Small versus very large arrays
- Contiguous versus strided access
- float32 versus float16
- One element per thread versus multiple elements per thread



### Block-size experiment

- Array size: 2^24 elements
- Data type: float32
- Access: contiguous
- Repetitions: 100
- GPU: NVIDIA L4

| Block | Basic ms | Basic GB/s | Grid ms | Grid GB/s | float4 ms | float4 GB/s | Max error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.825416 | 243.909 | 0.913101 | 220.487 | 0.842228 | 239.04 | 0 |
| 128 | 0.816077 | 246.701 | 0.863293 | 233.208 | 0.840868 | 239.427 | 0 |
| 256 | 0.806287 | 249.696 | 0.850749 | 236.646 | 0.834068 | 241.379 | 0 |
| 512 | 0.802939 | 250.737 | 0.8465 | 237.834 | 0.827167 | 243.393 | 0 |

#### Block-size observations
The basic kernel was the fastest at every tested block size. Increasing the
block size from 64 to 512 produced a small improvement, from 0.825 ms to
0.803 ms. This is only about a 2.7% difference, indicating that the basic
kernel is not highly sensitive to block size in this range.

The grid-stride kernel improved more noticeably as block size increased,
although it remained approximately 5–11% slower than the basic kernel.

The float4 implementation did not outperform the basic kernel. Its results
were relatively stable across block sizes, with 512 threads producing its
best measured time.





### Array-size experiment

- Block size: 256
- Data type: float32
- Access: contiguous
- GPU: NVIDIA L4

| Array size | Basic ms | Basic GB/s | Grid ms | Grid GB/s | float4 ms | float4 GB/s | Max error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2^10 | 0.00189952 | 6.469 | 0.0020225 | 6.07566 | 0.00191283 | 6.42398 | 0 |
| 2^26 | 3.22929 | 249.376 | 3.4349 | 234.449 | 3.33952 | 241.144 | 0 |

#### Array-size observations
For 2^10 elements, all kernels completed in approximately 0.002 ms and
achieved only about 6 GB/s. At this size, fixed kernel-launch overhead
dominates execution time.

For 2^26 elements, effective bandwidth increased to 234–249 GB/s because
the execution was dominated by global-memory traffic rather than launch
overhead.

The basic kernel was fastest for both array sizes. The float4 version did
not improve performance, while the grid-stride implementation was about
6% slower for the large array.





### Memory-access experiment

- Logical array size: 2^24 elements
- Block size: 256
- Data type: float32
- Elements per thread: 1
- Strided-access distance: 4 elements
- GPU: NVIDIA L4

| Access pattern | Stride | Average ms | Effective GB/s | Max error |
|---|---:|---:|---:|---:|
| Contiguous | 1 | 0.805827 | 249.839 | 0 |
| Strided | 4 | 4.41335 | 45.6177 | 0 |

#### Memory-access observations

Changing from contiguous access to a stride of four increased execution time
from 0.806 ms to 4.413 ms, making the strided kernel approximately 5.5 times
slower.

Effective bandwidth decreased from 249.8 GB/s to 45.6 GB/s. With contiguous
access, adjacent threads access adjacent float values, allowing their requests
to be combined into coalesced memory transactions. With stride-four access,
adjacent threads access values 16 bytes apart, requiring substantially more
memory transactions and transferring more hardware data per useful element.

The reported bandwidth for the strided kernel is useful effective bandwidth;
actual hardware traffic is higher.  
"GPU performance depends not only on how much data is used, but also on whether neighboring threads access neighboring memory. Contiguous, coalesced access uses memory bandwidth far more efficiently than strided access."





### Data-type experiment

- Array size: 2^24 elements
- Block size: 256
- Access: contiguous
- Elements per thread: 1
- GPU: NVIDIA L4

| Data type | Bytes/element | Average ms | Effective GB/s | Max error |
|---|---:|---:|---:|---:|
| float32 | 4 | 0.806564 | 249.61 | 0 |
| float16 | 2 | 0.404367 | 248.94 | 0 |

#### Data-type observations

The FP16 kernel completed in 0.404 ms, approximately half the 0.807 ms
required by FP32. However, both kernels achieved nearly identical effective
bandwidth of approximately 249 GB/s.

FP16 stores each element in two bytes instead of four, so it transfers half
as much data for the same number of additions. The similar bandwidth and
approximately 2x difference in execution time confirm that vector addition
is limited primarily by memory bandwidth rather than arithmetic throughput.





### Elements-per-thread experiment

- Array size: 2^24 elements
- Block size: 256
- Data type: float32
- Access: contiguous and coalesced
- GPU: NVIDIA L4

| Elements/thread | Average ms | Effective GB/s | Max error |
|---:|---:|---:|---:|
| 1 | 0.805949 | 249.801 | 0 |
| 4 | 0.822579 | 244.755 | 0|

#### Elements-per-thread observations

Processing four scalar elements per thread did not improve performance. The
four-element implementation completed in 0.823 ms, approximately 2.1% slower
than the one-element implementation at 0.806 ms.

Although processing multiple elements per thread reduces the number of
launched threads and blocks, both kernels transfer the same amount of global
memory. Because vector addition is memory-bandwidth-bound, reducing thread
count does not reduce its primary cost. The additional indexing, loop, and
per-thread work may account for the small slowdown.

Both implementations maintained coalesced access and achieved similar
effective bandwidth, approximately 245–250 GB/s.



=> The overall story is coherent: contiguous scalar FP32 already saturates memory bandwidth well; FP16 helps by halving transferred bytes, while float4 and extra elements per thread do not improve this bandwidth-bound workload. Strided access causes by far the largest performance loss.