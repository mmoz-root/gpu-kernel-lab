### What to learn
- Parallel reductions
- Shared memory
- Thread synchronization
- Warp-level execution
- Warp shuffle instructions
- Branch divergence
- Bank conflicts
- Multi-stage kernels
- Associativity and floating-point error

### Versions to implement
1. CPU/PyTorch reference
2. Naive CUDA reduction
3. Interleaved shared-memory reduction
4. Sequential-addressing reduction
5. Warp-shuffle reduction
6. Triton reduction

### Experiments
- Different block sizes
- Shared memory versus warp shuffle
- Sum versus maximum
- Power-of-two versus irregular input sizes
- float32 accumulation versus lower precision



### Block-size experiment

- Input size: 2^24 elements
- Input and accumulation type: float32
- Reductions: sum and maximum
- Block sizes: 64, 128, 256, and 512
- Repetitions: 100
- GPU: NVIDIA L4

#### Sum reduction

| Block | Naive ms | Naive GB/s | Interleaved ms | Interleaved GB/s | Sequential ms | Sequential GB/s | Warp ms | Warp GB/s | Largest absolute error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64  | 420.149 | 0.159726 | 0.271145 | 247.502 | 0.270428 | 248.158 | 0.270264 | 248.308 | 0.0266547 |
| 128 | 420.246 | 0.159689 | 0.258570 | 259.538 | 0.257433 | 260.685 | 0.257393 | 260.726 | 0.0266547 |
| 256 | 420.115 | 0.159739 | 0.270316 | 248.261 | 0.269169 | 249.319 | 0.268708 | 249.747 | 0.0266547 |
| 512 | 425.714 | 0.157638 | 0.271022 | 247.614 | 0.268646 | 249.804 | 0.267776 | 250.616 | 0.0266547 |
#### Maximum reduction

| Block | Naive ms | Naive GB/s | Interleaved ms | Interleaved GB/s | Sequential ms | Sequential GB/s | Warp ms | Warp GB/s | Largest absolute error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 420.133 | 0.159733 | 0.271533 | 247.148 | 0.270377 | 248.205 | 0.270223 | 248.346 | 0 |
| 128 | 420.220 | 0.159699 | 0.258765 | 259.343 | 0.257514 | 260.603 | 0.257108 | 261.014 | 0 |
| 256 | 420.115 | 0.159739 | 0.270469 | 248.120 | 0.269281 | 249.215 | 0.268626 | 249.823 | 0 |
| 512 | 425.782 | 0.157613 | 0.271094 | 247.549 | 0.268749 | 249.709 | 0.267858 | 250.539 | 0 |

### Observations
The 128-thread configuration produced the best performance for both
reductions. Warp-shuffle sum reached 260.726 GB/s, while warp-shuffle
maximum reached 261.014 GB/s.
Sequential addressing was consistently slightly faster than interleaved
addressing, while warp shuffle was the fastest implementation. The
differences between the parallel implementations were small because
reading the input array dominated the execution time.
The parallel implementations were approximately 1,630 times faster than
the naive single-thread implementation. Maximum reduction matched the CPU
reference exactly. Sum reduction produced small errors because changing
the block size and reduction strategy changed the floating-point addition
order.



### shmem vs. warp shuffle

- Input size: 2^24 elements
- Input and accumulation type: float32
- Block size: 128 threads
- Repetitions: 100
- Shared-memory implementation: sequential addressing
- Warp implementation: warp shuffle with shared warp totals
- GPU: NVIDIA L4

| Implementation | Sum ms | Sum GB/s | Sum absolute error | Maximum ms | Maximum GB/s | Maximum absolute error |
|---|---:|---:|---:|---:|---:|---:|
| Sequential shared memory | 0.257433 | 260.685 | 0.000104439 | 0.257514 | 260.603 | 0 |
| Warp shuffle | 0.257393 | 260.726 | 0.000104439 | 0.257108 | 261.014 | 0 |

#### Observations
Warp shuffle was slightly faster than the sequential shared-memory
implementation, but the difference was very small. Sum performance was
effectively identical, while warp-shuffle maximum was approximately 0.16%
faster.
The input is much larger than the partial reduction data, so both kernels
spend most of their execution time reading global memory. This makes the
differences in their block-level reduction logic difficult to observe.
Warp shuffle requires much less shared memory and fewer block-wide
synchronizations, but this advantage does not create a large performance
difference when global-memory bandwidth is the main bottleneck.



### Sum versus maximum

- Input size: 2^24 elements
- Input and accumulation type: float32
- Block size: 128 threads
- Parallel-kernel repetitions: 100
- Naive-kernel repetitions: 10
- Input access: contiguous and coalesced
- GPU: NVIDIA L4

| Implementation | Sum ms | Maximum ms | Sum GB/s | Maximum GB/s | Sum absolute error | Maximum absolute error |
|---|---:|---:|---:|---:|---:|---:|
| Naive | 420.246 | 420.220 | 0.159689 | 0.159699 | 0.0266547 | 0 |
| Interleaved shared memory | 0.258570 | 0.258765 | 259.538 | 259.343 | 0.000165474 | 0 |
| Sequential shared memory | 0.257433 | 0.257514 | 260.685 | 260.603 | 0.000104439 | 0 |
| Warp shuffle | 0.257393 | 0.257108 | 260.726 | 261.014 | 0.000104439 | 0 |

### Observations
Sum and maximum reduction produced nearly identical performance for every
implementation. Both operations read the same amount of global memory and
follow the same reduction structure, making them primarily limited by
input-memory bandwidth.
Their numerical behavior differed. Sum produced small floating-point
errors because addition order affects float32 rounding. The naive sum had
the largest error because one thread accumulated every input sequentially.
Maximum matched the CPU reference exactly because it selects an existing
input value instead of repeatedly combining values through floating-point
addition.



### Power-of-two versus irregular input sizes

- Power-of-two size: 2^24 = 16,777,216 elements
- Irregular size: 2^24 + 123 = 16,777,339 elements
- Input and accumulation type: float32
- Block size: 128 threads
- Parallel-kernel repetitions: 100
- Naive-kernel repetitions: 10
- GPU: NVIDIA L4

#### Sum reduction

| Input size | Shape | Naive ms | Interleaved ms | Sequential ms | Warp ms | Largest absolute error |
|---:|:---|---:|---:|---:|---:|---:|
| 16,777,216 | Power of two | 420.246 | 0.258570 | 0.257433 | 0.257393 | 0.0266547 |
| 16,777,339 | Irregular | 420.080 | 0.258468 | 0.257710 | 0.257475 | 0.0265079 |

#### Maximum reduction

| Input size | Shape | Naive ms | Interleaved ms | Sequential ms | Warp ms | Largest absolute error |
|---:|:---|---:|---:|---:|---:|---:|
| 16,777,216 | Power of two | 420.220 | 0.258765 | 0.257514 | 0.257108 | 0 |
| 16,777,339 | Irregular | 420.121 | 0.258580 | 0.257669 | 0.257249 | 0 |

### Observations
All implementations correctly processed both power-of-two and irregular
input sizes. Invalid positions in the final input segment were replaced
with the reduction identity: zero for sum and negative infinity for
maximum.
Performance was effectively unchanged because the irregular input added
only 123 elements to an input containing more than 16 million elements.
This demonstrates that the bounds checks support arbitrary input sizes
without materially affecting performance for large arrays.
The input size may be irregular, although the shared-memory reduction
kernels still require a power-of-two block size.