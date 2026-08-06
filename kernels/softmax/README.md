### What to learn

- Row-wise reductions
- Numerical stability
- Maximum and sum reductions
- Shared memory
- Warp shuffle operations
- Kernel fusion
- Register pressure
- Different algorithms for different row sizes

### Versions to implement

1. PyTorch reference
2. Naive multi-kernel CUDA softmax
3. One-block-per-row CUDA softmax
4. Warp-based CUDA softmax
5. Triton fused softmax
6. Optional vectorized CUDA softmax

### Experiments

- Correctness across power-of-two and irregular row widths
- Negative inputs and large positive logits
- Naive multi-kernel versus block-per-row versus warp-per-row
- Performance as row width increases

### Correctness experiment

- Rows: 10
- Data type: float32
- Row widths: 1, 3, 31, 32, 33, 128, 256, 512, 1000, 1024, 2048, 4096, and 8192
- Inputs: repeating values from -8 through 8
- Large-logit case: 10000 added to every value at width 33
- Absolute-error tolerance: 1e-5
- GPU: NVIDIA L4

All three CUDA implementations passed every tested shape. The largest
observed absolute error was approximately 5.96e-08.

#### Observations

Width 1 produced exact outputs because every one-element row has softmax 1.
Widths 31, 32, and 33 validated behavior around the warp boundary, while
width 1000 tested an irregular non-power-of-two row.

The large-logit case passed without overflow because every implementation
subtracts the row maximum before computing exponentials. Small differences
between implementations are expected because their reduction orders differ.


### Row-width benchmark

- Rows: 4096
- Data type: float32
- Warm-up runs: 10
- Measured repetitions: 100
- Timing: CUDA events
- GPU: NVIDIA L4
- Correctness: all implementations passed with tolerance 1e-5

| Columns | Naive ms | Block ms | Warp ms |
|---:|---:|---:|---:|
| 128  | 0.045128 | 0.035574 | 0.005714 |
| 512  | 0.163922 | 0.041358 | 0.019804 |
| 1024 | 0.329462 | 0.049848 | 0.040970 |
| 2048 | 0.834324 | 0.281160 | 0.282307 |
| 4096 | 1.932390 | 0.553585 | 1.015640 |
| 8192 | 4.194180 | 1.097380 | 2.736680 |

#### Observations

The warp-per-row kernel was fastest for widths 128 through 1024. At width
128, it was approximately 6.2 times faster than the block kernel because
warp shuffles avoid shared-memory reductions and block-wide barriers.

The block and warp implementations were effectively tied at width 2048.
For wider rows, the block kernel became faster because 256 threads cooperate
on each row instead of only 32 lanes. At width 8192, block softmax was about
2.5 times faster than warp softmax.

The naive implementation was slower at every width. It uses four kernel
launches, sequential per-row maximum and sum reductions, and additional
global-memory traffic through intermediate buffers. At width 8192, the
block kernel was approximately 3.8 times faster than the naive pipeline.

These timings measure only the GPU kernel pipelines. Device allocation,
host-device copies, and the CPU reference are outside the CUDA-event timing.
The crossover is specific to the tested row count, GPU, and implementations.