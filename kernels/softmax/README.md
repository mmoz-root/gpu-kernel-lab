### What to learn

- Row-wise reductions
- Numerical stability
- Maximum and sum reductions
- Shared memory
- Warp shuffle operations
- Kernel fusion
- Register pressure
- Different algorithms for different row sizes
- Vectorized `float4` memory access
- Alignment requirements and scalar fallbacks

### Versions to implement

1. PyTorch reference
2. Naive multi-kernel CUDA softmax
3. One-block-per-row CUDA softmax
4. Warp-based CUDA softmax
5. Triton fused softmax
6. Vectorized CUDA softmax using float4

### Experiments

- Correctness across power-of-two and irregular row widths
- Negative inputs and large positive logits
- Naive multi-kernel versus block-per-row versus warp-per-row versus vectorized CUDA
- Performance as row width increases

### Correctness experiment

- Rows: 10
- Data type: float32
- Row widths: 1, 3, 31, 32, 33, 128, 256, 512, 1000, 1024, 2048, 4096, and 8192
- Inputs: repeating values from -8 through 8
- Large-logit case: 10000 added to every value at width 33
- Absolute-error tolerance: 1e-5
- GPU: NVIDIA L4

All 4 CUDA implementations passed every tested shape. The largest
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

| Columns | Naive ms | Block ms | Warp ms | Vectorized ms |
|---:|---:|---:|---:|---:|
| 128  | 0.045562 | 0.036147 | 0.005856 | 0.034816 |
| 512  | 0.162693 | 0.041267 | 0.019825 | 0.036680 |
| 1024 | 0.327260 | 0.049705 | 0.041820 | 0.040059 |
| 2048 | 0.825887 | 0.282041 | 0.286054 | 0.279450 |
| 4096 | 1.938340 | 0.552100 | 1.017540 | 0.558510 |
| 8192 | 4.138650 | 1.096850 | 2.722930 | 1.103250 |

#### Observations

Warp-per-row softmax was fastest for short rows. At width 128, it was
approximately six times faster than both block-based kernels because it
uses only 32 lanes per row and avoids block-wide shared-memory barriers.
It remained fastest at width 512.

Vectorized block softmax was fastest at widths 1024 and 2048. At width 1024,
`float4` improved the scalar block kernel by approximately 19% and was about
4% faster than warp softmax. At width 2048, the three fused implementations
were close, with vectorized softmax ahead by only 1–2%.

For widths 4096 and 8192, scalar block and vectorized block softmax were
effectively tied. Both were much faster than warp softmax because 256
threads cooperate on each wide row instead of only 32 lanes.

Vectorization did not provide a universal improvement. The scalar block
kernel already uses coalesced memory access, while exponentials, reductions,
and synchronization remain unchanged. Reduced memory-instruction counts can
be offset by additional registers and component-wise `float4` arithmetic.

The vectorized launcher uses `float4` only when the width is divisible by
four. Other widths fall back to scalar block softmax so every row remains
properly aligned and all tail elements are processed.

The naive implementation remained slower because it uses four kernel
launches, sequential row reductions, and intermediate global-memory buffers.
These timings exclude allocation, host-device copies, and the CPU reference.
The crossover is specific to the NVIDIA L4 and the tested row count.