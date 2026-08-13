### What to learn

- Mean and variance calculations
- Multiple related reductions
- Numerical precision
- Fusing normalization with scaling
- Register reuse
- ML-framework integration
- Forward and backward considerations

### Versions to implement

1. PyTorch reference
2. Triton RMSNorm
3. Basic CUDA RMSNorm
4. Warp/block optimized CUDA RMSNorm
5. PyTorch CUDA extension


### Basic versus warp/block-optimized CUDA RMSNorm

- Rows: 4096
- Input and output type: float32
- Threads per block: 256
- Warm-up launches: 10
- Measured repetitions: 100
- GPU: NVIDIA L4

| Width | Basic ms | Optimized ms | Speedup |
|---:|---:|---:|---:|
| 128  | 0.020122 | 0.011029 | 1.82× |
| 512  | 0.024740 | 0.016189 | 1.53× |
| 1024 | 0.032277 | 0.025130 | 1.28× |
| 2048 | 0.270500 | 0.269517 | 1.00× |
| 4096 | 0.567612 | 0.567214 | 1.00× |
| 8192 | 1.124960 | 1.119800 | 1.00× |

The hierarchical warp/block reduction provided its largest benefit for
short rows. At width 128 it was approximately 1.82 times faster because
warp shuffles replaced most shared-memory accesses and block-wide
synchronizations.

The speedup decreased as row width increased. At widths 2048 and above,
the implementations performed nearly identically because reading the
input twice and writing the output dominated execution time. Optimizing
the reduction structure did not reduce this global-memory traffic.

### PyTorch CUDA extension

The optimized float32 CUDA kernel was exposed through a forward-only
PyTorch extension. It launches on PyTorch's current CUDA stream and
returns a PyTorch-managed output tensor.

For shape 4096 × 1024 on an NVIDIA L4, the extension matched
`torch.nn.functional.rms_norm` with a maximum absolute error of
approximately 1.91e-6.

Custom backward/autograd support was not implemented.