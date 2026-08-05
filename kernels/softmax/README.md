### What it teaches

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

### Input sizes

Test row widths such as:
128, 256, 512, 1024, 2048, 4096, 8192