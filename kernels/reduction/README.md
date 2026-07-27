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