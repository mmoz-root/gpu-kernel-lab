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