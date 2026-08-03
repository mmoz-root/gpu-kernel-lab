### What to learn
- Two-dimensional thread indexing
- Coalesced reads and writes
- Shared-memory tiling
- Shared-memory bank conflicts
- Padding shared-memory tiles
- Why logically simple operations can be difficult to optimize

### Version to implement
1. PyTorch reference
2. Naive CUDA transpose
3. Shared-memory tiled transpose
4. Bank-conflict-free tiled transpose
5. Optional Triton transpose

### Experiments
- Square and rectangular matrices
- Different tile sizes
- Shared-memory tiles with and without padding
- Compare effective bandwidth with a memory copy