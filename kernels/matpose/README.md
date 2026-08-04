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

### Tile-size and shared-memory padding experiment

- Matrix shape: 4096 × 4096
- Matrix elements: 2^24
- Data type: float32
- Tile sizes: 8 × 8, 16 × 16, and 32 × 32
- Naive block size: 16 × 16
- Warm-up runs: 10
- Measured repetitions: 100
- GPU: NVIDIA L4
- Correctness: all implementations passed exact comparisons

| Tile | Unpadded ms | Unpadded GB/s | Padded ms | Padded GB/s |
|---:|---:|---:|---:|---:|
| 8  | 0.561633 | 238.978 | 0.560753 | 239.353 |
| 16 | 0.560906 | 239.287 | 0.560497 | 239.462 |
| 32 | 0.583363 | 230.076 | 0.554435 | 242.080 |

#### Baselines

| Implementation | Average ms | Effective GB/s |
|:---|---:|---:|
| Naive transpose | 0.559974 | 239.685 |
| Device-to-device copy | 0.583895 | 229.866 |

#### Observations

Performance was similar across all implementations at approximately
230–242 GB/s, confirming that transpose is primarily memory-bandwidth-bound.

Padding had almost no effect for tile sizes 8 and 16. For tile size 32,
padding improved execution time by approximately 5%, from 0.583 ms to
0.554 ms, by removing the 32-way shared-memory bank conflict.

The padded 32 × 32 kernel was the fastest implementation, although it was
only about 1% faster than the naive kernel. The measured memory copy was
slightly slower, so it serves as a comparison baseline rather than a strict
performance ceiling.



### Square versus rectangular matrices

- Matrix elements: 2^24
- Data type: float32
- Shared-memory tile size: 32 × 32
- Repetitions: 100
- GPU: NVIDIA L4

| Shape | Naive ms | Naive GB/s | Tiled ms | Tiled GB/s | Padded ms | Padded GB/s | Copy GB/s |
|:---|---:|---:|---:|---:|---:|---:|---:|
| 4096 × 4096 | 0.559974 | 239.685 | 0.583363 | 230.076 | 0.554435 | 242.080 | 229.866 |
| 2048 × 8192 | 0.565709 | 237.256 | 0.583332 | 230.088 | 0.555151 | 241.768 | 230.270 |
| 8192 × 2048 | 0.559708 | 239.799 | 0.575457 | 233.237 | 0.549663 | 244.182 | 230.072 |

#### Observations

Matrix orientation had only a small effect because all shapes contained the
same number of elements.

The padded 32 × 32 kernel was fastest for every shape, reaching
241.8–244.2 GB/s. Padding improved the unpadded tiled kernel by approximately
4.5–5.0% by removing shared-memory bank conflicts.

Memory-copy bandwidth remained nearly constant at approximately 230 GB/s
because copying does not depend on matrix shape.


### Effective bandwidth versus memory copy

- Matrix shape: 4096 × 4096
- Matrix elements: 2^24
- Data type: float32
- Shared-memory tile size: 32 × 32
- Repetitions: 100
- GPU: NVIDIA L4

| Implementation | Average ms | Effective GB/s | Relative to copy |
|:---|---:|---:|---:|
| Device-to-device copy | 0.583895 | 229.866 | 100.0% |
| Naive transpose | 0.559974 | 239.685 | 104.3% |
| Unpadded tiled transpose | 0.583363 | 230.076 | 100.1% |
| Padded tiled transpose | 0.554435 | 242.080 | 105.3% |

#### Observations

The unpadded 32 × 32 transpose closely matched the measured memory-copy
bandwidth. Removing bank conflicts increased bandwidth from 230.1 GB/s to
242.1 GB/s.

The padded transpose slightly exceeded the measured copy bandwidth, so the
copy result is a practical comparison baseline rather than a strict ceiling.


> Transpose moves little computation but demands careful memory layout. Shared-memory tiling coalesces global accesses, while padding changes the physical row stride to eliminate bank conflicts.