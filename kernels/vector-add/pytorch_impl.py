import time

import torch


def vector_add(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    """Add x and y elementwise, writing the result into out."""
    torch.add(x, y, out=out)


def main() -> None:
    n = 1 << 20  # 1,048,576 elements
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.ones(n, device=device, dtype=torch.float32)
    y = torch.full((n,), 2.0, device=device, dtype=torch.float32)
    result = torch.empty_like(x)

    # Warm up PyTorch/CUDA before measuring.
    for _ in range(10):
        vector_add(x, y, result)

    if device.type == "cuda":
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        vector_add(x, y, result)
        end.record()

        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        start_time = time.perf_counter()
        vector_add(x, y, result)
        elapsed_ms = (time.perf_counter() - start_time) * 1_000

    max_error = (result - 3.0).abs().max().item()

    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Elements: {n:,}")
    print(f"Elapsed time: {elapsed_ms:.4f} ms")
    print(f"Max error: {max_error}")


if __name__ == "__main__":
    main()
