"""Compile and run CUDA RMSNorm kernels on a Modal L4 GPU."""

from pathlib import Path
import subprocess

import modal


local_cuda_file = (
    Path(__file__).resolve().parent
    / "cuda_impl.cu"
)

remote_cuda_file = "/root/cuda_impl.cu"


base_cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.12",
    )
    .entrypoint([])
)


cuda_image = (
    base_cuda_image
    .add_local_file(
        local_cuda_file,
        remote_path=remote_cuda_file,
    )
)


extension_image = (
    base_cuda_image
    .pip_install(
        "torch",
        "ninja",
    )
    .add_local_file(
        local_cuda_file,
        remote_path=remote_cuda_file,
    )
)


app = modal.App("rmsnorm-lab")


@app.function(
    image=cuda_image,
    gpu="L4",
    timeout=5 * 60,
)
def run_cuda(
    rows: int,
    cols: int,
    repetitions: int,
    sweep: bool,
):
    executable = "/tmp/rmsnorm"

    print("Compiling CUDA RMSNorm program...")

    subprocess.run(
        [
            "nvcc",
            "-O3",
            "-lineinfo",
            "-std=c++17",
            "-arch=sm_89",
            remote_cuda_file,
            "-o",
            executable,
        ],
        check=True,
    )

    print("Running CUDA RMSNorm program...")

    command = (
        [executable]
        if sweep
        else [
            executable,
            str(rows),
            str(cols),
            str(repetitions),
        ]
    )

    subprocess.run(command, check=True)

@app.function(
    image=extension_image,
    gpu="L4",
    timeout=10 * 60,
)
def run_pytorch_extension(
    rows: int,
    cols: int,
):
    import torch
    import torch.nn.functional as F
    from torch.utils.cpp_extension import load

    print("Building PyTorch CUDA extension...")

    extension = load(
        name="rmsnorm_cuda",
        sources=[remote_cuda_file],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "-arch=sm_89",
        ],
        verbose=True,
    )

    print("Testing PyTorch CUDA extension...")

    torch.manual_seed(0)

    input = torch.randn(
        rows,
        cols,
        device="cuda",
        dtype=torch.float32,
    )

    weight = torch.randn(
        cols,
        device="cuda",
        dtype=torch.float32,
    )

    eps = 1e-6

    actual = extension.forward(
        input,
        weight,
        eps,
    )

    expected = F.rms_norm(
        input,
        normalized_shape=(cols,),
        weight=weight,
        eps=eps,
    )

    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-5,
        rtol=1e-5,
    )

    max_error = (
        actual - expected
    ).abs().max().item()

    print("shape:", actual.shape)
    print("dtype:", actual.dtype)
    print("max error:", max_error)
    print("PyTorch CUDA extension: PASSED")


@app.local_entrypoint()
def main(
    rows: int = 4096,
    cols: int = 1024,
    repetitions: int = 100,
    sweep: bool = False,
    extension: bool = False,
):
    if rows <= 0 or cols <= 0 or repetitions <= 0:
        raise ValueError(
            "rows, cols, and repetitions must be positive"
        )

    if extension:
        run_pytorch_extension.remote(
            rows,
            cols,
        )
        return

    run_cuda.remote(
        rows,
        cols,
        repetitions,
        sweep,
    )
