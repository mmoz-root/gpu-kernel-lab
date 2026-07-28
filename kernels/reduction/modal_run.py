"""Compile and run CUDA reduction kernels on Modal."""

from pathlib import Path
import subprocess

import modal


local_cuda_file = (
    Path(__file__).resolve().parent
    / "cuda_impl.cu"
)

remote_cuda_file = "/root/cuda_impl.cu"


cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.12",
    )
    .entrypoint([])
    .add_local_file(
        local_cuda_file,
        remote_path=remote_cuda_file,
    )
)


app = modal.App("reduction-lab")


@app.function(
    image=cuda_image,
    gpu="L4",
    timeout=5 * 60,
)
def run_cuda(
    n: int,
    block_size: int,
):
    executable = "/tmp/reduction"

    print("Compiling CUDA reduction program...")

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

    print("Running CUDA reduction program...")

    subprocess.run(
        [
            executable,
            str(n),
            str(block_size),
        ],
        check=True,
    )


@app.local_entrypoint()
def main(
    n: int = 1 << 20,
    block_size: int = 256,
):
    run_cuda.remote(n, block_size)