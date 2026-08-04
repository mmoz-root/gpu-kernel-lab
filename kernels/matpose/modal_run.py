"""Compile and run matrix-transpose CUDA experiments on Modal."""

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


app = modal.App("matpose-lab")


@app.function(
    image=cuda_image,
    gpu="L4",
    timeout=5 * 60,
)
def run_cuda(
    rows: int,
    cols: int,
    repetitions: int,
):
    executable = "/tmp/matpose"

    print("Compiling CUDA program...")

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

    print("Running CUDA program...")

    subprocess.run(
        [
            executable,
            str(rows),
            str(cols),
            str(repetitions),
        ],
        check=True,
    )


@app.local_entrypoint()
def main(
    rows: int = 4096,
    cols: int = 4096,
    repetitions: int = 100,
):
    run_cuda.remote(
        rows,
        cols,
        repetitions,
    )