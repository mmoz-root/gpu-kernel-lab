"""Compile and run CUDA RMSNorm kernels on a Modal L4 GPU."""

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


@app.local_entrypoint()
def main(
    rows: int = 4096,
    cols: int = 1024,
    repetitions: int = 100,
    sweep: bool = False,
):
    if rows <= 0 or cols <= 0 or repetitions <= 0:
        raise ValueError(
            "rows, cols, and repetitions must be positive"
        )

    run_cuda.remote(
        rows,
        cols,
        repetitions,
        sweep,
    )
