#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>


void check_cuda(
    cudaError_t error,
    const char* operation,
    const char* file,
    int line)
{
    if (error != cudaSuccess) {
        std::cerr
            << "CUDA error during "
            << operation
            << ": "
            << cudaGetErrorString(error)
            << " at "
            << file
            << ":"
            << line
            << "\n";

        std::exit(EXIT_FAILURE);
    }
}


#define CUDA_CHECK(operation) \
    check_cuda(               \
        (operation),          \
        #operation,           \
        __FILE__,             \
        __LINE__              \
    )
    
// constexpr int TILE_SIZE = 32;

__global__
void matpose_naive(const float* input, float* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= cols || row >= rows) return;

    int inp_idx = row * cols + col;
    int out_idx = col * rows + row;

    output[out_idx] = input[inp_idx];
}

void launch_matpose_naive(const float* input, float* output, int rows, int cols){
    dim3 block(16, 16);
    dim3 grid(
        ((cols + block.x - 1)/block.x),
        ((rows + block.y - 1)/block.y)
    );

    matpose_naive<<<grid, block>>>(input, output, rows, cols);

}


template <int TILE>
__global__
void matpose_tiled(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE][TILE];

    int input_col = blockIdx.x * TILE + threadIdx.x;
    int input_row = blockIdx.y * TILE + threadIdx.y;

    if (input_col < cols && input_row < rows) {
        tile[threadIdx.y][threadIdx.x] = input[input_row * cols + input_col];
    }

    __syncthreads();

    int output_col = blockIdx.y * TILE + threadIdx.x;
    int output_row = blockIdx.x * TILE + threadIdx.y;

    if (output_col < rows && output_row < cols) {
        output[output_row * rows + output_col] = tile[threadIdx.x][threadIdx.y];
    }
}

template <int TILE>
void launch_matpose_tiled(const float* input, float* output, int rows, int cols) {
    dim3 block(TILE, TILE);

    dim3 grid(
        ((cols + TILE - 1)/TILE),
        ((rows + TILE - 1)/TILE)
    );

    matpose_tiled<TILE><<<grid, block>>>(input, output, rows, cols);
}


template <int TILE>
__global__
void matpose_tiled_padded(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE][TILE+1];

    int input_col = blockIdx.x * TILE + threadIdx.x;
    int input_row = blockIdx.y * TILE + threadIdx.y;

    if (input_col < cols && input_row < rows) {
        tile[threadIdx.y][threadIdx.x] = input[input_row * cols + input_col];
    }

    __syncthreads();

    int output_col = blockIdx.y * TILE + threadIdx.x;
    int output_row = blockIdx.x * TILE + threadIdx.y;

    if (output_col < rows && output_row < cols) {
        output[output_row * rows + output_col] = tile[threadIdx.x][threadIdx.y];
    }
}

template <int TILE>
void launch_matpose_tiled_padded(const float* input, float* output, int rows, int cols) {
    dim3 block(TILE, TILE);

    dim3 grid(
        ((cols + TILE - 1)/TILE),
        ((rows + TILE - 1)/TILE)
    );

    matpose_tiled_padded<TILE><<<grid, block>>>(input, output, rows, cols);
}

using TransposeLauncher = void (*)(
    const float*,
    float*,
    int,
    int
);

void launch_memory_copy(
    const float* input,
    float* output,
    int rows,
    int cols)
{
    std::size_t bytes =
        static_cast<std::size_t>(rows)
        * cols
        * sizeof(float);

    CUDA_CHECK(cudaMemcpyAsync(
        output,
        input,
        bytes,
        cudaMemcpyDeviceToDevice
    ));
}



bool check_transpose(
    const char* name,
    TransposeLauncher launch,
    int rows,
    int cols)
{
    std::size_t element_count =
        static_cast<std::size_t>(rows) * cols;

    std::size_t bytes =
        element_count * sizeof(float);

    std::vector<float> input(element_count);
    std::vector<float> output(element_count);

    for (std::size_t i = 0; i < element_count; ++i) {
        input[i] = static_cast<float>(i);
    }

    float* d_input = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_input),
        bytes
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_output),
        bytes
    ));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        input.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    launch(
        d_input,
        d_output,
        rows,
        cols
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        output.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            std::size_t input_index =
                static_cast<std::size_t>(row) * cols + col;

            std::size_t output_index =
                static_cast<std::size_t>(col) * rows + row;

            if (output[output_index] != input[input_index]) {
                std::cerr
                    << name
                    << " failed for "
                    << rows << "x" << cols
                    << " at input coordinate ("
                    << row << ", " << col << ")\n";

                return false;
            }
        }
    }

    return true;
}

float benchmark_transpose(
    TransposeLauncher launch,
    const float* d_input,
    float* d_output,
    int rows,
    int cols,
    int repetitions)
{
    constexpr int WARMUP_RUNS = 10;

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < WARMUP_RUNS; ++i) {
        launch(
            d_input,
            d_output,
            rows,
            cols
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repetitions; ++i) {
        launch(
            d_input,
            d_output,
            rows,
            cols
        );
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &total_ms,
        start,
        stop
    ));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / repetitions;
}

double transpose_bandwidth(
    int rows,
    int cols,
    float average_ms)
{
    double element_count =
        static_cast<double>(rows)
        * static_cast<double>(cols);

    // One complete input read and one complete output write.
    double transferred_bytes =
        2.0
        * element_count
        * sizeof(float);

    return transferred_bytes / (
        average_ms * 1.0e6
    );
}

void run_benchmark(
    const char* name,
    TransposeLauncher launch,
    const float* d_input,
    float* d_output,
    int rows,
    int cols,
    int repetitions)
{
    float average_ms = benchmark_transpose(
        launch,
        d_input,
        d_output,
        rows,
        cols,
        repetitions
    );

    double bandwidth = transpose_bandwidth(
        rows,
        cols,
        average_ms
    );

    std::cout
        << name
        << ":\n"
        << "  Average time: "
        << average_ms
        << " ms\n"
        << "  Effective bandwidth: "
        << bandwidth
        << " GB/s\n";
}


int main(int argc, char** argv)
{
    int benchmark_rows = argc > 1 ? std::atoi(argv[1]) : 4096;

    int benchmark_cols = argc > 2 ? std::atoi(argv[2]) : 4096;

    int repetitions = argc > 3 ? std::atoi(argv[3]) : 100;

if (
    benchmark_rows <= 0
    || benchmark_cols <= 0
    || repetitions <= 0
) {
    std::cerr
        << "Rows, columns, and repetitions "
        << "must be positive\n";

    return EXIT_FAILURE;
}
    struct Shape {
        int rows;
        int cols;
    };

    struct Implementation {
        const char* name;
        TransposeLauncher launch;
    };

    Shape shapes[] = {
        {1, 1},
        {2, 3},
        {3, 2},
        {31, 47},
        {32, 32},
        {33, 65},
        {1000, 1500},
    };

    Implementation implementations[] = {
        {
            "Naive",
            launch_matpose_naive
        },
        {
            "Tiled 8",
            launch_matpose_tiled<8>
        },
        {
            "Tiled 16",
            launch_matpose_tiled<16>
        },
        {
            "Tiled 32",
            launch_matpose_tiled<32>
        },
        {
            "Padded 8",
            launch_matpose_tiled_padded<8>
        },
        {
            "Padded 16",
            launch_matpose_tiled_padded<16>
        },
        {
            "Padded 32",
            launch_matpose_tiled_padded<32>
        },
    };

    bool all_passed = true;

    for (const Implementation& implementation : implementations) {
        for (const Shape& shape : shapes) {
            bool passed = check_transpose(
                implementation.name,
                implementation.launch,
                shape.rows,
                shape.cols
            );

            if (!passed) {
                all_passed = false;
            }
        }
    }
    std::size_t benchmark_elements =
    static_cast<std::size_t>(benchmark_rows)
    * benchmark_cols;

    std::size_t benchmark_bytes =
        benchmark_elements * sizeof(float);

    float* d_benchmark_input = nullptr;
    float* d_benchmark_output = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_benchmark_input),
        benchmark_bytes
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_benchmark_output),
        benchmark_bytes
    ));

    CUDA_CHECK(cudaMemset(
        d_benchmark_input,
        0,
        benchmark_bytes
    ));

    std::cout
    << "\nBenchmark shape: "
    << benchmark_rows
    << "x"
    << benchmark_cols
    << "\n"
    << "Repetitions: "
    << repetitions
    << "\n\n";

    run_benchmark(
        "Naive",
        launch_matpose_naive,
        d_benchmark_input,
        d_benchmark_output,
        benchmark_rows,
        benchmark_cols,
        repetitions
    );

    run_benchmark(
        "Tiled 8",
        launch_matpose_tiled<8>,
        d_benchmark_input,
        d_benchmark_output,
        benchmark_rows,
        benchmark_cols,
        repetitions
    );

    run_benchmark(
        "Tiled 16",
        launch_matpose_tiled<16>,
        d_benchmark_input,
        d_benchmark_output,
        benchmark_rows,
        benchmark_cols,
        repetitions
    );

    run_benchmark(
        "Tiled 32",
        launch_matpose_tiled<32>,
        d_benchmark_input,
        d_benchmark_output,
        benchmark_rows,
        benchmark_cols,
        repetitions
    );

    run_benchmark(
        "Padded 8",
        launch_matpose_tiled_padded<8>,
        d_benchmark_input,
        d_benchmark_output,
        benchmark_rows,
        benchmark_cols,
        repetitions
    );

    run_benchmark(
        "Padded 16",
        launch_matpose_tiled_padded<16>,
        d_benchmark_input,
        d_benchmark_output,
        benchmark_rows,
        benchmark_cols,
        repetitions
    );

    run_benchmark(
        "Padded 32",
        launch_matpose_tiled_padded<32>,
        d_benchmark_input,
        d_benchmark_output,
        benchmark_rows,
        benchmark_cols,
        repetitions
    );

    run_benchmark(
        "Device-to-device copy",
        launch_memory_copy,
        d_benchmark_input,
        d_benchmark_output,
        benchmark_rows,
        benchmark_cols,
        repetitions
    );

    CUDA_CHECK(cudaFree(d_benchmark_input));
    CUDA_CHECK(cudaFree(d_benchmark_output));

    return EXIT_SUCCESS;
}