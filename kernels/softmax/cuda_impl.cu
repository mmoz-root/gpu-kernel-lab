#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <limits>


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



__global__
void naive_row_max(const float* input, float* row_max, int rows, int cols) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows) return;

    float local_max = -CUDART_INF_F;

    int row_start = row*cols;

    for(int col = 0; col < cols; ++col) {
        local_max = fmaxf(local_max, input[row_start + col]);
    }

    row_max[row] = local_max;
}
void launch_naive_row_max(const float* input, float* row_max, int rows, int cols) {
    constexpr int threads = 256;

    int blocks = (rows + threads - 1) / threads;

    naive_row_max<<<blocks, threads>>>(input, row_max, rows, cols);
}



__global__
void compute_exp(const float* input, const float* row_max,
                 float* numerator, int rows, int cols) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int element_count = rows * cols;

    if(idx >= element_count) return;

    int row = idx / cols;

    numerator[idx] = expf(input[idx] - row_max[row]);
}
void launch_compute_exp(
    const float* input,
    const float* row_max,
    float* numerator,
    int rows,
    int cols)
{
    constexpr int threads = 256;

    int element_count = rows * cols;

    int blocks =
        (element_count + threads - 1)
        / threads;

    compute_exp<<<blocks, threads>>>(
        input,
        row_max,
        numerator,
        rows,
        cols
    );
}



__global__
void naive_row_sum(const float* numerator, float* row_sum, int rows, int cols) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows) return;

    float local_sum = 0.0;

    int row_start = row*cols;

    for(int col = 0; col < cols; ++col) {
        local_sum += numerator[row_start + col];
    }

    row_sum[row] = local_sum;
}
void launch_naive_row_sum(
    const float* numerator,
    float* row_sum,
    int rows,
    int cols)
{
    constexpr int threads = 256;

    int blocks =
        (rows + threads - 1) / threads;

    naive_row_sum<<<blocks, threads>>>(
        numerator,
        row_sum,
        rows,
        cols
    );
}



__global__
void normalize(const float* numerator, const float* row_sum, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int element_count = rows * cols;

    if(idx >= element_count) return;

    int row = idx / cols;

    output[idx] = numerator[idx] / row_sum[row];
}
void launch_normalize(const float* numerator, const float* row_sum,
                      float* output, int rows, int cols){
    constexpr int threads = 256;
    
    int element_count = rows * cols;

    int blocks = (element_count + threads - 1) / threads;

    normalize<<<blocks, threads>>>(
        numerator,
        row_sum,
        output,
        rows,
        cols
    );
}



/*
row max completes
    ↓
exponentials complete
    ↓
row sums complete
    ↓
normalization completes
*/
void launch_naive_softmax(
    const float* input,
    float* output,
    float* row_max,
    float* numerator,
    float* row_sum,
    int rows,
    int cols)
{
    launch_naive_row_max(
        input,
        row_max,
        rows,
        cols
    );

    launch_compute_exp(
        input,
        row_max,
        numerator,
        rows,
        cols
    );

    launch_naive_row_sum(
        numerator,
        row_sum,
        rows,
        cols
    );

    launch_normalize(
        numerator,
        row_sum,
        output,
        rows,
        cols
    );
}


void cpu_softmax(
    const std::vector<float>& input,
    std::vector<float>& output,
    int rows,
    int cols)
{
    output.resize(input.size());

    for (int row = 0; row < rows; ++row) {
        int row_start = row * cols;

        float max_val = -std::numeric_limits<float>::infinity();

        for (int col = 0; col < cols; ++col) {
            int index = row_start + col;

            max_val = std::max(
                max_val,
                input[index]
            );
        }

        float sum = 0.0f;

        for (int col = 0; col < cols; ++col) {
            int index = row_start + col;

            float numerator = std::exp(
                input[index] - max_val
            );

            output[index] = numerator;
            sum += numerator;
        }

        float inverse_sum = 1.0f / sum;

        for (int col = 0; col < cols; ++col) {
            int index = row_start + col;

            output[index] *= inverse_sum;
        }
    }
}


int main()
{
    constexpr int rows = 2;
    constexpr int cols = 3;

    std::vector<float> input = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };

    std::vector<float> expected;

    cpu_softmax(
        input,
        expected,
        rows,
        cols
    );

    std::cout << "CPU reference:\n";

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            std::cout
                << expected[row * cols + col]
                << " ";
        }

        std::cout << "\n";
    }
    std::size_t element_count =
    static_cast<std::size_t>(rows) * cols;

    std::size_t matrix_bytes =
        element_count * sizeof(float);

    std::size_t row_bytes =
        static_cast<std::size_t>(rows) * sizeof(float);

    std::vector<float> actual(element_count);

    float* d_input = nullptr;
    float* d_output = nullptr;
    float* d_row_max = nullptr;
    float* d_numerator = nullptr;
    float* d_row_sum = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_input),
        matrix_bytes
    ));
    
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_output),
        matrix_bytes
    ));
    
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_row_max),
        row_bytes
    ));
    
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_numerator),
        matrix_bytes
    ));
    
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_row_sum),
        row_bytes
    ));
    CUDA_CHECK(cudaMemcpy(
        d_input,
        input.data(),
        matrix_bytes,
        cudaMemcpyHostToDevice
    ));
    
    launch_naive_softmax(
        d_input,
        d_output,
        d_row_max,
        d_numerator,
        d_row_sum,
        rows,
        cols
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(
        actual.data(),
        d_output,
        matrix_bytes,
        cudaMemcpyDeviceToHost
    ));

    std::cout << "GPU reference:\n";
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            std::cout
                << actual[row * cols + col]
                << " ";
        }

        std::cout << "\n";
    }

    float max_error = 0.0f;

    for (std::size_t i = 0; i < element_count; ++i) {
        max_error = std::max(
            max_error,
            std::abs(actual[i] - expected[i])
        );
    }

    bool passed = max_error < 1e-5f;

    std::cout
        << "Naive softmax max error: "
        << max_error
        << "\n";

    std::cout
        << "Naive softmax: "
        << (passed ? "PASSED" : "FAILED")
        << "\n";

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_row_max));
    CUDA_CHECK(cudaFree(d_numerator));
    CUDA_CHECK(cudaFree(d_row_sum));

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}




