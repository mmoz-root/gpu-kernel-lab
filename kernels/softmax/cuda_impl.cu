#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <system_error>
#include <vector>
#include <limits>
#include <string>



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




__global__
void block_softmax(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    int row_start = row*cols;

    // each thread finds its local max
    float local_max = -CUDART_INF_F;

    for(int col=tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, input[row_start+col]);
    }

    // reduce the thread local maxima
    shared[tid] = local_max;
    __syncthreads();

    for(int stride = blockDim.x/2; stride>0; stride /= 2) {
        if(tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid+stride]);
        }
        __syncthreads();
    }

    float row_max = shared[0];
    __syncthreads();


    // compute numerators and thread-local sums
    float local_sum = 0.0f;

    for(int col = tid; col<cols; col+=blockDim.x) {
        int index = row_start + col;

        float numerator = expf(input[index] - row_max);

        output[index] = numerator;

        local_sum += numerator;
    }

    // reduce thread local sums

    shared[tid] = local_sum;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride>0; stride /= 2) {
        if(tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float denominator = shared[0];

    float inverse_denominator = 1.0f / denominator;

    for(int col = tid; col < cols; col += blockDim.x) {
        int index = row_start + col;

        output[index] *= inverse_denominator;
    }
}
void launch_block_softmax(
    const float* input,
    float* output,
    int rows,
    int cols)
{
    constexpr int threads = 256;

    int blocks = rows;

    std::size_t shared_bytes =
        threads * sizeof(float);

    block_softmax<<<
        blocks,
        threads,
        shared_bytes
    >>>(
        input,
        output,
        rows,
        cols
    );
}



// reusable device helpers for warp design
__device__
float warp_reduce_max(float value) {
    constexpr unsigned mask = 0xffffffffu;

    for(int offs = 16; offs > 0; offs /= 2) {
        value = fmaxf(value, __shfl_down_sync(mask, value, offs));
    }
    return __shfl_sync(mask, value, 0);
}

__device__
float warp_reduce_sum(float value) {
    constexpr unsigned mask = 0xffffffffu;

    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(
            mask,
            value,
            offset
        );
    }

    return __shfl_sync(
        mask,
        value,
        0
    );
}

__global__
void warp_softmax(const float* input, float* output, int rows, int cols) {
    constexpr int warp_size = 32;

    int warp_in_block = threadIdx.x / warp_size;

    int lane = threadIdx.x % warp_size;

    int warps_per_block = blockDim.x / warp_size;

    int row = blockIdx.x * warps_per_block + warp_in_block;

    if(row >= rows) return;

    int row_start = row * cols;

    float local_max = -CUDART_INF_F;

    for(int col = lane; col < cols; col += warp_size) {
        local_max = fmaxf(local_max, input[row_start + col]);
    }

    float row_max = warp_reduce_max(local_max);

    float local_sum = 0.0f;

    for(int col = lane; col < cols; col += warp_size) {
        int index = row_start + col;

        float numerator = expf(input[index] - row_max);

        output[index] = numerator;
        local_sum += numerator;
    }

    float denominator = warp_reduce_sum(local_sum);

    float inverse_denominator = 1.0f / denominator;

    for(int col = lane; col<cols; col+=warp_size) {
        int index = row_start + col;

        output[index] *= inverse_denominator;
    }
}
void launch_warp_softmax(const float* input, float* output, int rows, int cols) {
    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (rows + warps_per_block - 1) / warps_per_block;

    warp_softmax<<<blocks, threads>>>(input, output, rows, cols);
}


__device__
float max_float4(float4 val) {
    return fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w));
}
__device__
float sum_float4(float4 val) {
    return (val.x + val.y) + (val.z + val.w);
}

__global__
void vectorized_block_softmax(const float* input, float* output, int rows, int cols) {
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if(row >= rows) return;

    int row_start = row*cols;

    int vector_count = cols/4;

    const float4* input4 = reinterpret_cast<const float4*>(input+row_start);

    float local_max = -CUDART_INF_F;

    for(int vec = tid; vec < vector_count; vec+=blockDim.x) {
        float4 values = input4[vec];
        local_max = fmaxf(local_max, max_float4(values));
    }

    shared[tid] = local_max;
    __syncthreads();

    for(int stride = blockDim.x/2; stride > 0; stride/=2) {
        if(tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid+stride]);
        }
        __syncthreads();
    }
    float row_max = shared[0];

    __syncthreads();

    float4* output4 = reinterpret_cast<float4*>(output+row_start);

    float local_sum = 0.0f;

    for(int vec = tid; vec < vector_count; vec+=blockDim.x) {
        float4 values = input4[vec];

        float4 numerators;

        numerators.x = expf(values.x - row_max);
        numerators.y = expf(values.y - row_max);
        numerators.z = expf(values.z - row_max);
        numerators.w = expf(values.w - row_max);

        output4[vec] = numerators;
        local_sum += sum_float4(numerators);
    }

    shared[tid] = local_sum;
    __syncthreads();

    for(int stride = blockDim.x/2; stride > 0; stride/=2) {
        if(tid < stride) {
            shared[tid] += shared[tid+stride];
        }
        __syncthreads();
    }

    float denominator = shared[0];

    float inverse_denominator = 1.0f / denominator;

    for(int vec = tid; vec < vector_count; vec+=blockDim.x) {
        float4 values = output4[vec];
        values.x *= inverse_denominator;
        values.y *= inverse_denominator;
        values.z *= inverse_denominator;
        values.w *= inverse_denominator;
       output4[vec] = values;
    }
}

void launch_vectorized_softmax(const float* input, float* output, int rows, int cols) {
    if (cols % 4 != 0) {
        launch_block_softmax(input, output, rows, cols);
        return;
    }

    int threads = 256;
    int blocks = rows;

    std::size_t shared_bytes = threads * sizeof(float);

    vectorized_block_softmax<<<blocks, threads, shared_bytes>>>(input, output, rows, cols);

}



float max_abs_error(
    const std::vector<float>&actual,
    const std::vector<float>&expected
) {
    float max_error = 0.0f;

    for(std::size_t i = 0; i < actual.size(); ++i) {
        max_error = std::max(max_error, std::abs(actual[i] - expected[i]));
    }
    return max_error;
}

std::vector<float> make_test_input(int rows, int cols) {
    std::size_t element_count = static_cast<std::size_t>(rows)*cols;

    std::vector<float> input(element_count);

    for(std::size_t i = 0; i<element_count; ++i) {
        input[i] = static_cast<float>(static_cast<int>(i%17) -8);
    }

    return input;
}
std::vector<float> make_large_logit_input(int rows, int cols) {
    std::vector<float> input = make_test_input(rows, cols);

    for(float& value : input) {
        value += 10000.0f;
    }
    return input;    
}


template <typename LaunchFunction>
void warmup_cuda_kernel(LaunchFunction launch, int warmups) {
    for(int i = 0; i<warmups; ++i) {
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename LaunchFunction>
float benchmark_cuda_kernel(LaunchFunction launch, int warmups, int repetitions) {
    warmup_cuda_kernel(launch, warmups);

    cudaEvent_t start, stop;
    
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));

    for(int i = 0; i < repetitions; i++) {
        launch();
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / repetitions;
}




bool run_correctness_test(const std::vector<float>& input,
int rows, int cols, int warmups, int repetitions) {

    std::cout << "Correctness test: rows="
    << rows
    << ", cols="
    << cols
    << "\n";

    std::vector<float> expected;

    cpu_softmax(
        input,
        expected,
        rows,
        cols
    );

    /*std::cout << "CPU reference:\n";

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            std::cout
                << expected[row * cols + col]
                << " ";
        }

        std::cout << "\n";
    }*/

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

    /*std::cout << "Naive Softmax Output:\n";
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            std::cout
                << actual[row * cols + col]
                << " ";
        }

        std::cout << "\n";
    }*/

    float max_error = max_abs_error(actual, expected);

    bool passed = max_error < 1e-5f;

    std::cout
        << "Naive softmax max error: "
        << max_error
        << "\n";

    std::cout
        << "Naive softmax: "
        << (passed ? "PASSED" : "FAILED")
        << "\n";
    auto launch_naive = [&]() {
        launch_naive_softmax(d_input, d_output, d_row_max, d_numerator, d_row_sum, rows, cols);
    };
 
    float naive_ms =
        benchmark_cuda_kernel(launch_naive, warmups, repetitions);
 
    std::cout << "Naive avg time: "
              << naive_ms
              << " ms\n";


    launch_block_softmax(
        d_input,
        d_output,
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
    /*std::cout << "Block softmax output:\n";

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            std::cout
                << actual[row * cols + col]
                << " ";
        }
    
        std::cout << "\n";
    }*/

    float block_max_error = max_abs_error(actual, expected);

    bool block_passed =
        block_max_error < 1e-5f;

    std::cout
        << "Block softmax max error: "
        << block_max_error
        << "\n";

    std::cout
        << "Block softmax: "
        << (block_passed ? "PASSED" : "FAILED")
        << "\n";
    auto launch_block = [&]() {
        launch_block_softmax(d_input, d_output, rows, cols);
    };
 
    float block_ms =
        benchmark_cuda_kernel(launch_block, warmups, repetitions);
 
    std::cout << "Block avg time: "
              << block_ms
              << " ms\n";

    launch_warp_softmax(d_input, d_output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(actual.data(), d_output, matrix_bytes, cudaMemcpyDeviceToHost));


    /*std::cout << "Warp softmax output:\n";
    for(int row = 0; row<rows; ++row) {
        for(int col = 0; col<cols; ++col) {
            std::cout << actual[row*cols + col] << " ";
        }
        std::cout << "\n";
    }*/


    float warp_max_error = max_abs_error(actual, expected);
    bool warp_passed = warp_max_error < 1e-5f;

    std::cout << "Warp softmax max error: "
    << warp_max_error
    << "\n";

    std::cout << "Warp softmax: "
    << (warp_passed ? "PASSED" : "FAILED")
    << "\n";

   
    auto launch_warp = [&]() {
        launch_warp_softmax(d_input, d_output, rows, cols);
    };
    float warp_ms = benchmark_cuda_kernel(launch_warp, warmups, repetitions);

    std::cout << "Warp avg time: "
    << warp_ms
    << " ms\n";


    launch_vectorized_softmax(d_input, d_output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(actual.data(), d_output, matrix_bytes, cudaMemcpyDeviceToHost));
    
    float vectorized_max_error = max_abs_error(actual, expected);

    bool vectorized_passed = vectorized_max_error < 1e-5f;

    std::cout << "Vectorized softmax max error: "
    << vectorized_max_error
    << "\n";

    std::cout << "Vectorized softmax: "
    << (vectorized_passed ? "PASSED" : "FAILED")
    << "\n";

    auto launch_vectorized = [&]() {
        launch_vectorized_softmax(d_input, d_output, rows, cols);
    };
    float vectorized_ms = benchmark_cuda_kernel(launch_vectorized, warmups, repetitions);

    std::cout << "Vectorized avg time: "
    << vectorized_ms
    << " ms\n";
    
    passed = passed && block_passed && warp_passed && vectorized_passed;

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_row_max));
    CUDA_CHECK(cudaFree(d_numerator));
    CUDA_CHECK(cudaFree(d_row_sum));

    return passed;
}



int main(int argc, char* argv[]) {
    if (argc != 1 && argc != 4) {
        std::cerr
            << "Usage: "
            << argv[0]
            << " [rows cols repetitions]\n";
 
        return EXIT_FAILURE;
    }

    if (argc == 4) {
        int rows = std::stoi(argv[1]);
        int cols = std::stoi(argv[2]);
        int repetitions = std::stoi(argv[3]);
        int warmups = 10;
 
        if (rows <= 0 || cols <= 0 || repetitions <= 0) {
            std::cerr
                << "rows, cols, and repetitions must be positive\n";
            return EXIT_FAILURE;
        }
 
        std::vector<float> input =
            make_test_input(rows, cols);
 
        bool passed = run_correctness_test(
            input,
            rows,
            cols,
            warmups,
            repetitions
        );
 
        return passed ? EXIT_SUCCESS : EXIT_FAILURE;
    }
 
    int rows = 10;
    int repetitions = 100;
    int warmups = 10;

    std::vector<int> widths = {
        1, 3, 31, 32, 33,
        128, 256, 512, 1000, 1024,
        2048, 4096, 8192
    };
    
    bool all_passed = true;
    for(int cols : widths) {
        std::vector<float> input = make_test_input(rows, cols);

        bool passed = run_correctness_test(input, rows, cols, warmups, repetitions);

        all_passed = all_passed && passed;
    }

    int large_cols = 33;

    std::cout<< "Large positive logits\n";

    std::vector<float> large_input = make_large_logit_input(rows, large_cols);

    bool large_passed = run_correctness_test(large_input, rows, large_cols, warmups, repetitions);

    all_passed = all_passed && large_passed;

    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

