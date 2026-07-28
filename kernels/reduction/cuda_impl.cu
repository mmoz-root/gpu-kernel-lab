
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>


constexpr int MAX_BLOCKS = 256;



// Naive reduction

__global__
void naive_sum(const float* x, float* output, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float result = 0.0f;

        for (int i = 0; i < n; ++i) {
            result += x[i];
        }

        output[0] = result;
    }
}


__global__
void naive_max(const float* x, float* output, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float result = -CUDART_INF_F;

        for (int i = 0; i < n; ++i) {
            result = fmaxf(
                result,
                x[i]
            );
        }

        output[0] = result;
    }
}


// These launchers use the same signature as the other versions
// partials and threads_per_block are unused here

void launch_naive_sum(const float* x, float* output, float* partials, int n, int threads_per_block) {
    naive_sum<<<1, 1>>>(x, output, n);
}


void launch_naive_max(const float* x, float* output, float* partials, int n, int threads_per_block) {
    naive_max<<<1, 1>>>(x, output, n);
}





// Interleaved shared-memory reduction

__global__
void interleaved_sum_stage(const float* x, float* partials, int n) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;

    int index =
        blockIdx.x * blockDim.x + threadIdx.x;

    int grid_stride =
        blockDim.x * gridDim.x;

    float local_sum = 0.0f;

    for (int i = index; i < n; i += grid_stride) {
        local_sum += x[i];
    }

    shared[tid] = local_sum;

    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            shared[tid] += shared[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        partials[blockIdx.x] = shared[0];
    }
}


__global__
void interleaved_max_stage(const float* x, float* partials, int n) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;

    int index =blockIdx.x * blockDim.x + threadIdx.x;

    int grid_stride = blockDim.x * gridDim.x;

    float local_max = -CUDART_INF_F;

    for (int i = index; i < n; i += grid_stride) {
        local_max = fmaxf(local_max, x[i]);
    }

    shared[tid] = local_max;

    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }

        __syncthreads();
    }

    if (tid == 0) {
        partials[blockIdx.x] = shared[0];
    }
}


void launch_interleaved_sum(const float* x, float* output, float* partials, int n, int threads_per_block) {
    int required_blocks = (n + threads_per_block - 1) / threads_per_block;

    int blocks = std::min(required_blocks, MAX_BLOCKS);

    int shared_bytes = threads_per_block * sizeof(float);

    // Stage 1: input → block partials
    interleaved_sum_stage<<<
        blocks,
        threads_per_block,
        shared_bytes
    >>>(x, partials, n);

    // Stage 2: block partials → final result
    interleaved_sum_stage<<<
        1,
        threads_per_block,
        shared_bytes
    >>>(partials, output, blocks);
}


void launch_interleaved_max(const float* x, float* output, float* partials, int n, int threads_per_block) {
    int required_blocks =(n + threads_per_block - 1)/ threads_per_block;

    int blocks = std::min(required_blocks, MAX_BLOCKS);

    int shared_bytes =threads_per_block * sizeof(float);

    interleaved_max_stage<<<
        blocks,
        threads_per_block,
        shared_bytes
    >>>(x, partials, n);

    interleaved_max_stage<<<
        1,
        threads_per_block,
        shared_bytes
    >>>(partials, output, blocks);
}



// Sequential-addressing reduction

__global__
void sequential_sum_stage(const float* x, float* partials, int n) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int grid_stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;

    for (int i = index; i < n; i += grid_stride) {
        local_sum += x[i];
    }

    shared[tid] = local_sum;

    __syncthreads();

    for (int stride = blockDim.x / 2;stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] +=
                shared[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        partials[blockIdx.x] = shared[0];
    }
}


__global__
void sequential_max_stage(const float* x, float* partials, int n) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int grid_stride = blockDim.x * gridDim.x;

    float local_max = -CUDART_INF_F;

    for (int i = index; i < n; i += grid_stride) {
        local_max = fmaxf(local_max, x[i]);
    }

    shared[tid] = local_max;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }

        __syncthreads();
    }

    if (tid == 0) {
        partials[blockIdx.x] = shared[0];
    }
}


void launch_sequential_sum(const float* x, float* output, float* partials, int n, int threads_per_block) {
    int required_blocks = (n + threads_per_block - 1) / threads_per_block;

    int blocks = std::min(required_blocks, MAX_BLOCKS);

    int shared_bytes = threads_per_block * sizeof(float);

    sequential_sum_stage<<<
        blocks,
        threads_per_block,
        shared_bytes
    >>>(x, partials, n);

    sequential_sum_stage<<<
        1,
        threads_per_block,
        shared_bytes
    >>>(partials, output, blocks);
}


void launch_sequential_max(const float* x, float* output, float* partials, int n, int threads_per_block) {
    int required_blocks =
        (n + threads_per_block - 1)
        / threads_per_block;

    int blocks = std::min(
        required_blocks,
        MAX_BLOCKS
    );

    int shared_bytes =
        threads_per_block
        * sizeof(float);

    sequential_max_stage<<<
        blocks,
        threads_per_block,
        shared_bytes
    >>>(x, partials, n);

    sequential_max_stage<<<
        1,
        threads_per_block,
        shared_bytes
    >>>(partials, output, blocks);
}





// FP16 input, FP32 accumulation

__global__ void sequential_sum_f16_f32_stage(
    const __half* input,
    float* partials,
    int n
) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    int grid_stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;

    while (index < n) {
        local_sum += __half2float(input[index]);
        index += grid_stride;
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        partials[blockIdx.x] = shared[0];
    }
}


void launch_sequential_sum_f16_f32(
    const __half* input,
    float* output,
    float* partials,
    int n,
    int threads
) {
    int blocks = std::min(
        (n + threads - 1) / threads,
        MAX_BLOCKS
    );

    sequential_sum_f16_f32_stage<<<
        blocks,
        threads,
        threads * sizeof(float)
    >>>(
        input,
        partials,
        n
    );

    // Partials are already float32, so reuse the existing
    // float32 sequential reduction for the second stage.
    sequential_sum_stage<<<
        1,
        threads,
        threads * sizeof(float)
    >>>(
        partials,
        output,
        blocks
    );
}


// FP16 input, FP16 accumulation

__global__ void sequential_sum_f16_f16_stage(
    const __half* input,
    __half* partials,
    int n
) {
    extern __shared__ __half half_shared[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    int grid_stride = blockDim.x * gridDim.x;

    __half local_sum = __float2half(0.0f);

    while (index < n) {
        local_sum = __hadd(
            local_sum,
            input[index]
        );

        index += grid_stride;
    }

    half_shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            half_shared[tid] = __hadd(
                half_shared[tid],
                half_shared[tid + stride]
            );
        }

        __syncthreads();
    }

    if (tid == 0) {
        partials[blockIdx.x] = half_shared[0];
    }
}


void launch_sequential_sum_f16_f16(
    const __half* input,
    __half* output,
    __half* partials,
    int n,
    int threads
) {
    int blocks = std::min(
        (n + threads - 1) / threads,
        MAX_BLOCKS
    );

    sequential_sum_f16_f16_stage<<<
        blocks,
        threads,
        threads * sizeof(__half)
    >>>(
        input,
        partials,
        n
    );

    sequential_sum_f16_f16_stage<<<
        1,
        threads,
        threads * sizeof(__half)
    >>>(
        partials,
        output,
        blocks
    );
}

// Warp-shuffle reduction

__device__
float warp_reduce_sum(float value)
{
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(
            0xffffffff,
            value,
            offset
        );
    }

    return value;
}


__device__
float warp_reduce_max(float value)
{
    for (int offset = 16;offset > 0; offset /= 2) {
        float other =
            __shfl_down_sync(
                0xffffffff,
                value,
                offset
            );

        value = fmaxf(
            value,
            other
        );
    }

    return value;
}


__global__
void warp_sum_stage(const float* x, float* partials, int n) {
    extern __shared__ float warp_results[];

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    int number_of_warps = blockDim.x / 32;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int grid_stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;

    for (int i = index; i < n; i += grid_stride) {
        local_sum += x[i];
    }

    local_sum = warp_reduce_sum(local_sum);

    if (lane == 0) {
        warp_results[warp_id] =local_sum;
    }

    __syncthreads();

    // The first warp reduces the per-warp results.
    if (warp_id == 0) {
        float block_sum = 0.0f;

        if (lane < number_of_warps) {
            block_sum = warp_results[lane];
        }

        block_sum =
            warp_reduce_sum(block_sum);

        if (lane == 0) {
            partials[blockIdx.x] = block_sum;
        }
    }
}


__global__
void warp_max_stage(const float* x, float* partials, int n) {
    extern __shared__ float warp_results[];

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    int number_of_warps = blockDim.x / 32;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int grid_stride = blockDim.x * gridDim.x;

    float local_max = -CUDART_INF_F;

    for (int i = index; i < n; i += grid_stride) {
        local_max = fmaxf(
            local_max,
            x[i]
        );
    }

    local_max = warp_reduce_max(local_max);

    if (lane == 0) {
        warp_results[warp_id] = local_max;
    }

    __syncthreads();

    if (warp_id == 0) {
        float block_max = -CUDART_INF_F;

        if (lane < number_of_warps) {
            block_max = warp_results[lane];
        }

        block_max = warp_reduce_max(block_max);

        if (lane == 0) {
            partials[blockIdx.x] = block_max;
        }
    }
}


void launch_warp_sum(const float* x, float* output, float* partials, int n, int threads_per_block) {
    int required_blocks =(n + threads_per_block - 1) / threads_per_block;

    int blocks = std::min(required_blocks, MAX_BLOCKS);

    int shared_bytes =(threads_per_block / 32) * sizeof(float);

    warp_sum_stage<<<
        blocks,
        threads_per_block,
        shared_bytes
    >>>(x, partials, n);

    warp_sum_stage<<<
        1,
        threads_per_block,
        shared_bytes
    >>>(partials, output, blocks);
}


void launch_warp_max(const float* x, float* output, float* partials, int n, int threads_per_block) {
    int required_blocks = (n + threads_per_block - 1) / threads_per_block;

    int blocks = std::min(
        required_blocks,
        MAX_BLOCKS
    );

    int shared_bytes = (threads_per_block / 32) * sizeof(float);

    warp_max_stage<<<
        blocks,
        threads_per_block,
        shared_bytes
    >>>(x, partials, n);

    warp_max_stage<<<
        1,
        threads_per_block,
        shared_bytes
    >>>(partials, output, blocks);
}



// CPU references

double cpu_sum(
    const float* x,
    int n)
{
    double result = 0.0;

    for (int i = 0; i < n; ++i) {
        result += x[i];
    }

    return result;
}


float cpu_max(
    const float* x,
    int n)
{
    float result = -INFINITY;

    for (int i = 0; i < n; ++i) {
        result = std::fmax(
            result,
            x[i]
        );
    }

    return result;
}

double cpu_half_sum_for_precision(
    const __half* input,
    int n
) {
    double result = 0.0;

    for (int i = 0; i < n; ++i) {
        result += static_cast<double>(
            __half2float(input[i])
        );
    }

    return result;
}


// Benchmarking

using ReductionLauncher = void (*)(
    const float*,
    float*,
    float*,
    int,
    int
);


float benchmark_reduction(
    ReductionLauncher launch,
    const float* x,
    float* output,
    float* partials,
    int n,
    int threads_per_block,
    int repetitions)
{
    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup also moves Unified Memory to the GPU.
    for (int i = 0; i < 10; ++i) {
        launch(
            x,
            output,
            partials,
            n,
            threads_per_block
        );
    }

    cudaDeviceSynchronize();

    cudaEventRecord(start);

    for (int i = 0; i < repetitions; ++i) {
        launch(
            x,
            output,
            partials,
            n,
            threads_per_block
        );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;

    cudaEventElapsedTime(
        &total_ms,
        start,
        stop
    );

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total_ms / repetitions;
}


double input_bandwidth(
    int n,
    float average_ms)
{
    double bytes =
        static_cast<double>(n)
        * sizeof(float);

    return bytes / (
        average_ms * 1.0e6
    );
}


void run_benchmark(
    const char* name,
    ReductionLauncher launch,
    const float* x,
    float* output,
    float* partials,
    int n,
    int threads_per_block,
    int repetitions,
    double reference)
{
    float average_ms =
        benchmark_reduction(
            launch,
            x,
            output,
            partials,
            n,
            threads_per_block,
            repetitions
        );

    double error = std::fabs(
        static_cast<double>(output[0])
        - reference
    );

    std::cout
        << name
        << ":\n"
        << "  Result: "
        << output[0]
        << "\n"
        << "  Absolute error: "
        << error
        << "\n"
        << "  Average time: "
        << average_ms
        << " ms\n"
        << "  Input bandwidth: "
        << input_bandwidth(
            n,
            average_ms
        )
        << " GB/s\n";
}



float benchmark_f32_accumulation(
    const float* input,
    float* output,
    float* partials,
    int n,
    int threads,
    int repetitions
) {
    for (int i = 0; i < 5; ++i) {
        launch_sequential_sum(
            input,
            output,
            partials,
            n,
            threads
        );
    }

    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < repetitions; ++i) {
        launch_sequential_sum(
            input,
            output,
            partials,
            n,
            threads
        );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms;
    cudaEventElapsedTime(&total_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total_ms / repetitions;
}


float benchmark_f16_f32_accumulation(
    const __half* input,
    float* output,
    float* partials,
    int n,
    int threads,
    int repetitions
) {
    for (int i = 0; i < 5; ++i) {
        launch_sequential_sum_f16_f32(
            input,
            output,
            partials,
            n,
            threads
        );
    }

    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < repetitions; ++i) {
        launch_sequential_sum_f16_f32(
            input,
            output,
            partials,
            n,
            threads
        );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms;
    cudaEventElapsedTime(&total_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total_ms / repetitions;
}


float benchmark_f16_f16_accumulation(
    const __half* input,
    __half* output,
    __half* partials,
    int n,
    int threads,
    int repetitions
) {
    for (int i = 0; i < 5; ++i) {
        launch_sequential_sum_f16_f16(
            input,
            output,
            partials,
            n,
            threads
        );
    }

    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);

    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < repetitions; ++i) {
        launch_sequential_sum_f16_f16(
            input,
            output,
            partials,
            n,
            threads
        );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms;
    cudaEventElapsedTime(&total_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total_ms / repetitions;
}




int main(
    int argc,
    char** argv)
{
    int n =
        argc > 1
        ? std::atoi(argv[1])
        : (1 << 20) + 1;

    int threads_per_block =
        argc > 2
        ? std::atoi(argv[2])
        : 256;

    if (
        n <= 0
        || threads_per_block < 32
        || threads_per_block > 1024
        || (threads_per_block
            & (threads_per_block - 1)) != 0
        || threads_per_block % 32 != 0
    ) {
        std::cerr
            << "N must be positive.\n"
            << "Block size must be a power of two, "
            << "divisible by 32, and between 32 and 1024.\n";

        return 1;
    }

    float* x;
    float* output;
    float* partials;

    __half* half_input;
    __half* half_output;
    __half* half_partials;

    std::size_t input_bytes =
        static_cast<std::size_t>(n)
        * sizeof(float);

    std::size_t partial_bytes =
        MAX_BLOCKS
        * sizeof(float);

    cudaMallocManaged(
        &x,
        input_bytes
    );

    cudaMallocManaged(
        &output,
        sizeof(float)
    );

    cudaMallocManaged(
        &partials,
        partial_bytes
    );

    cudaMallocManaged(
        &half_input,
        static_cast<size_t>(n) * sizeof(__half)
    );
    
    cudaMallocManaged(
        &half_output,
        sizeof(__half)
    );
    
    cudaMallocManaged(
        &half_partials,
        MAX_BLOCKS * sizeof(__half)
    );

    // Mixed positive and negative values.
    for (int i = 0; i < n; ++i) {
        x[i] = std::sin(
            static_cast<float>(i)
            * 0.001f
        );
    }
    for (int i = 0; i < n; ++i) {
        half_input[i] = __float2half(x[i]);
    }

    double expected_sum =
        cpu_sum(x, n);

    float expected_max =
        cpu_max(x, n);

    double expected_half_sum =
        cpu_half_sum_for_precision(
        half_input,
        n
    );

    std::cout
        << "N: "
        << n
        << "\n"
        << "Threads per block: "
        << threads_per_block
        << "\n"
        << "CPU sum: "
        << expected_sum
        << "\n"
        << "CPU max: "
        << expected_max
        << "\n\n";

    int repetitions =
        n < (1 << 20)
        ? 1000
        : 100;

    int naive_repetitions =
        std::min(
            repetitions,
            10
        );

    // Naive versions are intentionally much slower.
    run_benchmark(
        "Naive sum",
        launch_naive_sum,
        x,
        output,
        partials,
        n,
        threads_per_block,
        naive_repetitions,
        expected_sum
    );

    run_benchmark(
        "Naive maximum",
        launch_naive_max,
        x,
        output,
        partials,
        n,
        threads_per_block,
        naive_repetitions,
        expected_max
    );

    run_benchmark(
        "Interleaved sum",
        launch_interleaved_sum,
        x,
        output,
        partials,
        n,
        threads_per_block,
        repetitions,
        expected_sum
    );

    run_benchmark(
        "Interleaved maximum",
        launch_interleaved_max,
        x,
        output,
        partials,
        n,
        threads_per_block,
        repetitions,
        expected_max
    );

    run_benchmark(
        "Sequential sum",
        launch_sequential_sum,
        x,
        output,
        partials,
        n,
        threads_per_block,
        repetitions,
        expected_sum
    );

    run_benchmark(
        "Sequential maximum",
        launch_sequential_max,
        x,
        output,
        partials,
        n,
        threads_per_block,
        repetitions,
        expected_max
    );

    run_benchmark(
        "Warp-shuffle sum",
        launch_warp_sum,
        x,
        output,
        partials,
        n,
        threads_per_block,
        repetitions,
        expected_sum
    );

    run_benchmark(
        "Warp-shuffle maximum",
        launch_warp_max,
        x,
        output,
        partials,
        n,
        threads_per_block,
        repetitions,
        expected_max
    );
        // Accumulation precision experiment

        int precision_repetitions = 100;

        float f32_time =
            benchmark_f32_accumulation(
                x,
                output,
                partials,
                n,
                threads_per_block,
                precision_repetitions
            );
    
        float f32_result = output[0];
    
    
        float f16_f32_time =
            benchmark_f16_f32_accumulation(
                half_input,
                output,
                partials,
                n,
                threads_per_block,
                precision_repetitions
            );
    
        float f16_f32_result = output[0];
    
    
        float f16_f16_time =
            benchmark_f16_f16_accumulation(
                half_input,
                half_output,
                half_partials,
                n,
                threads_per_block,
                precision_repetitions
            );
    
        float f16_f16_result =
            __half2float(half_output[0]);
    
    
        double f32_bandwidth =
            (static_cast<double>(n) * sizeof(float))
            / (f32_time * 1.0e6);
    
        double f16_f32_bandwidth =
            (static_cast<double>(n) * sizeof(__half))
            / (f16_f32_time * 1.0e6);
    
        double f16_f16_bandwidth =
            (static_cast<double>(n) * sizeof(__half))
            / (f16_f16_time * 1.0e6);
    
    
        std::cout
            << "\nAccumulation precision experiment\n"
            << "Float32 CPU reference: "
            << expected_sum
            << "\n"
            << "Float16 CPU reference: "
            << expected_half_sum
            << "\n"
            << "Input quantization difference: "
            << std::fabs(
                   expected_sum
                   - expected_half_sum
               )
            << "\n";
    
    
        std::cout
            << "\nFloat32 input, float32 accumulation:\n"
            << "  Result: "
            << f32_result
            << "\n"
            << "  Absolute error: "
            << std::fabs(
                   static_cast<double>(f32_result)
                   - expected_sum
               )
            << "\n"
            << "  Average time: "
            << f32_time
            << " ms\n"
            << "  Input bandwidth: "
            << f32_bandwidth
            << " GB/s\n";
    
    
        std::cout
            << "\nFloat16 input, float32 accumulation:\n"
            << "  Result: "
            << f16_f32_result
            << "\n"
            << "  Absolute error: "
            << std::fabs(
                   static_cast<double>(f16_f32_result)
                   - expected_half_sum
               )
            << "\n"
            << "  Average time: "
            << f16_f32_time
            << " ms\n"
            << "  Input bandwidth: "
            << f16_f32_bandwidth
            << " GB/s\n";
    
    
        std::cout
            << "\nFloat16 input, float16 accumulation:\n"
            << "  Result: "
            << f16_f16_result
            << "\n"
            << "  Absolute error: "
            << std::fabs(
                   static_cast<double>(f16_f16_result)
                   - expected_half_sum
               )
            << "\n"
            << "  Average time: "
            << f16_f16_time
            << " ms\n"
            << "  Input bandwidth: "
            << f16_f16_bandwidth
            << " GB/s\n";
    
    
        // Minimal CUDA error check.
    
        cudaError_t error =
            cudaDeviceSynchronize();
    
        if (error != cudaSuccess) {
            std::cerr
                << "CUDA error: "
                << cudaGetErrorString(error)
                << "\n";
        }
    
    
        // Free every allocation exactly once.
    
        cudaFree(x);
        cudaFree(output);
        cudaFree(partials);
    
        cudaFree(half_input);
        cudaFree(half_output);
        cudaFree(half_partials);
    
        return 0;
}