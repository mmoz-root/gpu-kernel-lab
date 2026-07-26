#include <cuda_runtime.h>

#include <iostream>

#include <cmath>
#include <cstdlib>

#include <algorithm>

#include <cuda_fp16.h>

__global__
void add_basic_half(
    const __half* x,
    const __half* y,
    __half* z,
    int n)
{
    int index =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        z[index] = __hadd(
            x[index],
            y[index]
        );
    }
}


void launch_add_basic_half(
    const __half* x,
    const __half* y,
    __half* z,
    int n,
    int threads_per_block)
{
    int blocks =
        (n + threads_per_block - 1)
        / threads_per_block;

    add_basic_half<<<blocks, threads_per_block>>>(
        x,
        y,
        z,
        n
    );
}


// basic simple kernel func to add elements of two vectors
__global__ 
void add_basic(const float *x, const float *y, float *z, int n) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < n) {
        z[index] = x[index] + y[index];

    }
}
void launch_add_basic(const float *x, const float *y, float *z, int n,
                      int threads_per_block) {
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    add_basic<<<blocks, threads_per_block>>>(x, y, z, n);
}





// the first parameter of the execution configuration specifies the number of thread blocks. 
// Together, the blocks of parallel threads make up what is known as the grid
__global__
void add_gridnstride(const float* x, const float *y, float *z, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=idx; i<n; i+=stride){
        z[i] = x[i] + y[i];
    }
}
void launch_add_gridnstride(
    const float* x,
    const float* y,
    float* z,
    int n,
    int threads_per_block)
{
    // Query the number of SMs only once.
    static int multiprocessor_count = 0;

    if (multiprocessor_count == 0) {
        int device;
        cudaDeviceProp properties;

        cudaGetDevice(&device);

        cudaGetDeviceProperties(
            &properties,
            device
        );

        multiprocessor_count =
            properties.multiProcessorCount;
    }

    int required_blocks =
        (n + threads_per_block - 1)
        / threads_per_block;

    int maximum_blocks =
        multiprocessor_count * 4;

    // Avoid launching many empty blocks for small arrays.
    int blocks = std::min(
        required_blocks,
        maximum_blocks
    );

    add_gridnstride<<<blocks, threads_per_block>>>(
        x,
        y,
        z,
        n
    );
}





//vectorized version using float4
__global__
void add_float4(const float* x, const float *y, float *z, int n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int vector_count = n / 4;

    if (index < vector_count) {
        const float4* x4 = reinterpret_cast<const float4*>(x);
        const float4* y4 = reinterpret_cast<const float4*>(y);
        float4* z4 = reinterpret_cast<float4*>(z);

        float4 xv = x4[index];
        float4 yv = y4[index];

        z4[index] = make_float4(
            xv.x + yv.x,
            xv.y + yv.y,
            xv.z + yv.z,
            xv.w + yv.w
        );
    }

     // Handle N values that are not divisible by four
     int tail_start = vector_count * 4;
     int tail_size = n - tail_start;

     if (index < tail_size) {
         int tail_index = tail_start + index;
         z[tail_index] = x[tail_index] + y[tail_index];
     }

}
void launch_add_float4(const float *x, const float *y, float *z, int n,
                        int threads_per_block) {

    int vector_count = n/4;
    int work_items = vector_count > 0 ? vector_count : 1;
    int blocks = (work_items + threads_per_block - 1) / threads_per_block;
    add_float4 <<<blocks, threads_per_block>>>(x, y, z, n);
}





constexpr int ACCESS_STRIDE = 4;
// Scalar vector addition with strided memory access.
__global__
void add_strided(
    const float* x,
    const float* y,
    float* z,
    int n,
    int access_stride)
{
    int logical_index =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (logical_index < n) {
        int memory_index =
            logical_index * access_stride;

        z[memory_index] =
            x[memory_index] + y[memory_index];
    }
}


void launch_add_strided(
    const float* x,
    const float* y,
    float* z,
    int n,
    int threads_per_block)
{
    int blocks =
        (n + threads_per_block - 1)
        / threads_per_block;

    add_strided<<<blocks, threads_per_block>>>(
        x,
        y,
        z,
        n,
        ACCESS_STRIDE
    );
}



constexpr int ELEMENTS_PER_THREAD = 4;
__global__
void add_multiple(
    const float* x,
    const float* y,
    float* z,
    int n)
{
    int block_start =
        blockIdx.x
        * blockDim.x
        * ELEMENTS_PER_THREAD;

    int thread_offset = threadIdx.x;

    #pragma unroll
    for (
        int item = 0;
        item < ELEMENTS_PER_THREAD;
        ++item
    ) {
        int index =
            block_start
            + thread_offset
            + item * blockDim.x;

        if (index < n) {
            z[index] = x[index] + y[index];
        }
    }
}


void launch_add_multiple(
    const float* x,
    const float* y,
    float* z,
    int n,
    int threads_per_block)
{
    int elements_per_block =
        threads_per_block
        * ELEMENTS_PER_THREAD;

    int blocks =
        (n + elements_per_block - 1)
        / elements_per_block;

    add_multiple<<<blocks, threads_per_block>>>(
        x,
        y,
        z,
        n
    );
}



float max_error(const float *result, int n) {
    float maxError = 0.0f;

    for (int i = 0; i < n; ++i) {
        maxError = std::fmax(
            maxError,
            std::fabs(result[i] - 3.0f)
        );
    }

    return maxError;
}
float max_error_strided(
    const float* result,
    int n)
{
    float maxError = 0.0f;

    for (int i = 0; i < n; ++i) {
        int memory_index =
            i * ACCESS_STRIDE;

        maxError = std::fmax(
            maxError,
            std::fabs(
                result[memory_index] - 3.0f
            )
        );
    }

    return maxError;
}
float max_error_half(
    const __half* result,
    int n)
{
    float maxError = 0.0f;

    for (int i = 0; i < n; ++i) {
        float value =
            __half2float(result[i]);

        maxError = std::fmax(
            maxError,
            std::fabs(value - 3.0f)
        );
    }

    return maxError;
}

template <typename T>
using LaunchFunction = void (*)(
    const T*,
    const T*,
    T*,
    int,
    int
);

template <typename T>
float benchmark_kernel(
    LaunchFunction<T> launch,
    const T* x,
    const T* y,
    T* z,
    int n,
    int threads_per_block,
    int repetitions)
{
    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm up the GPU and move Unified Memory to it.
    for (int i = 0; i < 10; ++i) {
        launch(
            x,
            y,
            z,
            n,
            threads_per_block
        );
    }

    cudaDeviceSynchronize();

    // Begin timing.
    cudaEventRecord(start);

    for (int i = 0; i < repetitions; ++i) {
        launch(
            x,
            y,
            z,
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


double calculate_bandwidth(
    int n,
    float average_ms,
    std::size_t bytes_per_element = sizeof(float))
{
    // Read x + read y + write z.
    double bytes =
        3.0
        * static_cast<double>(n)
        * static_cast<double>(bytes_per_element);

    return bytes / (average_ms * 1.0e6);
}

int main(int argc, char** argv) {

    int N = argc > 1 ? std::atoi(argv[1]) : (1 << 20);
    int threads_per_block = argc > 2 ? std::atoi(argv[2]) : 256;

    if (N <= 0 ||
        threads_per_block <= 0 ||
        threads_per_block > 1024) {
            std::cerr
            << "N must be positive and block size must be between 1 and 1024\n";
        return 1;
    }

    float *x;
    float *y;
    float *z;

    __half *x_half;
    __half *y_half;
    __half *z_half;

    std::size_t half_bytes =
    static_cast<std::size_t>(N)
    * sizeof(__half);

    cudaMallocManaged(&x_half, half_bytes);
    cudaMallocManaged(&y_half, half_bytes);
    cudaMallocManaged(&z_half, half_bytes);

    // std::size_t bytes = static_cast<std::size_t>(N) * sizeof(float);
    std::size_t allocated_elements =
    (
        static_cast<std::size_t>(N) - 1
    ) * ACCESS_STRIDE + 1;

    std::size_t bytes =
    allocated_elements * sizeof(float);

    cudaMallocManaged(&x, bytes);
    cudaMallocManaged(&y, bytes);
    cudaMallocManaged(&z, bytes);

    /*for (int i = 0; i<N; ++i) {
        x[i] = 1.0f;
        y[i] = 2.0f;
        z[i] = 0.0f;
    }*/
    for (
        std::size_t i = 0;
        i < allocated_elements;
        ++i
    ) {
        x[i] = 1.0f;
        y[i] = 2.0f;
        z[i] = 0.0f;
    }

    for (int i = 0; i < N; ++i) {
        x_half[i] = __float2half(1.0f);
        y_half[i] = __float2half(2.0f);
        z_half[i] = __float2half(0.0f);
    }

    std::cout
            << "N: " << N
            << ", block size: " << threads_per_block
            << "\n";

    
            int repetitions =
            N < (1 << 20) ? 1000 : 100;
        
        std::cout
            << "Measured repetitions: "
            << repetitions
            << "\n";
        
        
        // Basic kernel
        float basic_ms = benchmark_kernel(
            launch_add_basic,
            x,
            y,
            z,
            N,
            threads_per_block,
            repetitions
        );
        
        std::cout
            << "Basic kernel:\n"
            << "  Average time: "
            << basic_ms
            << " ms\n"
            << "  Effective bandwidth: "
            << calculate_bandwidth(N, basic_ms)
            << " GB/s\n"
            << "  Max error: "
            << max_error(z, N)
            << "\n";
            
        // Basic FP16 kernel
        float half_ms = benchmark_kernel(
            launch_add_basic_half,
            x_half,
            y_half,
            z_half,
            N,
            threads_per_block,
            repetitions
            );
            
        std::cout
            << "Basic FP16 kernel:\n"
            << "  Average time: "
            << half_ms
            << " ms\n"
            << "  Effective bandwidth: "
            << calculate_bandwidth(
                N,
                half_ms,
                sizeof(__half)
            )
            << " GB/s\n"
            << "  Max error: "
            << max_error_half(z_half, N)
            << "\n";

        // Strided-access kernel
        float strided_ms = benchmark_kernel(
            launch_add_strided,
            x,
            y,
            z,
            N,
            threads_per_block,
            repetitions
        );

        std::cout
            << "Strided-access kernel:\n"
            << "  Access stride: "
            << ACCESS_STRIDE
            << "\n"
            << "  Average time: "
            << strided_ms
            << " ms\n"
            << "  Effective bandwidth: "
            << calculate_bandwidth(N, strided_ms)
            << " GB/s\n"
            << "  Max error: "
            << max_error_strided(z, N)
            << "\n";

            float multiple_ms = benchmark_kernel(
                launch_add_multiple,
                x,
                y,
                z,
                N,
                threads_per_block,
                repetitions
            );
            
            std::cout
                << "Multiple-elements kernel:\n"
                << "  Elements per thread: "
                << ELEMENTS_PER_THREAD
                << "\n"
                << "  Average time: "
                << multiple_ms
                << " ms\n"
                << "  Effective bandwidth: "
                << calculate_bandwidth(N, multiple_ms)
                << " GB/s\n"
                << "  Max error: "
                << max_error(z, N)
                << "\n";
        // Grid-stride kernel
        float grid_ms = benchmark_kernel(
            launch_add_gridnstride,
            x,
            y,
            z,
            N,
            threads_per_block,
            repetitions
        );
        
        std::cout
            << "Grid-stride kernel:\n"
            << "  Average time: "
            << grid_ms
            << " ms\n"
            << "  Effective bandwidth: "
            << calculate_bandwidth(N, grid_ms)
            << " GB/s\n"
            << "  Max error: "
            << max_error(z, N)
            << "\n";
        
        
        // float4 kernel
        float float4_ms = benchmark_kernel(
            launch_add_float4,
            x,
            y,
            z,
            N,
            threads_per_block,
            repetitions
        );
        
        std::cout
            << "float4 kernel:\n"
            << "  Average time: "
            << float4_ms
            << " ms\n"
            << "  Effective bandwidth: "
            << calculate_bandwidth(N, float4_ms)
            << " GB/s\n"
            << "  Max error: "
            << max_error(z, N)
            << "\n";

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    cudaFree(x_half);
    cudaFree(y_half);
    cudaFree(z_half);

    return 0;
}