#include <cuda_runtime.h>

#include <iostream>

#include <cmath>
#include <math.h>



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
void launch_add_gridnstride(const float *x, const float *y, float *z, int n,
                            int threads_per_block) {
    cudaDeviceProp properties;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&properties, device);

    int blocks = properties.multiProcessorCount * 4;
    add_gridnstride <<<blocks, threads_per_block>>>(x, y, z, n);
}





//vectorized version using float4
__global__
void add_float4(const float* x, const float *y, float *z, int n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int vector_count / 4;

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

        // Handle N values that are not divisible by four
        int tail_start = vector_count * 4;
        int tail_size = n - tail_start;

        if (index < tail_size) {
            int tail_index = tail_start + index;
            z[tail_index] = x[tail_index] + y[tail_index];
        }
    }
}
void launch_add_float4(const float *x, const float *y, float *z, int n,
                        int threads_per_block) {

    int vector_count = n/4;
    int work_items = vector_count > 0 ? vector_count : 1;
    int blocks = (work_items + threads_per_block - 1) / threads_per_block;
    add_float4 <<<blocks, threads_per_block>>>(x, y, z, n);
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



int main(int argc, char** argv) {

    int N = argc > 1 ? std::atoi(argv[1]) : (1 << 20);
    int threads_per_block = argc > 2 ? std::atoi(argv[2]) : 256;

    if (N <= 0 || threads_per_block <= 0) {
        std::cerr << "N and block size must be pos\n";
        return 1;
    }

    float *x;
    float *y;
    float *z;

    std::size_t bytes = static_cast<std::size_t>(N) * sizeof(float);

    cudaMallocManaged(&x, bytes);
    cudaMallocManaged(&y, bytes);
    cudaMallocManaged(&z, bytes);

    for (int i = 0; i<N; ++i) {
        x[i] = 1.0f;
        y[i] = 2.0f;
        z[i] = 0.0f;
    }

    std::cout
            << "N: " << N
            << ", block size: " << threads_per_block
            << "\n";

    
    // Basic simple kernel
    launch_add_basic(x, y, z, N, threads_per_block);
    cudaDeviceSynchronize();
    std::cout
            << "Basic kernel max error: "
            << max_error(z, N)
            << "\n";

    // grid - stride loop kernel
    launch_add_gridnstride(x, y, z, N, threads_per_block);
    cudaDeviceSynchronize();
    std::cout
            << "Grid-stride kernel max error: "
            << max_error(z, N)
            << "\n";

    // float4 kernel
    launch_add_float4(x, y, z, N, threads_per_block);
    cudaDeviceSynchronize();
    std::cout
            << "float4 kernel max error: "
            << max_error(z, N)
            << "\n";

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    return 0;
}