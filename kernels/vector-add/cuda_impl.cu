#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

// simple kernel func to add elements of two vectors
__global__ 
void add_basic(const float *a, const float *b, float *c, int n) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < n) {
        c[index] = a[index] + b[index];

    }
}
void launch_add_basic(const float *a, const float *b, float *c, int n) {
    constexpr int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    add_basic <<<blocks, threads_per_block>>>(a, b, c, n);
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
void launch_add_gridnstride(const float *x, const float *y, float *z, int n) {
    constexpr int threads_per_block = 256;
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
        const float4* a4 = reinterpret_cast<const float4*>(a);
        const float4* b4 = reinterpret_cast<const float4*>(b);
        float4* c4 = reinterpret_cast<float4*>(c);

        float4 av = a4[index];
        float4 bv = b4[index];

        c4[index] = make_float4(
            av.x + bv.x,
            av.y + bv.y,
            av.z + bv.z,
            av.w + bv.w
        );
        int tail_start = vector_count * 4;
        int tail_size = n - tail_start;

        if (index < tail_size) {
            int tail_index = tail_start + index;
            c[tail_index] = a[tail_index] + b[tail_index];
        }
    }
}
void launch_add_float4(const float *x, const float *y, float *z, int n) {
    constexpr int threads_per_block = 256;
    int vector_count = n/4;
    int work_items = vector_count > 0 ? vector_count : 1;
    int blocks = (work_items + threads_per_block - 1) / threads_per_block;
    add_float4 <<<blocks, threads_per_block>>>(x, y, z, n);
}



int main() {}