#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

// simple kernel func to add elements of two vectors
__global__ 
void add(int n, float *sum, float *x, float *y) {
    for(int i=0; i<n; i++) {
        sum[i] = x[i] + y[i];
    }
}

__global__
void add_2(int n, float* x, float* y) {
    int idx = threadIdx.x;
    int stride = blockDim.x;
    for(int i = idx; i < n; i+=stride) {
        y[i] = x[i] + y[i];
    }
}

// the first parameter of the execution configuration specifies the number of thread blocks. 
// Together, the blocks of parallel threads make up what is known as the grid
__global__
void add_3(int n, float* x, float *y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=idx; i<n; i+=stride){
        y[i] = x[i] + y[i];
    }
}


int main() {
    int N = 1<<20;
    float *x, *y;

    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // Prefetch the x and y arrays to the GPU
    cudaMemPrefetchAsync(x, N*sizeof(float), 0, 0);
    cudaMemPrefetchAsync(y, N*sizeof(float), 0, 0);

    for(int i=0; i<N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    //blocks of threads that are a mult of 32 in size
    add <<<1, 1>>>(N, sum, x, y); // 1 parallel thread

    add_2 <<<1, 256>>>(N, x, y); // 256 parallel threads

    int blockSize = 256;
    int numBlocks = (N + blockSize-1) / blockSize;
    add_3 <<<numBlocks, blockSize>>>(N, x, y);


    // I need the CPU to wait until the kernel is done before it accesses the results 
    // (because CUDA kernel launches don’t block the calling CPU thread)
    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for(int i=0; i<N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "Max Error: " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);
    return 0;
}