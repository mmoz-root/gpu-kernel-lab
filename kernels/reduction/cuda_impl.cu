#include <cstdint>
#include <cuda_runtime.h>
#include <system_error>

// naive sum kernel
__global__
void naive_sum(const float* x, float* output, int64_t n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float result = 0.0f;

        for(int64_t i = 0; i<n; i++){
            result += x[i];
        }

        output[0] = result;
    }
}
void launch_naive_sum(const float* x, float* output, int64_t n, cudaStream_t stream) {
    naive_sum<<<1, 1, 0, stream>>>(x, output, n);
}



// naive max kernel
__global__
void naive_max(const float* x, float* output, int64_t n) {
    if(blockIdx.x == 0 && threadIdx.x == 0) {
        float result = x[0];

        fot(int64_t i = 1; i<n; ++i) {
            if(x[i] > result) {
                result = x[i];
            }
        }
        output[0] = result;
    }
}
/*
    one thread ; only one thread works, so the GPU's parallel hardware is mostly unused
        ↓
    reads x[0]
    reads x[1]
    reads x[2]
    ...
    writes output[0]
*/
void launch_naive_max(
    const float* x,
    float* output,
    int64_t n,
    cudaStream_t stream
) {
    naive_max_kernel<<<1, 1, 0, stream>>>(
        x,
        output,
        n
    );
}





// interleaved shmem reduction kernel
/*
    calculates a partial result
    stores it in shmem
    coop w/ other threads to reduce 256 partial results to one

    global mem -> thread-local res -> shmem -> output
*/
__global__
void interleaved_shmem_sum(const float* x, float* output, int64_t n) {
    extern __shared__ float shared[];

    unsigned int tid = threadIdx.x;
    
    float local_sum = 0.0f;

    for(int64_t i = tid; i<n; i+=blockDim.x) {
        local_sum += x[i];
    }
    shared[tid] = local_sum;

    __syncthreads();

/*
    thread 0: x[0], x[4], x[8], ...
    thread 1: x[1], x[5], x[9], ...
    thread 2: x[2], x[6], x[10], ...
    thread 3: x[3], x[7], x[11], ...
*/

    for(unsigned int stride=1; stride < blockDim.x; stride*=2) {
        if(tid % (2*stride) == 0) {
            shared[tid] += shared[tid+stride];
        }
        __syncthreads();
    }
/*
    Initial:  [0] [1] [2] [3] [4] [5] [6] [7]

    stride 1: [0+1]   [2+3]   [4+5]   [6+7]
    stride 2: [0+1+2+3]       [4+5+6+7]
    stride 4: [0+1+2+3+4+5+6+7]

    stride 1: 0, 2, 4, 6
    stride 2: 0, 4
    stride 4: 0
*/

    if(tid == 0) {
        output[0] = shared[0];
    }
}
void launch_interleaved_sum(
    const float* x,
    float* output,
    int64_t n,
    cudaStream_t stream
) {
    constexpr int threads = 256;
    constexpr int shared_bytes =
        threads * sizeof(float);

    interleaved_sum_kernel<<<
        1,
        threads,
        shared_bytes,
        stream
    >>>(
        x,
        output,
        n
    );
}
/*
Global input
[x0][x1][x2][x3][x4][x5][x6][x7]
 │   │   │   │   │   │   │   │
 └─T0┘   └─T1┘   └─T2┘   └─T3┘
   │       │       │       │
   A       B       C       D       thread-local registers
   │       │       │       │
   ▼       ▼       ▼       ▼
 [ A ]   [ B ]   [ C ]   [ D ]    shared memory
    \     /           \     /
    [ A+B ]           [ C+D ]
           \           /
            [A+B+C+D]
                  │
                  ▼
              output[0]
*/



__global__
void interleaved_shmem_max(const float* x, float* output, int64_t n) {
    extern __shared__ float shared[];

    const unsigned int tid = threadIdx,x;

    float local_max = -CUDART_INF_F;

    for(int64_t i = tid; i<n; i+=blockDim.x) {
        if(x[i] > local_max) 
            local_max = x[i];
    }
    shared[tid] = local_max;

    __syncthreads();

    for(unsigned int stride =1; stride<blockDim.x; stride*=2) {
        if(tid % (2*stride) == 0) {
            const float candidate = shared[tid + stride];
            if(candidate > shared[tid]) {
                shared[tid] = candidate;
            }
        
        }
        __syncthreads();
    }

    if(tid == 0) {
        output[0] = shared[0];
    }
}
void launch_interleaved_max(
    const float* x,
    float* output,
    int64_t n,
    cudaStream_t stream
) {
    constexpr int threads = 256;
    constexpr int shared_bytes =
        threads * sizeof(float);

    interleaved_max_kernel<<<
        1,
        threads,
        shared_bytes,
        stream
    >>>(
        x,
        output,
        n
    );
}





// sequantial-addressing kernel
/* 
Interleaved: active threads are scattered
Sequential:  active threads are grouped together 

Interleaved warp: [work][skip][work][skip]...
Sequential warp:  [work][work][work][work]...

sequatnial addressing organizes 
the participating threads and memory accesses more efficiently
*/
__global__ void sequential_sum_kernel(
    const float* x,
    float* output,
    int64_t n
) {
    extern __shared__ float shared[];

    const unsigned int tid = threadIdx.x;

    float local_sum = 0.0f;

    for (int64_t i = tid; i < n; i += blockDim.x) {
        local_sum += x[i];
    }

    shared[tid] = local_sum;

    __syncthreads();

    for (
        unsigned int stride = blockDim.x / 2;
        stride > 0;
        stride /= 2
    ) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        output[0] = shared[0];
    }
}
void launch_sequential_sum(
    const float* x,
    float* output,
    int64_t n,
    cudaStream_t stream
) {
    constexpr int threads = 256;
    constexpr int shared_bytes =
        threads * sizeof(float);

    sequential_sum_kernel<<<
        1,
        threads,
        shared_bytes,
        stream
    >>>(
        x,
        output,
        n
    );
}



__global__
void sequantial_max(const float* x, float* output, int64_t n) {
    extern __shared__ float shared[];

    const unsigned int tid = threadIdx.x;

    float local_max = -CUDART_INF_F;

    for(int64_t i = tid; i<n; i+=blockDim.x) {
        if(x[i] > local_max) {
            local_max = x[i];
        }
    }
    shared[tid] = local_max;

    __syncthreads();

    for(unsigned int stride = blockDim.x / 2; stride > 0; stride/=2) {
        if(tid < stride) {
            const float candidate = shared[tid+stride];
            if(candidate > shared[tid]) {
                shared[tid] = candidate;
            }
        }
        __syncthreads();
    }
    if(tid == 0) {
        output[0] = shared[0];
}
void launch_sequantial_max(
    const float* x,
    float* output,
    int64_t n,
    cudaStream_t stream
) {
    constexpr int threads = 256;
    constexpr int shared_bytes =
        threads * sizeof(float);

    sequantial_max_kernel<<<
        1,
        threads,
        shared_bytes,
        stream
    >>>(x, output, n);
}




// warp-shuffle kernel
/* 
256 threads
    │
    ├── warp 0: lanes 0–31   → one result
    ├── warp 1: lanes 0–31   → one result
    ├── warp 2: lanes 0–31   → one result
    ├── ...
    └── warp 7: lanes 0–31   → one result
                                  │
                                  ▼
                     8 results in shared memory
                                  │
                                  ▼
                     first warp combines them
                                  │
                                  ▼
                               output
*/
// warp-level helpers
__device__ 
float warp_reduce_sum(float value) {
    for(int offset = 16; offset > 0; offset/=2) {
        value += __shfl_down_f(0xffffffff, value, offset);
    }
    return value;
    }
}
__device__ 
float warp_reduce_max(float value) {
    for(int offset = 16; offset > 0; offset/=2) {
        const other = __shfl_down_f(0xffffffff, value, offset);
        value = fmaxf(value, other);
    }
    return value;
}


__global__
void warp_shuffle_sum(const float* x, float* output, int64_t n) {
    __shared__ float warp_results[32];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % 32;
    const unsigned int warp_id = tid / 32;

    const unsigned int num_warps = blockDim.x /32;

    float local_sum = 0.0f;

    for(int64_t i = tid; i<n; i+=blockDim.x) {
        local_sum+=x[i];
    }

    //reduce each warp independently
    local_sum = warp_reduce_sum(local_sum);

    if(lane == 0) {
        warp_results[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = 0.0f;

        if (lane < num_warps) {
            block_sum = warp_results[lane];
        }

        block_sum = warp_reduce_sum(block_sum);

        if (lane == 0) {
            output[0] = block_sum;
        }
    }
}

// 256 threads / 32 threads per warp = 8 warp results

void launch_warp_shuffle_sum(
    const float* x,
    float* output,
    int64_t n,
    cudaStream_t stream
) {
    constexpr int threads = 256;

    warp_shuffle_sum_kernel<<<
        1,
        threads,
        0,
        stream
    >>>(x, output, n);
}



__global__
void warp_shuffle_max(const float* x, float* output, int64_ n) {
    __shared__ float warp_results[32];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % 32;
    const unsigned int warp_id = tid / 32;

    const unsigned int num_warps = blockDim.x / 32;

    float local_max = -CUDART_INF_F;

    for(int64_t i = tid; i<n; i+=blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }
    local_max = warp_reduce_max(local_max);

    if(lane == 0) {
        warp_results[warp_id] = local_max;
    }
    __syncthreads();

    if(warp_id == 0) {
        float block_max = -CUDART_INF_F;

        if(lane < num_warps) {
            block_max = warp_results[lane];
        }
        block_max = warp_reduce_max(block_max);

        if(lane == 0) {
            output[0] = block_max;
        }
    }
}
void launch_warp_shuffle_max(
    const float* x,
    float* output,
    int64_t n,
    cudaStream_t stream
) {
    constexpr int threads = 256;

    warp_shuffle_max_kernel<<<
        1,
        threads,
        0,
        stream
    >>>(x, output, n);
}



int main(){}