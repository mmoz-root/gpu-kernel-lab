#include <cuda_runtime.h>

#ifdef TORCH_EXTENSION_NAME
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

void check_cuda(
    cudaError_t error,
    const char* operation,
    const char* file,
    int line
) {
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



/*
thread-local sum(x²)
        ↓
shared-memory block reduction
        ↓
normalize and scale
*/
__global__
void basic_rms_norm(
    const float* input,
    const float* weight,
    float* output,
    int rows,
    int cols,
    float eps
) {
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if(row >= rows) return;

    int row_start = row * cols;

    // each thread accumulates part of sum(x^2)
    float local_sum_sqr = 0.0f;

    for(int col = tid; col < cols; col += blockDim.x) {
        float value = input[row_start + col];
        local_sum_sqr += value * value;
    }


    // combine thread local results thru shmem
    shared[tid] = local_sum_sqr;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride >0; stride /=2) {
        if(tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }


    // one thread calculates the rows norm factor
    if(tid == 0) {
        float mean_square = shared[0] / static_cast<float>(cols);

        shared[0] = rsqrtf(mean_square + eps);
    }
    __syncthreads();

    float reciprocal_rms = shared[0];


    // normalize, scale, and write the row
    for(int col = tid; col<cols; col += blockDim.x) {
        int index = row_start + col;

        output[index] = input[index]
            * reciprocal_rms
            * weight[col];
    }
}

void launch_basic_rms_norm(
    const float* input,
    const float* weight,
    float* output,
    int rows,
    int cols,
    float eps
) {
    constexpr int threads = 256;

    int blocks = rows;

    std::size_t shared_bytes = threads * sizeof(float);

    basic_rms_norm<<<blocks, threads, shared_bytes>>>(
        input, weight, output,
        rows, cols,
        eps
    );
}



__device__ 
float warp_reduce_sum(float val) {
    constexpr unsigned full_max = 0xffffffffu;

    for(int offs = 16; offs > 0; offs/=2) {
        val += __shfl_down_sync(full_max, val, offs);
    }
    return val;
}


__global__
void optimized_rms_norm(const float* input, const float* weight, float* output,
                        int rows, int cols, float eps) {
    extern __shared__ float warp_sums[];

    constexpr int warp_size = 32;

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if(row >= rows) return;

    int lane = tid % warp_size;
    int warp_id = tid / warp_size;
    int warp_count = blockDim.x / warp_size;

    int row_start = row * cols;
    
    float local_sum_sqr = 0.0f;

    for(int col = tid; col<cols; col+=blockDim.x){
        float val = input[row_start + col];
        local_sum_sqr += val*val;
    }

    float warp_sum = warp_reduce_sum(local_sum_sqr);

    if(lane == 0) warp_sums[warp_id] = warp_sum;

    __syncthreads();

    if (warp_id == 0) {
        float block_sum = lane < warp_count ? warp_sums[lane] : 0.0f;

        block_sum = warp_reduce_sum(block_sum);

        if(lane==0) {
            float mean_sqr = block_sum / static_cast<float>(cols);
            warp_sums[0] = rsqrtf(mean_sqr + eps);
        }
    }
    __syncthreads();

    float reciprocal_rms = warp_sums[0];

    for (int col = tid; col < cols; col += blockDim.x) {
        int index = row_start + col;

        output[index] =
            input[index]
            * reciprocal_rms
            * weight[col];
    }
}

void launch_optimized_rms_norm(
    const float* input, const float* weight, float* output,
    int rows, int cols, float eps
) {
    constexpr int threads = 256;
    constexpr int warp_size = 32;

    int blocks = rows;
    int warp_count = threads / warp_size;

    std::size_t shared_bytes = warp_count * sizeof(float);

    optimized_rms_norm<<<blocks, threads, shared_bytes>>>(
        input, weight, output,
        rows, cols, eps
    );
}

#ifdef TORCH_EXTENSION_NAME
void launch_optimized_rms_norm_stream(
    const float* input,
    const float* weight,
    float* output,
    int rows, int cols, float eps,
    cudaStream_t stream
) {
    constexpr int threads = 256;
    constexpr int warp_size = 32;

    int blocks = rows;
    int warp_count = threads / warp_size;

    std::size_t shared_bytes = warp_count * sizeof(float);

    optimized_rms_norm<<<blocks, threads, shared_bytes, stream>>>(
        input, weight, output,
        rows, cols, eps
    );
}

torch::Tensor rms_norm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    double eps
) {
    TORCH_CHECK(
        input.is_cuda(),
        "input must be a CUDA tensor"
    );

    TORCH_CHECK(
        weight.is_cuda(),
        "weight must be a CUDA tensor"
    );

    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32,
        "input must have dtype float32"
    );

    TORCH_CHECK(
        weight.scalar_type() == torch::kFloat32,
        "weight must have dtype float32"
    );

    TORCH_CHECK(
        input.dim() >= 1,
        "input must have at least one dimension"
    );

    TORCH_CHECK(
        input.size(-1) > 0,
        "the normalized dimension must be non-empty"
    );

    TORCH_CHECK(
        weight.dim() == 1
        && weight.size(0) == input.size(-1),
        "weight must have shape (input.shape[-1],)"
    );

    TORCH_CHECK(
        input.device() == weight.device(),
        "input and weight must be on the same CUDA device"
    );

    TORCH_CHECK(
        eps >= 0.0,
        "eps must be non-negative"
    );

    torch::Tensor input_contiguous = input.contiguous();
    torch::Tensor weight_contiguous = weight.contiguous();

    torch::Tensor output = torch::empty_like(input_contiguous);

    int cols = static_cast<int>(input_contiguous.size(-1));
    int rows = static_cast<int>(input_contiguous.numel() / input_contiguous.size(-1));

    c10::cuda::CUDAGuard device_guard(
        input_contiguous.device()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(
        input_contiguous.get_device()
    );

    launch_optimized_rms_norm_stream(
        input_contiguous.data_ptr<float>(),
        weight_contiguous.data_ptr<float>(),
        output.data_ptr<float>(),
        rows, cols, static_cast<float>(eps),
        stream
    );
    
    cudaError_t error = cudaGetLastError();

    TORCH_CHECK(
        error == cudaSuccess,
        "RMSNorm CUDA kernel launch failed: ",
        cudaGetErrorString(error)
    );

    return output;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    module.def(
        "forward",
        &rms_norm_cuda,
        "RMSNorm forward (CUDA)",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("eps") = 1e-6
    );
}
#endif

// host / reference implementation
void cpu_rms_norm(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    std::vector<float>& output,
    int rows,
    int cols,
    float eps
) {
    output.resize(rows * cols);

    for(int row = 0; row < rows; ++row) {
        int row_start = row*cols;

        double sum_sqr = 0.0;

        for(int col = 0; col < cols; ++col) {
            double value = input[row_start + col];

            sum_sqr += value*value;
        }

        double mean_sqr = sum_sqr / static_cast<double>(cols);

        double reciprocal_rms = 1.0 / std::sqrt(mean_sqr + eps);

        for(int col = 0; col<cols; ++col) {
            int index = row_start + col;

            output[index] = static_cast<float>(
                input[index] * reciprocal_rms * weight[col]
            );
        }
    }
}


std::vector<float> make_test_input (int rows, int cols) {
    std::size_t element_count = static_cast<std::size_t>(rows)*cols;

    std::vector<float> input(element_count);

    for(std::size_t i = 0; i<element_count; ++i) {
        input[i] = static_cast<float>(
            static_cast<int>(i%17)-8) * 0.25f;
    }

    return input;
}

std::vector<float> make_test_weight(int cols) {
    std::vector<float> weight(cols);

    for(int col = 0; col<cols; ++col) {
        weight[col] = 0.5f + static_cast<float>(col%7) * 0.125f;
    }

    return weight;
}

float max_abs_error(
    const std::vector<float>& actual,
    const std::vector<float>& expected
) {
    float max_error = 0.0f;

    for(std::size_t i = 0; i<actual.size(); ++i) {
        max_error = std::max(max_error, std::abs(actual[i] - expected[i]));
    }

    return max_error;
}


using RmsNormLauncher = void(*) (
    const float*,
    const float*,
    float*,
    int,
    int,
    float
);


bool check_implementation(
    const std::string& name,
    RmsNormLauncher launcher,
    const float* d_input,
    const float* d_weight,
    float* d_output,
    std::vector<float>& actual,
    const std::vector<float>& expected,
    int rows,
    int cols,
    float eps
) {
    launcher(
        d_input, d_weight, d_output,
        rows, cols, eps
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::size_t output_bytes = actual.size() * sizeof(float);

    CUDA_CHECK(cudaMemcpy(
        actual.data(),
        d_output,
        output_bytes,
        cudaMemcpyDeviceToHost
    ));

    float error = max_abs_error(actual, expected);
    bool passed = error < 1e-5f;

    std::cout
    << name
    << " max error: "
    << error
    << " — "
    << (passed ? "PASSED" : "FAILED")
    << "\n";

    return passed;
}

float benchmark_implementation(
    RmsNormLauncher launcher,
    const float* d_input,
    const float* d_weight,
    float* d_output,
    int rows, int cols, float eps,
    int repetitions
) {
    constexpr int warmups = 10;

    for(int i = 0; i<warmups; ++i) {
        launcher(
            d_input, d_weight, d_output,
            rows, cols, eps
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for(int i = 0; i < repetitions; ++i) {
        launcher(
            d_input, d_weight, d_output,
            rows, cols, eps
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float total_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / repetitions;
}

bool run_correctness_test(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    int rows,
    int cols,
    float eps,
    int repetitions = 0
) {

    std::cout << "Correctness test: rows="
        << rows
        << ", cols="
        << cols
        << "\n";

    std::vector<float> expected;

    cpu_rms_norm(
        input, weight, expected,
        rows, cols, eps
    );

    std::vector<float> actual(input.size());

    std::size_t matrix_bytes = input.size() * sizeof(float);

    std::size_t weight_bytes = weight.size() * sizeof(float);

    float* d_input = nullptr;
    float* d_weight = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), matrix_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_weight), weight_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), matrix_bytes));

    CUDA_CHECK(cudaMemcpy(d_input, input.data(), matrix_bytes,
                            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, weight.data(), weight_bytes,
                            cudaMemcpyHostToDevice));

    bool passed = check_implementation(
        "basic rmsnorm",
        launch_basic_rms_norm,
        d_input, d_weight, d_output,
        actual, expected,
        rows, cols, eps
    );
    
    bool optimized_passed = check_implementation(
        "Optimized RMSNorm",
        launch_optimized_rms_norm,
        d_input, d_weight, d_output,
        actual, expected,
        rows, cols, eps
    );

    passed = passed && optimized_passed;

    if(repetitions > 0) {
        float basic_ms = benchmark_implementation(
            launch_basic_rms_norm,
            d_input, d_weight, d_output,
            rows, cols, eps,
            repetitions
        );

        float optimized_ms = benchmark_implementation(
            launch_optimized_rms_norm,
            d_input, d_weight, d_output,
            rows, cols, eps,
            repetitions
        );

        float speedup = basic_ms / optimized_ms;

        std::cout
            << "\nBenchmark\n"
            << "Basic RMSNorm:     "
            << basic_ms
            << " ms\n"
            << "Optimized RMSNorm: "
            << optimized_ms
            << " ms\n"
            << "Speedup:           "
            << speedup
            << "x\n";
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_output));

    return passed;
}



#ifndef TORCH_EXTENSION_NAME
int main(int argc, char* argv[]) {
    constexpr float eps = 1e-6f;

    if(argc != 1 && argc != 4) {
        std::cerr
            << "Usage: "
            << argv[0]
            << " [rows cols repetitions]\n";

            return EXIT_FAILURE;
    }

    if(argc == 4) {
        int rows = std::stoi(argv[1]);
        int cols = std::stoi(argv[2]);
        int repetitions = std::stoi(argv[3]);

        if(
            rows <= 0 || cols <= 0 || repetitions <= 0
        ) {
            std::cerr << "rows, cols, and repetitions "
                << "must be positive\n";

            return EXIT_FAILURE;
        }

        (void)repetitions;

        std::vector<float>input = make_test_input(rows, cols);
        std::vector<float>weight = make_test_weight(cols);

        bool passed = run_correctness_test(input, weight, rows, cols, eps, repetitions);

        return passed ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    int rows = 10;

    std::vector<int> widths = {
        1,
        3,
        31,
        32,
        33,
        127,
        128,
        129,
        256,
        511,
        512,
        1000,
        1024,
        2048,
        4096,
        8192,
    };

    bool all_passed = true;

    for (int cols : widths) {
        std::vector<float> input =
            make_test_input(rows, cols);

        std::vector<float> weight =
            make_test_weight(cols);

        bool passed = run_correctness_test(
            input,
            weight,
            rows,
            cols,
            eps
        );

        all_passed = all_passed && passed;
    }

    return all_passed
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
#endif