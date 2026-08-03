#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>

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

int main() {
    int rows = 2;
    int cols = 3;

    std::vector<float> input = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    std::vector<float> output(rows * cols);

    std::size_t bytes = static_cast<std::size_t>(rows) * cols * sizeof(float);

    float* d_input;
    float* d_output;

    cudaMalloc(reinterpret_cast<void**>(&d_input), bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_output), bytes);

    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);
    

    launch_matpose_naive(d_input, d_output, rows, cols);
    
    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    std::vector<float> expected = {
        1.0f, 4.0f,
        2.0f, 5.0f,
        3.0f, 6.0f
    };

    assert(output == expected);
    std::cout << "Test passed" << std::endl;
    return 0;
}