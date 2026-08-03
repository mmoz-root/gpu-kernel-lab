#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>


constexpr int TILE_SIZE = 32;

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


__global__
void matpose_tiled(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int input_col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int input_row = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (input_col < cols && input_row < rows) {
        tile[threadIdx.y][threadIdx.x] = input[input_row * cols + input_col];
    }

    __syncthreads();

    int output_col = blockIdx.y * TILE_SIZE + threadIdx.x;
    int output_row = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (output_col < rows && output_row < cols) {
        output[output_row * rows + output_col] = tile[threadIdx.x][threadIdx.y];
    }
}

void launch_matpose_tiled(const float* input, float* output, int rows, int cols) {
    dim3 block(TILE_SIZE, TILE_SIZE);

    dim3 grid(
        ((cols + TILE_SIZE - 1)/TILE_SIZE),
        ((rows + TILE_SIZE - 1)/TILE_SIZE)
    );

    matpose_tiled<<<grid, block>>>(input, output, rows, cols);
}


__global__
void matpose_tiled_padded(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE+1];

    int input_col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int input_row = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (input_col < cols && input_row < rows) {
        tile[threadIdx.y][threadIdx.x] = input[input_row * cols + input_col];
    }

    __syncthreads();

    int output_col = blockIdx.y * TILE_SIZE + threadIdx.x;
    int output_row = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (output_col < rows && output_row < cols) {
        output[output_row * rows + output_col] = tile[threadIdx.x][threadIdx.y];
    }
}

void launch_matpose_tiled_padded(const float* input, float* output, int rows, int cols) {
    dim3 block(TILE_SIZE, TILE_SIZE);

    dim3 grid(
        ((cols + TILE_SIZE - 1)/TILE_SIZE),
        ((rows + TILE_SIZE - 1)/TILE_SIZE)
    );

    matpose_tiled_padded<<<grid, block>>>(input, output, rows, cols);
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
    
    // launch_matpose_naive(d_input, d_output, rows, cols);
    // launch_matpose_tiled(d_input, d_output, rows, cols);
    launch_matpose_tiled_padded(d_input, d_output, rows, cols);
    
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