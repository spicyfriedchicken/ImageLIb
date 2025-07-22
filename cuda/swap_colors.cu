#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


__global__ void swap_rgb_kernel(unsigned char* d_input, unsigned char* d_output, size_t width, size_t height) {
    int y = blockDim.y * blockIdx.y  + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (y >= height || x >= width) return;

    int idx = (y * width + x) * 3;

    d_output[idx] = d_input[idx + 2];
    d_output[idx + 1] = d_input[idx];
    d_output[idx + 2] = d_input[idx + 1];

    return;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: ./main input.jpg" << std::endl; return 1; }

    size_t width;
    size_t height;
    size_t channels;

    unsigned char* h_input = stbi_image_load(argv[1], &width, &height, &channels, 3);

    int rgb_size = width * height * channels;

    unsigned char* d_input, *d_output;

    cudaMalloc(&d_input, rgb_size);
    cudaMalloc(&d_output, rgb_size);

    cudaMemcpy(d_input, h_input, rgb_size, cudaMemcpyHostToDevice);

    dim3 blockDim(16,16);
    dim3 gridDim((width + 15) / 16, (height + 15) / 16);

    swap_rgb_kernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    unsigned char* h_output = new unsigned char[rgb_size];
    cudaMemcpy(h_output, d_output, rgb_size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_output;
    stbi_image_free(h_input);

    return 0;

}