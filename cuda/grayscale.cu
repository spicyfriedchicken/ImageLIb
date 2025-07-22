#include <iostream>

__global__ void rgb_to_grayscale (unsigned char* d_input, unsigned char* d_output, int width, int height, int channels) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y >= height || x >= width) return;

    int idx = (y * width + x) * channels;

    d_output[y * width + x] = static_cast<unsigned char>(0.299f * d_input[idx] +
                                                         0.587f * d_input[idx + 1] +
                                                         0.114f * d_input[idx + 2]);

}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./grayscale_cuda input.jpg" << std::endl;
        return 1;
    }

    int width, height, channels;
    unsigned char* h_input = stbi_load(argv[1], &width, &height, &channels, 3);
    if (!h_input) {
        std::cerr << "stbi has encountered an error with your image." << std::endl;
        return 1;
    }

    size_t rgbSize = width * height * channels; // get total RGBSize for gpu malloc
    size_t bwSize = width * height * 1; // get b/w size for gpu malloc

    unsigned char *d_input, *d_output;

    cudaMalloc(&d_input, rgbSize);
    cudaMalloc(&d_output, bwSize);

    cudaMemcpy(d_input, h_input, rgbSize, cudaMemcpyHostToDevice);

    dim3 blockDim(16,16);
    dim3 gridDim((width + 15) / 16, (height + 15) / 16);

    rgb_to_grayscale<<gridDim, blockDim>>>(d_input, d_output, width, height, channels);

    cudaDeviceSynchronize();
    unsigned char* h_output = new unsigned char[bwSize];
    cudaMemcpy(h_output, d_output, bwSize, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    stbi_image_free(h_input);
    delete[] h_output;

    return 0;
}