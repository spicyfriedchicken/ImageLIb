#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./grayscale_cuda input.jpg" << std::endl;
        return 1;
    }

    int width , height, channels;
    unsigned_char* h_input = stbi_load(argv[1], &width, &height, &channels, 3);
    if (!h_input) {
        std::cerr << "Error! Failed to load image" << std::endl;
        return 1;
    }

    size_t rgbSize = width * height * 3; // size of our current image when flattened!
    size_t graySize = width * height * 1; // output size (only 1 channel per pixel)

    unsigned char *d_input, *d_output; // allocate input and output pointer
    cudaMalloc(&d_input, rgbSize);
    cudaMalloc(&d_output, graySize);

    cudaMemcpy(d_input, h_input, rgbSize, cudaMemcpyHostToDevice);

    dim3 blockDim(16,16);
    dim3 gridDim((width + 15) / 16, (height + 15) / 16); // roof if not divisible by 16, add an extra grid for remainder

    rgb_to_grayscale<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    unsigned char* h_output = new unsigned char[graySize];
    cudaMemcpy(h_output, d_output, graySize, cudaMemcpyDeviceToHost);
    


    stbi_write_png("output.png", width, height, 1, h_output, width);

    stbi_image_free(h_input);
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;    
}
