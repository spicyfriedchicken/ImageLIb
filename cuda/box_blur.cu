//
// Created by Oscar Abreu on 7/23/25.
//

#ifndef BOX_BLUR_CUH
#define BOX_BLUR_CUH

__global__ void box_blur_cuda(unsigned char* d_input, unsigned char* d_output, size_t width, size_t height, size_t channels, int radius) {
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y >= height || x >= width) return;
    size_t rsum = 0, gsum = 0, bsum = 0;
    size_t count = 0;
    for (int ry = -radius; ry <= radius; ry++) {
        if (y + ry < 0 || y + ry >= height) continue;
        for (int rx = -radius; rx <= radius; rx++) {
            if (x + rx < 0 || x + rx >= width) continue;
            int idx = (((y + ry) * width) + (x + rx)) * channels;
            rsum += d_input[idx];
            gsum += d_input[idx + 1];
            bsum += d_input[idx + 2];
            ++count;
        }
    }
    size_t out_idx = (y * width + x) * channels;
    d_output[out_idx] = (rsum / count);
    d_output[out_idx + 1] = (gsum / count);
    d_output[out_idx + 2] = (bsum / count);

}


int main(int argc, char** argv) {
    if (argc < 3 || is_int(stoi(argv[2]))) {
        std::cerr << "Error, invald usage. \nUsage: ./box_blur input.jpg radius" << std::endl;
        return 1;
    }

    int height, width, channels, radius;
    unsigned char* h_input = stbi_image_load(argv[1], &height, &width, &channels, 3);
    if (!h_input) {
        std::cerr << "Error, STBI could not load your image at " << argv[1] << std::endl;
        return 1;
    }
    if (channels != 3 || argc != 2) {
        std::cerr << "Image not RGB!" << std::endl;
        return 1;
    }

    int imageSize = width * height * channels;
    unsigned char* d_input, *d_output;

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);

    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);

    dim3 blockDim(16,16)
    dim3 gridDim ((width + 15) / 16, (height + 15) / 16);

    box_blur_cuda<<<gridDim, blockDim>>>(d_input, d_output, width, height, channels, radius);
    cudaDeviceSynchronize();
    unsigned char* h_output = new unsigned char[imageSize];

    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(&d_input);
    cudaFree(&d_output)
    stbi_image_free(h_input);
    delete[] h_output;

    return 0;
}

#endif //BOX_BLUR_CUH
