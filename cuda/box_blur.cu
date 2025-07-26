//
// Created by Oscar Abreu on 7/23/25.
//

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


__global__ void box_blur_cuda_shared(unsigned char* d_input, unsigned char* d_output,
                                     int width, int height, int channels, int radius) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;

    const int shared_width = blockDim.x + 2 * radius;
    const int shared_height = blockDim.y + 2 * radius;

    extern __shared__ unsigned char tile[];

    unsigned char* tile_r = tile;
    unsigned char* tile_g = &tile_r[shared_width * shared_height];
    unsigned char* tile_b = &tile_g[shared_width * shared_height];

    for (int dy = ty; dy < shared_height; dy += blockDim.y) {
        for (int dx = tx; dx < shared_width; dx += blockDim.x) {
            int global_x = blockIdx.x * blockDim.x + dx - radius;
            int global_y = blockIdx.y * blockDim.y + dy - radius;
            int clamped_x = min(max(global_x, 0), width - 1);
            int clamped_y = min(max(global_y, 0), height - 1);
            int global_idx = (clamped_y * width + clamped_x) * channels;
            int shared_idx = dy * shared_width + dx;
            tile_r[shared_idx] = d_input[global_idx];
            tile_g[shared_idx] = d_input[global_idx + 1];
            tile_b[shared_idx] = d_input[global_idx + 2];
        }
    }

    __syncthreads();

    if (x >= width || y >= height) return;

    int rsum = 0, gsum = 0, bsum = 0;
    int count = 0;

    for (int ry = -radius; ry <= radius; ry++) {
        for (int rx = -radius; rx <= radius; rx++) {
            int sx = tx + rx + radius;
            int sy = ty + ry + radius;
            int shared_idx = sy * shared_width + sx;
            rsum += tile_r[shared_idx];
            gsum += tile_g[shared_idx];
            bsum += tile_b[shared_idx];
            count++;
        }
    }

    int out_idx = (y * width + x) * channels;
    d_output[out_idx]     = rsum / count;
    d_output[out_idx + 1] = gsum / count;
    d_output[out_idx + 2] = bsum / count;
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

    size_t sharedBytes = 3 * (16 + 2 * radius) * (16 + 2 * radius) * sizeof(unsigned char);
    box_blur_cuda_shared<<<gridDim, blockDim, sharedBytes>>>(d_input, d_output, width, height, channels, radius);

    /* box_blur_cuda<<<gridDim, blockDim>>>(d_input, d_output, width, height, channels, radius); */
    cudaDeviceSynchronize();
    unsigned char* h_output = new unsigned char[imageSize];

    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(&d_input);
    cudaFree(&d_output)
    stbi_image_free(h_input);
    delete[] h_output;

    return 0;
}

