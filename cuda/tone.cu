__global__ void ajust_brightness(uint8_t* d_input, uint8_t* d_output, int width, int height, int channels, int brightness) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y >= height || x >= width) return;
    int idx = (y * width + x) * channels;
    d_output[idx] = std::clamp(int(d_input[idx] * brightness), 0, 255);
    d_output[idx + 1] = std::clamp(int(d_input[idx + 1] * brightness), 0, 255);
    d_output[idx + 2] = std::clamp(int(d_input[idx + 2] * brightness), 0, 255);
}

__global__ void adjust_contrast(uint8_t* d_input, uint8_t* d_output, int width, int height, int channels, const float contrast_const) {
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (y >= height || x >= width) return;

    int idx = (y * width + x) * channels;

    d_output[idx] = std::clamp(static_cast<int>((float(d_input[idx]) - 128.0f) * contrast_const + 128.0f), 0, 255);
    d_output[idx + 1] = std::clamp(static_cast<int>((float(d_input[idx + 1]) - 128.0f) * contrast_const + 128.0f), 0, 255);
    d_output[idx + 2] = std::clamp(static_cast<int>((float(d_input[idx + 2]) - 128.0f) * contrast_const + 128.0f), 0, 255);
}

__global__ void adjust_gamma(uint8_t* d_input, uint8_t* d_output, int width, int height, int channels, const float gamma_const) {

    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (y >= height || x >= width) return;

    int idx = ((y * width) + x) * channels;

    d_output[idx] = std::clamp(int((std::pow((d_input[idx] / 255.0), 1.0 / gamma_const))*255.0), 0, 255);
    d_output[idx + 1] = std::clamp(int((std::pow((d_input[idx + 1] / 255.0), 1.0 / gamma_const))*255.0), 0, 255);
    d_output[idx + 2] = std::clamp(int((std::pow((d_input[idx + 2] / 255.0), 1.0 / gamma_const))*255.0), 0, 255);
}

__global__ void adjust_exposure(uint8_t* d_input, uint8_t* d_output, int width, int height, int channels, int exposure_factor) {

    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (y >= height || x >= width) return;

    int idx = ((y * width) + x) * channels;
    float exposure_multiplier = std::pow(2.0f, exposure_factor);

    d_output[idx] = std::clamp(int(d_input[idx] * exposure_multiplier), 0, 255);
    d_output[idx] = std::clamp(int(d_input[idx + 1] * exposure_multiplier), 0, 255);
    d_output[idx] = std::clamp(int(d_input[idx + 2] * exposure_multiplier), 0, 255);
}

__global__ void adjust_saturation(uint8_t* d_input, uint8_t* d_output, int width, int height, int channels, const float saturation_const) {

    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (y >= height || x >= width) return;

    int idx = ((y * width) + x) * channels;
    float gray = 0.299*d_input[idx] + 0.587*d_input[idx+1] + 0.114*d_input[idx+2];
    d_output[idx] = std::clamp(static_cast<int>(gray + (d_input[idx] - gray) * saturation_const), 0, 255);
    d_output[idx + 1] = std::clamp(static_cast<int>(gray + (d_input[idx + 1] - gray) * saturation_const), 0, 255);
    d_output[idx + 2] = std::clamp(static_cast<int>(gray + (d_input[idx + 2] - gray) * saturation_const), 0, 255);
}

__device__ __forceinline__ float clampf(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

__device__ __forceinline__ uint8_t clamp8(float v) {
    return static_cast<uint8_t>(clampf(v, 0.0f, 255.0f));
}

__global__ void adjust_hue(uint8_t* d_input, uint8_t* d_output, int width, int height, int channels, float hue_shift_deg) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;

    float r = d_input[idx + 0] / 255.0f;
    float g = d_input[idx + 1] / 255.0f;
    float b = d_input[idx + 2] / 255.0f;

    float cmax = fmaxf(fmaxf(r, g), b);
    float cmin = fminf(fminf(r, g), b);
    float delta = cmax - cmin;

    float h = 0.0f;
    if (delta > 1e-5f) {
        if (cmax == r)      h = fmodf((60.0f * ((g - b) / delta) + 360.0f), 360.0f);
        else if (cmax == g) h = fmodf((60.0f * ((b - r) / delta) + 120.0f), 360.0f);
        else                h = fmodf((60.0f * ((r - g) / delta) + 240.0f), 360.0f);
    }

    float s = (cmax == 0.0f) ? 0.0f : (delta / cmax);
    float v = cmax;

    h = fmodf(h + hue_shift_deg, 360.0f);
    if (h < 0) h += 360.0f;

    float c = v * s;
    float xh = c * (1 - fabsf(fmodf(h / 60.0f, 2.0f) - 1));
    float m = v - c;
    float r1, g1, b1;

    if (h < 60.0f)      { r1 = c;  g1 = xh; b1 = 0; }
    else if (h < 120.0f){ r1 = xh; g1 = c;  b1 = 0; }
    else if (h < 180.0f){ r1 = 0;  g1 = c;  b1 = xh; }
    else if (h < 240.0f){ r1 = 0;  g1 = xh; b1 = c; }
    else if (h < 300.0f){ r1 = xh; g1 = 0;  b1 = c; }
    else                { r1 = c;  g1 = 0;  b1 = xh; }

    d_output[idx + 0] = clamp8((r1 + m) * 255.0f);
    d_output[idx + 1] = clamp8((g1 + m) * 255.0f);
    d_output[idx + 2] = clamp8((b1 + m) * 255.0f);

    if (channels == 4) {
        d_output[idx + 3] = d_input[idx + 3];  // Copy alpha
    }
}

__global__ void compute_channel_minmax(const uint8_t* src, int width, int height, int channels, int* min_vals, int* max_vals) {
    __shared__ int local_min[3];
    __shared__ int local_max[3];

    if (threadIdx.x == 0) {
        local_min[0] = local_min[1] = local_min[2] = 255;
        local_max[0] = local_max[1] = local_max[2] = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;

    if (idx < total) {
        for (int c = 0; c < 3 && c < channels; ++c) {
            int val = src[idx * channels + c];
            atomicMin(&local_min[c], val);
            atomicMax(&local_max[c], val);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int c = 0; c < 3; ++c) {
            atomicMin(&min_vals[c], local_min[c]);
            atomicMax(&max_vals[c], local_max[c]);
        }
    }
}

void compute_channel_minmax_cpu(const uint8_t* src, int width, int height, int channels, int* min_vals, int* max_vals) {
    min_vals[0] = min_vals[1] = min_vals[2] = 255;
    max_vals[0] = max_vals[1] = max_vals[2] = 0;

    int total = width * height;
    for (int i = 0; i < total; ++i) {
        for (int c = 0; c < std::min(3, channels); ++c) {
            int val = src[i * channels + c];
            min_vals[c] = std::min(min_vals[c], val);
            max_vals[c] = std::max(max_vals[c], val);
        }
    }
}


__global__ void apply_contrast(const uint8_t* src, uint8_t* dst, int width, int height, int channels,
                               const int* min_vals, const int* max_vals) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;

    for (int c = 0; c < min(3, channels); ++c) {
        int min_c = min_vals[c];
        int max_c = max_vals[c];
        uint8_t val = src[idx + c];
        if (max_c != min_c) {
            float norm = (val - min_c) * 255.0f / (max_c - min_c);
            dst[idx + c] = static_cast<uint8_t>(fminf(fmaxf(norm, 0.0f), 255.0f));
        } else {
            dst[idx + c] = val;
        }
    }
    if (channels == 4)
        dst[idx + 3] = src[idx + 3];
}
