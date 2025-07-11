// SUNDAY
// gaussian_blur: Apply Gaussian-weighted blurring for a smooth, natural loo // DONE - ST
// box_blur	Average pixels in a square kernel; simplest and fastest blur // DONE - ST
// median_blur	Replaces each pixel with the median in its neighborhood; removes salt-noise // DONE - ST **
// bilateral_blur	Blur that preserves edges by considering pixel similarity and spatial distance // DONE - ST **
// motion_blur	Simulates blur caused by movement; directional kernel along X/Y/diagonal // DONE - NAIVE **
// radial_blur	Blurs pixels outward from a center point; gives a zoom or whirl look // DONE - NAIVE **
// ---------------------------
// box_blur_integral	Fast box blur using summed-area table (integral image)
// stack_blur	Approximation of Gaussian blur using multiple box blurs
// anisotropic_blur	Applies more blur in one direction than another (e.g., horizontal smoothing)
// variable_blur	Blur radius varies based on a mask or per-pixel map (depth-of-field effect)
// directional_blur(angle)	Applies blur along an arbitrary angle instead of just X or Y
// selective_blur(mask)	Applies blur only to selected areas of the image using a binary or alpha mask
// depth_blur(depth_map)	Simulates depth-of-field using a depth map to control blur strength
// edge_preserving_blur()	Smooths inside regions but avoids blurring across edges (e.g. bilateral)
// recursive_gaussian_blur()	Optimized separable Gaussian using recursive filters (faster for large radii)

#include "utils.hpp"

#include <vector>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <cmath>

inline Image box_blur(const Image& image, int radius = 1) {
    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels = image.getChannels();

    if (radius <= 0) return image; 

    Image result(width, height, channels);

    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    const int windowSize = 2 * radius + 1;
    const int windowArea = windowSize * windowSize;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum[4] = {0};  
            int count = 0;

            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;

                    nx = std::clamp(nx, 0, width - 1);
                    ny = std::clamp(ny, 0, height - 1);

                    const uint8_t* pixel = &src[(ny * width + nx) * channels];

                    for (int c = 0; c < channels; ++c) {
                        sum[c] += pixel[c];
                    }

                    count++;
                }
            }

            uint8_t* out_pixel = &dst[(y * width + x) * channels];
            for (int c = 0; c < channels; ++c) {
                out_pixel[c] = static_cast<uint8_t>(sum[c] / count);
            }
        }
    }

    return result;
}

std::vector<std::vector<float>> generateGaussianKernel(int radius, float sigma) {
    int size = 2 * radius + 1;
    std::vector<std::vector<float>> kernel(size, std::vector<float>(size));
    float sum = 0.0f;

    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            float exponent = -(x * x + y * y) / (2 * sigma * sigma);
            float value = std::exp(exponent) / (2 * M_PI * sigma * sigma);
            kernel[y + radius][x + radius] = value;
            sum += value;
        }
    }

    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x)
            kernel[y][x] /= sum;

    return kernel;
}

Image gaussianBlur(const Image& image, int radius = 1, float sigma = 1.0f) {
    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels = image.getChannels();

    if (radius <= 0 || sigma <= 0.0f) return image;

    const std::vector<std::vector<float>> kernel = generateGaussianKernel(radius, sigma);

    Image result(width, height, channels);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                float weighted_sum = 0.0f;

                for (int ky = -radius; ky <= radius; ++ky) {
                    for (int kx = -radius; kx <= radius; ++kx) {
                        int ix = std::clamp(x + kx, 0, width - 1);
                        int iy = std::clamp(y + ky, 0, height - 1);
                        float weight = kernel[ky + radius][kx + radius];
                        weighted_sum += src[(iy * width + ix) * channels + c] * weight;
                    }
                }

                dst[(y * width + x) * channels + c] = static_cast<uint8_t>(std::clamp(weighted_sum, 0.0f, 255.0f));
            }
        }
    }

    return result;
}

inline Image median_blur(const Image& image, int radius = 1) {
    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels = image.getChannels();

    if (radius <= 0) return image;

    Image result(width, height, channels);
    const uint8_t* src = image.data();
    uint8_t* dest = result.data();

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            
            std::vector<uint8_t> r_vals, g_vals, b_vals, a_vals;

            for (int wy = -radius; wy <= radius; wy++) {
                for (int wx = -radius; wx <= radius; wx++) {
                    int cx = std::clamp(x + wx, 0, width - 1);
                    int cy = std::clamp(y + wy, 0, height - 1);

                    const uint8_t* pixel = &src[(cy * width + cx) * channels];

                    r_vals.push_back(pixel[0]);
                    if (channels >= 3) g_vals.push_back(pixel[1]);
                    if (channels >= 3) b_vals.push_back(pixel[2]);
                    if (channels == 4) a_vals.push_back(pixel[3]);
                }
            }

            std::sort(r_vals.begin(), r_vals.end());
            if (channels >= 3) std::sort(g_vals.begin(), g_vals.end());
            if (channels >= 3) std::sort(b_vals.begin(), b_vals.end());
            if (channels == 4) std::sort(a_vals.begin(), a_vals.end());

            uint8_t* dst_pixel = &dest[(y * width + x) * channels];
            dst_pixel[0] = r_vals[r_vals.size() / 2];
            if (channels >= 3) dst_pixel[1] = g_vals[g_vals.size() / 2];
            if (channels >= 3) dst_pixel[2] = b_vals[b_vals.size() / 2];
            if (channels == 4) dst_pixel[3] = a_vals[a_vals.size() / 2];
        }
    }

    return result;
}


inline float gaussian(float x, float sigma) {
    return std::exp(-(x * x) / (2 * sigma * sigma));
}

inline Image bilateral_blur(const Image& image, int radius = 2, float sigma_spatial = 2.0f, float sigma_range = 25.0f) {
    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels = image.getChannels();

    if (channels < 3) throw std::runtime_error("Bilateral filter requires at least 3 channels (RGB)");

    Image result(width, height, channels);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum[3] = {0.0f, 0.0f, 0.0f};
            float weight_sum = 0.0f;

            const uint8_t* center_pixel = &src[(y * width + x) * channels];

            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    int nx = std::clamp(x + dx, 0, width - 1);
                    int ny = std::clamp(y + dy, 0, height - 1);
                    const uint8_t* neighbor = &src[(ny * width + nx) * channels];

                    float spatial_dist = dx * dx + dy * dy;
                    float spatial_weight = gaussian(std::sqrt(spatial_dist), sigma_spatial);

                    float color_dist = 0.0f;
                    for (int c = 0; c < 3; ++c)
                        color_dist += (neighbor[c] - center_pixel[c]) * (neighbor[c] - center_pixel[c]);
                    color_dist = std::sqrt(color_dist);
                    float range_weight = gaussian(color_dist, sigma_range);

                    float total_weight = spatial_weight * range_weight;

                    for (int c = 0; c < 3; ++c)
                        sum[c] += neighbor[c] * total_weight;

                    weight_sum += total_weight;
                }
            }

            uint8_t* out_pixel = &dst[(y * width + x) * channels];
            for (int c = 0; c < 3; ++c)
                out_pixel[c] = static_cast<uint8_t>(sum[c] / weight_sum);

            if (channels == 4)
                out_pixel[3] = center_pixel[3]; 
        }
    }

    return result;
}

inline Image motion_blur(const Image& image, int length = 5, const std::string& direction = "horizontal") {
    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels = image.getChannels();

    if (length <= 1 || length % 2 == 0)
        throw std::invalid_argument("Length must be an odd number ≥ 3");

    Image result(width, height, channels);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    int dx = 0, dy = 0;
    if (direction == "horizontal")       { dx = 1; dy = 0; }
    else if (direction == "vertical")    { dx = 0; dy = 1; }
    else if (direction == "diagonal")    { dx = 1; dy = 1; }
    else if (direction == "anti-diagonal") { dx = 1; dy = -1; }
    else throw std::invalid_argument("Unsupported motion blur direction");

    int half = length / 2;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum[4] = {0};
            int count = 0;

            for (int i = -half; i <= half; ++i) {
                int nx = std::clamp(x + i * dx, 0, width - 1);
                int ny = std::clamp(y + i * dy, 0, height - 1);

                if (direction == "anti-diagonal")
                    ny = std::clamp(y + i * dy, 0, height - 1);

                const uint8_t* pixel = &src[(ny * width + nx) * channels];

                for (int c = 0; c < channels; ++c)
                    sum[c] += pixel[c];

                count++;
            }

            uint8_t* out = &dst[(y * width + x) * channels];
            for (int c = 0; c < channels; ++c)
                out[c] = static_cast<uint8_t>(sum[c] / count);
        }
    }

    return result;
}

inline Image radial_blur(const Image& image, int strength = 5, float center_x = 0.5f, float center_y = 0.5f) {
    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels = image.getChannels();

    Image result(width, height, channels);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    int cx = static_cast<int>(center_x * width);
    int cy = static_cast<int>(center_y * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum[4] = {0};
            int count = 0;

            for (int s = 0; s < strength; ++s) {
                float t = s / static_cast<float>(strength);

                int nx = static_cast<int>((1.0f - t) * x + t * cx);
                int ny = static_cast<int>((1.0f - t) * y + t * cy);

                nx = std::clamp(nx, 0, width - 1);
                ny = std::clamp(ny, 0, height - 1);

                const uint8_t* pixel = &src[(ny * width + nx) * channels];
                for (int c = 0; c < channels; ++c)
                    sum[c] += pixel[c];

                count++;
            }

            uint8_t* out_pixel = &dst[(y * width + x) * channels];
            for (int c = 0; c < channels; ++c)
                out_pixel[c] = static_cast<uint8_t>(sum[c] / count);
        }
    }

    return result;
}
