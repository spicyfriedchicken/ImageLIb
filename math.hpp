// add(image1, image2)	Per-pixel addition (clamped)
// subtract(image1, image2)	Per-pixel subtraction
// multiply(image1, image2)	Per-pixel multiply (e.g. blend mask)
// divide(image1, image2)	Per-pixel division with zero check
// blend(image1, image2, alpha)	Alpha blend two images using a float
// abs_diff(image1, image2)	Absolute difference (used in motion detection, etc.) //

#include "utils.hpp"
Image add_two_images(const Image& image1, const Image& image2, bool blend_alpha = false) {
    if (image1.getWidth() != image2.getWidth() || 
        image1.getHeight() != image2.getHeight() || 
        image1.getChannels() != image2.getChannels()) {
        throw std::runtime_error("Error! Images must have the same width, height, and number of channels.");
    }

    int width = image1.getWidth();
    int height = image1.getHeight();
    int channels = image1.getChannels();
    int pixel_count = width * height;

    const uint8_t* data1 = image1.data();
    const uint8_t* data2 = image2.data();

    Image result(width, height, channels);
    uint8_t* dst = result.data();

    if (channels == 4 && blend_alpha) {
        for (int i = 0; i < pixel_count; ++i) {
            int idx = i * 4;

            float A_r = data1[idx + 0] / 255.0f;
            float A_g = data1[idx + 1] / 255.0f;
            float A_b = data1[idx + 2] / 255.0f;
            float A_a = data1[idx + 3] / 255.0f;

            float B_r = data2[idx + 0] / 255.0f;
            float B_g = data2[idx + 1] / 255.0f;
            float B_b = data2[idx + 2] / 255.0f;
            float B_a = data2[idx + 3] / 255.0f;

            float out_a = A_a + B_a * (1.0f - A_a);
            if (out_a < 1e-6f) out_a = 1e-6f; 

            float out_r = (A_r * A_a + B_r * B_a * (1.0f - A_a)) / out_a;
            float out_g = (A_g * A_a + B_g * B_a * (1.0f - A_a)) / out_a;
            float out_b = (A_b * A_a + B_b * B_a * (1.0f - A_a)) / out_a;

            dst[idx + 0] = std::clamp(int(out_r * 255.0f), 0, 255);
            dst[idx + 1] = std::clamp(int(out_g * 255.0f), 0, 255);
            dst[idx + 2] = std::clamp(int(out_b * 255.0f), 0, 255);
            dst[idx + 3] = std::clamp(int(out_a * 255.0f), 0, 255);
        }
    }
    else {
        for (int i = 0; i < pixel_count; ++i) {
            for (int c = 0; c < channels; ++c) {
                int idx = i * channels + c;
                if (channels == 4 && c == 3) {
                    dst[idx] = std::max(data1[idx], data2[idx]); 
                    continue;
                }

                int sum = data1[idx] + data2[idx];
                dst[idx] = std::clamp(sum, 0, 255);
            }
        }
    }

    return result;
}

Image subtract_two_images(const Image& image1, const Image& image2, bool blend_alpha = false) {
    if (image1.getWidth() != image2.getWidth() || 
        image1.getHeight() != image2.getHeight() || 
        image1.getChannels() != image2.getChannels()) {
        throw std::runtime_error("Error! Images must have the same width, height, and number of channels.");
    }

    int width = image1.getWidth();
    int height = image1.getHeight();
    int channels = image1.getChannels();
    int pixel_count = width * height;

    const uint8_t* data1 = image1.data();
    const uint8_t* data2 = image2.data();

    Image result(width, height, channels);
    uint8_t* dst = result.data();

    if (channels == 4 && blend_alpha) {
        for (int i = 0; i < pixel_count; ++i) {
            int idx = i * 4;

            float A_r = data1[idx + 0] / 255.0f;
            float A_g = data1[idx + 1] / 255.0f;
            float A_b = data1[idx + 2] / 255.0f;
            float A_a = data1[idx + 3] / 255.0f;

            float B_r = data2[idx + 0] / 255.0f;
            float B_g = data2[idx + 1] / 255.0f;
            float B_b = data2[idx + 2] / 255.0f;
            float B_a = data2[idx + 3] / 255.0f;

            float out_a = A_a + B_a * (1.0f - A_a);
            if (out_a < 1e-6f) out_a = 1e-6f;

            float out_r = std::clamp((A_r - B_r) * A_a, 0.0f, 1.0f);
            float out_g = std::clamp((A_g - B_g) * A_a, 0.0f, 1.0f);
            float out_b = std::clamp((A_b - B_b) * A_a, 0.0f, 1.0f);

            dst[idx + 0] = int(out_r * 255.0f);
            dst[idx + 1] = int(out_g * 255.0f);
            dst[idx + 2] = int(out_b * 255.0f);
            dst[idx + 3] = std::clamp(int(out_a * 255.0f), 0, 255);
        }
    } 
    else {
        for (int i = 0; i < pixel_count; ++i) {
            for (int c = 0; c < channels; ++c) {
                int idx = i * channels + c;

                if (channels == 4 && c == 3) {
                    dst[idx] = std::max(data1[idx], data2[idx]); 
                    continue;
                }

                int diff = data1[idx] - data2[idx];
                dst[idx] = std::clamp(diff, 0, 255);
            }
        }
    }

    return result;
}

Image multiply_two_images(const Image& image1, const Image& image2, bool blend_alpha = false) {
    if (image1.getWidth() != image2.getWidth() || 
        image1.getHeight() != image2.getHeight() || 
        image1.getChannels() != image2.getChannels()) {
        throw std::runtime_error("Error! Images must have the same width, height, and number of channels.");
    }

    int width = image1.getWidth();
    int height = image1.getHeight();
    int channels = image1.getChannels();
    int pixel_count = width * height;

    const uint8_t* data1 = image1.data();
    const uint8_t* data2 = image2.data();

    Image result(width, height, channels);
    uint8_t* dst = result.data();

    if (channels == 4 && blend_alpha) {
        for (int i = 0; i < pixel_count; ++i) {
            int idx = i * 4;

            float A_r = data1[idx + 0] / 255.0f;
            float A_g = data1[idx + 1] / 255.0f;
            float A_b = data1[idx + 2] / 255.0f;
            float A_a = data1[idx + 3] / 255.0f;

            float B_r = data2[idx + 0] / 255.0f;
            float B_g = data2[idx + 1] / 255.0f;
            float B_b = data2[idx + 2] / 255.0f;
            float B_a = data2[idx + 3] / 255.0f;

            float out_a = A_a + B_a * (1.0f - A_a);
            if (out_a < 1e-6f) out_a = 1e-6f;

            float out_r = (A_r * A_a * B_r * B_a * (1.0f - A_a)) / out_a;
            float out_g = (A_g * A_a * B_g * B_a * (1.0f - A_a)) / out_a;
            float out_b = (A_b * A_a * B_b * B_a * (1.0f - A_a)) / out_a;

            dst[idx + 0] = int(out_r * 255.0f);
            dst[idx + 1] = int(out_g * 255.0f);
            dst[idx + 2] = int(out_b * 255.0f);
            dst[idx + 3] = std::clamp(int(out_a * 255.0f), 0, 255);
        }
    } 
    else {
        for (int i = 0; i < pixel_count; ++i) {
            for (int c = 0; c < channels; ++c) {
                int idx = i * channels + c;

                if (channels == 4 && c == 3) {
                    dst[idx] = std::max(data1[idx], data2[idx]); 
                    continue;
                }

                int diff = data1[idx] * data2[idx];
                dst[idx] = std::clamp(diff, 0, 255);
            }
        }
    }

    return result;
}

Image multiply_two_images(const Image& image1, const Image& image2, bool blend_alpha = false) {
    if (image1.getWidth() != image2.getWidth() || 
        image1.getHeight() != image2.getHeight() || 
        image1.getChannels() != image2.getChannels()) {
        throw std::runtime_error("Error! Images must have the same width, height, and number of channels.");
    }

    int width = image1.getWidth();
    int height = image1.getHeight();
    int channels = image1.getChannels();
    int pixel_count = width * height;

    const uint8_t* data1 = image1.data();
    const uint8_t* data2 = image2.data();

    Image result(width, height, channels);
    uint8_t* dst = result.data();
    if (channels == 4 && blend_alpha) {
        for (int i = 0; i < pixel_count; ++i) {
            int idx = i * 4;

            float A_r = data1[idx + 0] / 255.0f;
            float A_g = data1[idx + 1] / 255.0f;
            float A_b = data1[idx + 2] / 255.0f;
            float A_a = data1[idx + 3] / 255.0f;

            float B_r = data2[idx + 0] / 255.0f;
            float B_g = data2[idx + 1] / 255.0f;
            float B_b = data2[idx + 2] / 255.0f;
            float B_a = data2[idx + 3] / 255.0f;

            float out_r = A_r * B_r;
            float out_g = A_g * B_g;
            float out_b = A_b * B_b;

            float out_a = A_a + B_a * (1.0f - A_a);

            dst[idx + 0] = std::clamp(int(out_r * 255.0f), 0, 255);
            dst[idx + 1] = std::clamp(int(out_g * 255.0f), 0, 255);
            dst[idx + 2] = std::clamp(int(out_b * 255.0f), 0, 255);
            dst[idx + 3] = std::clamp(int(out_a * 255.0f), 0, 255);
        }
    } 
    else {
        for (int i = 0; i < pixel_count; ++i) {
            for (int c = 0; c < channels; ++c) {
                int idx = i * channels + c;

                if (channels == 4 && c == 3) {
                    dst[idx] = std::max(data1[idx], data2[idx]);
                    continue;
                }
                int product = (data1[idx] * data2[idx]) / 255;
                dst[idx] = std::clamp(product, 0, 255);
            }
        }
    }

    return result;
}

Image divide_two_images(const Image& image1, const Image& image2, bool blend_alpha = false) {
    if (image1.getWidth() != image2.getWidth() || 
        image1.getHeight() != image2.getHeight() || 
        image1.getChannels() != image2.getChannels()) {
        throw std::runtime_error("Images must be the same size and type for division.");
    }

    int width = image1.getWidth();
    int height = image1.getHeight();
    int channels = image1.getChannels();
    int pixel_count = width * height;

    const uint8_t* data1 = image1.data();
    const uint8_t* data2 = image2.data();

    Image result(width, height, channels);
    uint8_t* dst = result.data();

    if (channels == 4 && blend_alpha) {
        for (int i = 0; i < pixel_count; ++i) {
            int idx = i * 4;

            float A_r = data1[idx + 0] / 255.0f;
            float A_g = data1[idx + 1] / 255.0f;
            float A_b = data1[idx + 2] / 255.0f;
            float A_a = data1[idx + 3] / 255.0f;

            float B_r = data2[idx + 0] / 255.0f;
            float B_g = data2[idx + 1] / 255.0f;
            float B_b = data2[idx + 2] / 255.0f;
            float B_a = data2[idx + 3] / 255.0f;

            float out_r = std::clamp(A_r / std::max(B_r, 1e-6f), 0.0f, 1.0f);
            float out_g = std::clamp(A_g / std::max(B_g, 1e-6f), 0.0f, 1.0f);
            float out_b = std::clamp(A_b / std::max(B_b, 1e-6f), 0.0f, 1.0f);
            float out_a = A_a + B_a * (1.0f - A_a);

            dst[idx + 0] = int(out_r * 255.0f);
            dst[idx + 1] = int(out_g * 255.0f);
            dst[idx + 2] = int(out_b * 255.0f);
            dst[idx + 3] = std::clamp(int(out_a * 255.0f), 0, 255);
        }
    }
    else {
        for (int i = 0; i < pixel_count * channels; ++i) {
            if (channels == 4 && (i % 4 == 3)) {
                dst[i] = std::max(data1[i], data2[i]);
                continue;
            }

            int result = int(data1[i]) * 255 / std::max(int(data2[i]), 1);  
            dst[i] = std::clamp(result, 0, 255);
        }
    }

    return result;
}


Image blend(const Image& image1, const Image& image2, float alpha) {
    if (image1.getWidth() != image2.getWidth() ||
        image1.getHeight() != image2.getHeight() ||
        image1.getChannels() != image2.getChannels()) {
        throw std::runtime_error("Images must be the same size and type to blend.");
    }

    int width = image1.getWidth();
    int height = image1.getHeight();
    int channels = image1.getChannels();
    int pixel_count = width * height;

    const uint8_t* data1 = image1.data();
    const uint8_t* data2 = image2.data();

    Image result(width, height, channels);
    uint8_t* dst = result.data();

    float beta = 1.0f - alpha;

    for (int i = 0; i < pixel_count * channels; ++i) {
        dst[i] = std::clamp(int(data1[i] * alpha + data2[i] * beta), 0, 255);
    }

    return result;
}

Image abs_diff(const Image& image1, const Image& image2) {
    if (image1.getWidth() != image2.getWidth() ||
        image1.getHeight() != image2.getHeight() ||
        image1.getChannels() != image2.getChannels()) {
        throw std::runtime_error("Images must be the same size and type to compute absolute difference.");
    }

    int width = image1.getWidth();
    int height = image1.getHeight();
    int channels = image1.getChannels();
    int pixel_count = width * height;

    const uint8_t* data1 = image1.data();
    const uint8_t* data2 = image2.data();

    Image result(width, height, channels);
    uint8_t* dst = result.data();

    for (int i = 0; i < pixel_count * channels; ++i) {
        dst[i] = static_cast<uint8_t>(std::abs(data1[i] - data2[i]));
    }

    return result;
}
