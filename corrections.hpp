// MONDAY
// lens_distortion_correction()	Corrects barrel/pincushion lens distortion // DONE - NAIVE
// chromatic_aberration_correction()	Adjusts color channel offsets on edges // DONE - NAIVE
// rolling_shutter_correction()	Corrects skew in frames captured line-by-line // DONE - NAIVE
// white_balance_auto()	Auto-detect and correct color temperature // DONE - NAIVE
// color_balance(r_adj, g_adj, b_adj)	Manual per-channel color curve adjust // DONE - NAIVE

#include "utils.hpp"

Image lens_distortion_correction(Image& image, float k1, float k2 = 0.0f) {
    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();
    Image result(width, height, channels);

    float cx = width / 2.0f;
    float cy = height / 2.0f;

    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float norm_x = (x - cx) / cx;
            float norm_y = (y - cy) / cy;
            float r2 = norm_x * norm_x + norm_y * norm_y;
            float factor = 1 + k1 * r2 + k2 * r2 * r2;

            float src_x = cx + norm_x * cx * factor;
            float src_y = cy + norm_y * cy * factor;

            int sx = std::clamp(int(src_x), 0, width - 1);
            int sy = std::clamp(int(src_y), 0, height - 1);

            int src_idx = (sy * width + sx) * channels;
            int dst_idx = (y * width + x) * channels;

            for (int c = 0; c < channels; ++c)
                dst[dst_idx + c] = src[src_idx + c];
        }
    }

    return result;
}

Image chromatic_aberration_correction(Image& image, float r_shift = 1.0f, float b_shift = -1.0f) {
    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    Image result(width, height, channels);

    float cx = width / 2.0f;
    float cy = height / 2.0f;

    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float dx = x - cx;
            float dy = y - cy;

            int idx = (y * width + x) * channels;

            int rx = std::clamp(int(x + r_shift * dx / cx), 0, width - 1);
            int ry = std::clamp(int(y + r_shift * dy / cy), 0, height - 1);
            int ridx = (ry * width + rx) * channels;

            int bx = std::clamp(int(x + b_shift * dx / cx), 0, width - 1);
            int by = std::clamp(int(y + b_shift * dy / cy), 0, height - 1);
            int bidx = (by * width + bx) * channels;
            dst[idx + 0] = src[ridx + 0]; 
            dst[idx + 1] = src[idx + 1];  
            dst[idx + 2] = src[bidx + 2];
        }
    }

   return result;
}

Image rolling_shutter_correction(Image& image, float shear_factor) {
    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    Image result(width, height, channels);

    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int y = 0; y < height; ++y) {
        float offset = shear_factor * (y - height / 2.0f);

        for (int x = 0; x < width; ++x) {
            int sx = std::clamp(int(x + offset), 0, width - 1);
            int idx_dst = (y * width + x) * channels;
            int idx_src = (y * width + sx) * channels;

            for (int c = 0; c < channels; ++c)
                dst[idx_dst + c] = src[idx_src + c];
        }
    }

    return result;
}

Image white_balance_auto(const Image& image) {
    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    Image result(width, height, channels);

    const uint8_t* data = image.data();
    uint8_t* destination = result.data();

    double r_sum = 0, g_sum = 0, b_sum = 0;
    int pixel_count = width * height;

    for (int i = 0; i < pixel_count; ++i) {
        int idx = i * channels;
        r_sum += data[idx + 0];
        g_sum += data[idx + 1];
        b_sum += data[idx + 2];
    }

    double r_avg = r_sum / pixel_count;
    double g_avg = g_sum / pixel_count;
    double b_avg = b_sum / pixel_count;

    double gray_avg = (r_avg + g_avg + b_avg) / 3.0;

    float r_gain = gray_avg / r_avg;
    float g_gain = gray_avg / g_avg;
    float b_gain = gray_avg / b_avg;

    for (int i = 0; i < pixel_count; ++i) {
        int idx = i * channels;
        destination[idx + 0] = std::clamp(int(data[idx + 0] * r_gain), 0, 255);
        destination[idx + 1] = std::clamp(int(data[idx + 1] * g_gain), 0, 255);
        destination[idx + 2] = std::clamp(int(data[idx + 2] * b_gain), 0, 255);
    }

    return result;
}


Image color_balance(Image& image, float r_gain, float g_gain, float b_gain) {
    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();
    
    if (channels != 3 || channels != 4) {
        throw std::runtime_error("Error, input image for color_balance must be of type RGB or RGBA, received the following channel count in input image: " + channels);
    }
    uint8_t* data = image.data();
    Image result(width, height, channels);
    uint8_t* destination = result.data();

    for (int i = 0; i < width * height; ++i) {
        int idx = i * channels;

        destination[idx + 0] = std::clamp(int(data[idx + 0] * r_gain), 0, 255); 
        destination[idx + 1] = std::clamp(int(data[idx + 1] * g_gain), 0, 255); 
        destination[idx + 2] = std::clamp(int(data[idx + 2] * b_gain), 0, 255);
        if (channels == 4) {
           destination[idx + 3] = data[idx + 3];
        }
    }

    return result;
}
