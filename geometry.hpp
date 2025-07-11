// THURSDAY && FRIDAY
// flip_vertically
// flip_diagonal
// flip_main_diagonal
// flip_anti_diagonal
// rotate_90_clockwise
// rotate_90_counterclockwise
// rotate_180
// rotate image270
// rotate_arbitrary (angle) w/ interpolation
// rotate 90cw and flip horizontally
// rotate 90ccw and flip vertically
//random_rotation
//random_flip_horizontal
//random_flip_vertical
//random_transpose

#ifndef imageFX_FLIP_HPP
#define imageFX_FLIP_HPP

#include "utils.hpp"

inline Image flip_horizontally(const Image& image) {
    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    if (channels < 1)
        throw std::runtime_error("Image must have at least 1 channel to flip.");
        
    Image result(width, height, channels);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < channels; c++) {
                dst[(y * width + x) * channels + c] = src[(y * width + (width - x - 1)) * channels + c];
            }
        }
    }

    return result;
}

inline Image flip_vertically(const Image& image) {
    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    if (channels < 1)
        throw std::runtime_error("Image must have at least 1 channel to flip.");
        
    Image result(width, height, channels);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < channels; c++) {
                dst[(y * width + x) * channels + c] = src[((height - y - 1) * width + x) * channels + c];
            }
        }
    }
    return result;
}

inline Image flip_both(const Image& image) {
    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    if (channels < 1)
        throw std::runtime_error("Image must have at least 1 channel to flip.");

    Image result(width, height, channels);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                dst[((height - 1 - y) * width + (width - 1 - x)) * channels + c] = src[(y * width + x) * channels + c];
            }
        }
    }

    return result;
}

inline Image flip_main_diagonal(const Image& image) {
    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    if (width != height)
        throw std::runtime_error("Main diagonal flip requires square image.");

    Image result(width, height, channels);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                dst[(x * width + y) * channels + c] = src[(y * width + x) * channels + c];
            }
        }
    }

    return result;
}

// i really liked doing this one, made me feel so very clever (for once)
inline Image flip_antidiagonal(const Image& image) {
    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    if (width != height)
        throw std::runtime_error("Anti-diagonal flip requires square image.");

    Image result(width, height, channels);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                int src_idx = (y * width + x) * channels + c;
                int dst_idx = ((width - 1 - x) * width + (height - 1 - y)) * channels + c;
                dst[dst_idx] = src[src_idx];
            }
        }
    }

    return result;
}

inline Image rotate_90_clockwise(const Image& image) {
    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    Image result(height, width, channels); 
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                dst[(x * height + (height - 1 - y)) * channels + c] = src[(y * width + x) * channels + c];
            }
        }
    }
    return result;
}

Image rotate_90_counterclockwise(const Image& image) {
    int w = image.getWidth(), h = image.getHeight(), c = image.getChannels();
    Image result(h, w, c);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int ch = 0; ch < c; ++ch)
                dst[(x * h + (h - 1 - y)) * c + ch] = src[(y * w + x) * c + ch];

    return result;
}

Image rotate_180(const Image& image) {
    int w = image.getWidth(), h = image.getHeight(), c = image.getChannels();
    Image result(w, h, c);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int ch = 0; ch < c; ++ch)
                dst[((h - 1 - y) * w + (w - 1 - x)) * c + ch] = src[(y * w + x) * c + ch];

    return result;
}

Image rotate_270(const Image& image) {
    return rotate_90_counterclockwise(image);
}

Image rotate_90cw_flip_horizontal(const Image& image) {
    Image rotated = rotate_90_counterclockwise(image);
    int w = rotated.getWidth(), h = rotated.getHeight(), c = rotated.getChannels();
    Image result(w, h, c);
    const uint8_t* src = rotated.data();
    uint8_t* dst = result.data();

    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int ch = 0; ch < c; ++ch)
                dst[(y * w + (w - 1 - x)) * c + ch] = src[(y * w + x) * c + ch];

    return result;
}

Image rotate_90ccw_flip_vertical(const Image& image) {
    Image rotated = rotate_90_counterclockwise(image);
    int w = rotated.getWidth(), h = rotated.getHeight(), c = rotated.getChannels();
    Image result(w, h, c);
    const uint8_t* src = rotated.data();
    uint8_t* dst = result.data();

    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int ch = 0; ch < c; ++ch)
                dst[((h - 1 - y) * w + x) * c + ch] = src[(y * w + x) * c + ch];

    return result;
}

Image rotate_arbitrary(const Image& image, double angle_deg) {
    int w = image.getWidth(), h = image.getHeight(), c = image.getChannels();
    double angle_rad = angle_deg * M_PI / 180.0;
    double cos_a = std::cos(angle_rad), sin_a = std::sin(angle_rad);

    int cx = w / 2, cy = h / 2;
    Image result(w, h, c);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int tx = x - cx;
            int ty = y - cy;
            double src_x = cos_a * tx + sin_a * ty + cx;
            double src_y = -sin_a * tx + cos_a * ty + cy;

            int ix = static_cast<int>(std::floor(src_x));
            int iy = static_cast<int>(std::floor(src_y));

            for (int ch = 0; ch < c; ++ch) {
                if (ix >= 0 && iy >= 0 && ix < w && iy < h)
                    dst[(y * w + x) * c + ch] = src[(iy * w + ix) * c + ch];
                else
                    dst[(y * w + x) * c + ch] = 0; 
            }
        }
    }

    return result;
}

Image random_rotation(const Image& image) {
    int choice = rand() % 4;
    switch (choice) {
        case 0: return image;
        case 1: return rotate_90_counterclockwise(image);
        case 2: return rotate_180(image);
        case 3: return rotate_270(image);
    }
    return image;
}

Image random_flip_horizontal(const Image& image) {
    if (rand() % 2 == 0) return image;
    int w = image.getWidth(), h = image.getHeight(), c = image.getChannels();
    Image result(w, h, c);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int ch = 0; ch < c; ++ch)
                dst[(y * w + (w - 1 - x)) * c + ch] = src[(y * w + x) * c + ch];

    return result;
}

Image random_flip_vertical(const Image& image) {
    if (rand() % 2 == 0) return image;
    int w = image.getWidth(), h = image.getHeight(), c = image.getChannels();
    Image result(w, h, c);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int ch = 0; ch < c; ++ch)
                dst[((h - 1 - y) * w + x) * c + ch] = src[(y * w + x) * c + ch];

    return result;
}

Image random_transpose(const Image& image) {
    if (image.getWidth() != image.getHeight() || rand() % 2 == 0) return image;
    int w = image.getWidth(), c = image.getChannels();
    Image result(w, w, c);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int y = 0; y < w; ++y)
        for (int x = 0; x < w; ++x)
            for (int ch = 0; ch < c; ++ch)
                dst[(x * w + y) * c + ch] = src[(y * w + x) * c + ch];

    return result;
}

void init_random_seed() {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
}



#endif 
