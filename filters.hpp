// grayscale: Convert the image to shades of gray (remove all color) // IMPLEMENTED - NAIVE 04/15
// median_filter: Reduce noise by replacing each pixel with the median of its neighborhood // IMPLEMENTED - NAIVE 04/15
// sharpen: Enhance edge contrast to make the image appear clearer or crisper // IMPLEMENTED - NAIVE 04/15
// unsharp_mask: Sharpen the image by subtracting a blurred copy to boost detail // IMPLEMENTED - NAIVE 04/15
// sobel_filter: Detect edges using the Sobel operator (emphasizes horizontal and vertical edges) // IMPLEMENTED - NAIVE 04/15
// emboss: Create a raised emboss effect by turning edges into shadows and highlights // IMPLEMENTED - NAIVE 04/15
// laplacian_filter: Detect edges using the Laplacian operator (finds outlines in all directions)  // IMPLEMENTED - NAIVE 04/15
// sepia: Apply a warm brown tone for an old-fashioned photo appearance // IMPLEMENTED - NAIVE 04/15
// invert_colors: Invert all the colors to produce a photographic negative effect  // IMPLEMENTED - NAIVE 04/15
// posterize: Reduce the number of color levels to create flat, poster-like regions // IMPLEMENTED - NAIVE 04/15
// solarize: Partially invert colors above a certain brightness threshold // IMPLEMENTED - NAIVE 04/15
// tint: Overlay a uniform color tint onto the image, shifting its overall hue // IMPLEMENTED - NAIVE 04/15
// pixelate: Reduce detail by enlarging pixels, creating a blocky retro effect // IMPLEMENTED - NAIVE 04/15
// dither: Apply dithering noise to simulate more colors when reducing color depth // IMPLEMENTED - NAIVE 04/15
// vignette: Darken the corners and edges to draw attention to the center of the image // IMPLEMENTED - NAIVE 04/15

#include "utils.hpp"    
#include "blur.hpp"

inline Image to_grayscale(const Image& image) {
    if (image.getChannels() < 3)
        throw std::runtime_error("Image must have at least 3 channels for grayscale conversion");

    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    std::vector<uint8_t> new_data(image.data(), image.data() + width * height * channels);
    Image result(width, height, channels, new_data);

    uint8_t* ptr = result.data();
    int total_pixels = width * height;

    for (int i = 0; i < total_pixels; ++i, ptr += channels) {
        uint8_t r = ptr[0];
        uint8_t g = ptr[1];
        uint8_t b = ptr[2];
        uint8_t gray = (r * 77 + g * 150 + b * 29) >> 8;
        ptr[0] = ptr[1] = ptr[2] = gray;
    }

    return result;
}

inline Image median_blur(const Image& image, int radius = 1) {
    if (radius <= 0) {
        throw std::runtime_error("Error. Median blur method requires an natural number radius. ");
    }
    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels = image.getChannels();
    if (channels != 3 || channels != 4) {
        throw std::runtime_error("Error. Median blur method requires an image in RGB or RGBA format.");
    }

    Image result(width, height, channels);
    
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();
    
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {

            std::vector<uint8_t> r_vals, g_vals, b_vals, a_vals;
            

            for (int wx = -radius; wx <= radius; wx++) {
                for (int wy = -radius; wy <= radius; wy++) {

                    int cx = std::clamp(x + wx, 0, width - 1);
                    int cy = std::clamp(y + wy, 0, height - 1);

                    const uint8_t* pixel = &src[(cy * width + cx) * channels];

                    r_vals.push_back(pixel[0]);
                    g_vals.push_back(pixel[1]);
                    b_vals.push_back(pixel[2]);
                    if (channels == 4) {a_vals.push_back(pixel[3]);}
                }
            }
            std::sort(r_vals.begin(), r_vals.end());
            std::sort(g_vals.begin(), g_vals.end());
            std::sort(b_vals.begin(), b_vals.end());
            if (channels == 4) {std::sort(a_vals.begin(), a_vals.end());}
            
            uint8_t* dst_pixel = &dst[(y * height + x) * channels];
            dst_pixel[0] = r_vals[r_vals.size()/2];
            dst_pixel[1] = g_vals[g_vals.size()/2];
            dst_pixel[2] = b_vals[b_vals.size()/2];
            if (channels == 4) dst_pixel[3] = a_vals[a_vals.size()/2];
        }
    }

    return result;

}

inline Image sharpen_kernel(const Image& image) {
    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels = image.getChannels();

    if (channels != 3 && channels != 4) {
        throw std::runtime_error("Error. Sharpen kernel requires an image in RGB or RGBA format.");
    }

    const int kernel[3][3] = {
        { 0, -1,  0 },
        {-1,  5, -1 },
        { 0, -1,  0 }
    };

    Image result(width, height, channels);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int red = 0, green = 0, blue = 0;

            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    int cx = std::clamp(x + j, 0, width - 1);
                    int cy = std::clamp(y + i, 0, height - 1);
                    int idx = (cy * width + cx) * channels;

                    int k = kernel[i + 1][j + 1]; // remember our bounds are -1 to 1!
                    red   += src[idx]     * k;
                    green += src[idx + 1] * k;
                    blue  += src[idx + 2] * k;
                }
            }

            int out_idx = (y * width + x) * channels;
            dst[out_idx]     = std::clamp(red,   0, 255);
            dst[out_idx + 1] = std::clamp(green, 0, 255);
            dst[out_idx + 2] = std::clamp(blue,  0, 255);
            if (channels == 4)
                dst[out_idx + 3] = src[out_idx + 3]; 
        }
    }

    return result;
}

Image unsharp_mask(const Image& image, float amount = 1.0f) {
    if (amount < 0.0f || amount > 5.0f) {
        throw std::runtime_error("Amount must be between 0.0 and 5.0");
    }

    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels = image.getChannels();

    if (channels != 3 && channels != 4) {
        throw std::runtime_error("Error. Sharpen kernel requires an image in RGB or RGBA format.");
    }

    Image blurred = gaussianBlur(image, 1, 1.0f);
    Image result = Image(width, height, channels);

    const uint8_t* src_data = image.data();
    uint8_t* blur_data  = blurred.data();
    uint8_t* dst_data = result.data();

    const int total_pixels = width * height * channels;

    for (int p = 0; p < total_pixels; p++) {
        float src_val = static_cast<float>(src_data[p]);
        float blur_val = static_cast<float>(blur_data[p]);
        float sharpened = src_val + amount * (src_val - blur_val);
        dst_data[p] = static_cast<uint8_t>(std::clamp(static_cast<int>(sharpened), 0, 255));
        
    }
    return result;
}

Image sobel_filter(const Image& image) {

    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels = image.getChannels();

    if (channels != 3 && channels != 4) {
        throw std::runtime_error("Error. Sobel filter requires an image in RGB or RGBA format.");
    }

    Image gray = to_grayscale(image);
    Image result = Image(width, height, 1);

    const int Gx[3][3] = { // horizontal filterrrr
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    const int Gy[3][3] = { // vertical filterrrrrr
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    const uint8_t* src = gray.data();
    uint8_t* dst = result.data();

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int gx = 0, gy = 0;
    
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int cx = std::clamp(x + kx, 0, width - 1);
                    int cy = std::clamp(y + ky, 0, height - 1);
    
                    uint8_t pixel = src[cy * width + cx];
                    gx += pixel * Gx[ky + 1][kx + 1];
                    gy += pixel * Gy[ky + 1][kx + 1];
                }
            }
    
            int magnitude = static_cast<int>(std::sqrt(gx * gx + gy * gy));
            dst[y * width + x] = std::clamp(magnitude, 0, 255);
        }
    }
    
    return result;
}

Image emboss (const Image& image) {

    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels = image.getChannels();

    const int kernel[3][3] = {
        {-2, -1, 0},
        {-1,  1, 1},
        { 0,  1, 2}
    };
    

    Image gray = to_grayscale(image);
    Image result(width, height, channels);

    const uint8_t* src= gray.data();
    uint8_t* dst = result.data();

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int px_val = 0;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int cx = std::clamp(x + i, 0, 255);
                    int cy = std::clamp(y + j, 0, 255);
                    px_val += src[(cy) * width + (cx)] * kernel[i + 1][j + 1];
                }
            }
            dst[y * width + x] = static_cast<uint8_t>(std::clamp(px_val + 128, 0, 255));
        }
    }
    return result;
}

Image laplacian_filter(const Image& image, bool aggressive = false) {
    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels = image.getChannels();
    int kernel[3][3];

    if (aggressive) {
        int temp[3][3] = {
            { -1, -1, -1 },
            { -1,  8, -1 },
            { -1, -1, -1 }
        };
        std::memcpy(kernel, temp, sizeof(kernel));

    } else {
        int temp[3][3] = {
            { 0, -1,  0 },
            {-1,  4, -1 },
            { 0, -1,  0 }
        };
        std::memcpy(kernel, temp, sizeof(kernel));
    }

    Image gray = to_grayscale(image);
    Image result(width, height, channels);

    const uint8_t* src  = gray.data();
    uint8_t* dst = result.data();

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int res = 0;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int cx = std::clamp(x + i, 0, width - 1);
                    int cy = std::clamp(y + j, 0, height - 1);

                    res += src[cy * width + cx] * kernel[i+1][j+1];
                }
            }
            dst[y * width + x] = static_cast<uint8_t>(std::clamp(res, 0, 255));
        }
    }

    return result;
}


Image sepia(const Image& image) {
    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels = image.getChannels();

    if (channels < 3) {
        throw std::runtime_error("Sepia filter requires RGB or RGBA image.");
    }

    Image result(width, height, channels);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int i = 0; i < width * height; ++i) {
        int idx = i * channels;

        uint8_t r = src[idx];
        uint8_t g = src[idx + 1];
        uint8_t b = src[idx + 2];

        int out_r = std::clamp(static_cast<int>(0.393f * r + 0.769f * g + 0.189f * b), 0, 255);
        int out_g = std::clamp(static_cast<int>(0.349f * r + 0.686f * g + 0.168f * b), 0, 255);
        int out_b = std::clamp(static_cast<int>(0.272f * r + 0.534f * g + 0.131f * b), 0, 255);

        dst[idx]     = static_cast<uint8_t>(out_r);
        dst[idx + 1] = static_cast<uint8_t>(out_g);
        dst[idx + 2] = static_cast<uint8_t>(out_b);

        if (channels == 4) {
            dst[idx + 3] = src[idx + 3];  
        }
    }

    return result;
}

Image invert_colors(const Image& image) {
    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels = image.getChannels();

    if (channels < 3) {
        throw std::runtime_error("Sepia filter requires RGB or RGBA image.");
    }

    Image result(width, height, channels);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int i = 0; i < width * height; ++i) {
        int idx = i * channels;
    
        dst[idx]     = static_cast<uint8_t>(255 - src[idx]);
        dst[idx + 1] = static_cast<uint8_t>(255 - src[idx + 1]);
        dst[idx + 2] = static_cast<uint8_t>(255 - src[idx + 2]);

        if (channels == 4) {
            dst[idx + 3] = src[idx + 3];  
        }
    }

    return result;
}


Image posterize(const Image& image, int levels = 4) {
    if (levels <= 1 || levels > 256) {
        throw std::runtime_error("Levels must be between 2 and 256");
    }

    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels = image.getChannels();

    Image result(width, height, channels);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    int step = 256 / levels;

    for (int i = 0; i < width * height; ++i) {
        int idx = i * channels;

        for (int c = 0; c < std::min(3, channels); ++c) {
            dst[idx + c] = static_cast<uint8_t>((src[idx + c] / step) * step);
        }

        if (channels == 4) {
            dst[idx + 3] = src[idx + 3];
        }
    }

    return result;
}

Image solarize(const Image& image, uint8_t threshold = 128) {
    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels = image.getChannels();

    Image result(width, height, channels);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int i = 0; i < width * height; ++i) {
        int idx = i * channels;

        for (int c = 0; c < std::min(3, channels); ++c) {
            uint8_t val = src[idx + c];
            dst[idx + c] = (val > threshold) ? (255 - val) : val;
        }

        if (channels == 4) {
            dst[idx + 3] = src[idx + 3];  
        }
    }

    return result;
}

Image tint(const Image& image, std::array<uint8_t, 3> color, float strength = 0.2f) {
    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels = image.getChannels();

    Image result(width, height, channels);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int i = 0; i < width * height; ++i) {
        int idx = i * channels;

        for (int c = 0; c < 3; ++c) {
            float blended = (1 - strength) * src[idx + c] + strength * color[c];
            dst[idx + c] = static_cast<uint8_t>(std::clamp(static_cast<int>(blended), 0, 255));
        }

        if (channels == 4)
            dst[idx + 3] = src[idx + 3]; 
    }

    return result;
}

Image pixelate(const Image& image, int blockSize = 8) {
    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels = image.getChannels();

    Image result(width, height, channels);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int y = 0; y < height; y += blockSize) {
        for (int x = 0; x < width; x += blockSize) {
            int idx = (y * width + x) * channels;

            for (int dy = 0; dy < blockSize; ++dy) {
                for (int dx = 0; dx < blockSize; ++dx) {
                    int px = x + dx;
                    int py = y + dy;
                    if (px >= width || py >= height) continue;

                    int out_idx = (py * width + px) * channels;
                    for (int c = 0; c < channels; ++c) {
                        dst[out_idx + c] = src[idx + c];
                    }
                }
            }
        }
    }

    return result;
}


Image dither(const Image& image) {
    Image gray = to_grayscale(image);
    const int width = gray.getWidth();
    const int height = gray.getHeight();
    const int channels = 1;

    Image result(width, height, channels);
    uint8_t* dst = result.data();
    std::vector<float> buffer(gray.data(), gray.data() + width * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            float old_pixel = buffer[idx];
            uint8_t new_pixel = old_pixel < 128 ? 0 : 255;
            float error = old_pixel - new_pixel;
            dst[idx] = new_pixel;

            if (x + 1 < width)       buffer[idx + 1] += error * 7 / 16.0f;
            if (x > 0 && y + 1 < height) buffer[idx + width - 1] += error * 3 / 16.0f;
            if (y + 1 < height)      buffer[idx + width] += error * 5 / 16.0f;
            if (x + 1 < width && y + 1 < height) buffer[idx + width + 1] += error * 1 / 16.0f;
        }
    }

    return result;
}

Image vignette(const Image& image, float radius = 0.5f, float strength = 0.5f) {
    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels = image.getChannels();

    const float cx = width / 2.0f;
    const float cy = height / 2.0f;
    const float max_dist = std::sqrt(cx * cx + cy * cy);

    Image result(width, height, channels);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float dx = x - cx;
            float dy = y - cy;
            float dist = std::sqrt(dx * dx + dy * dy);
            float factor = 1.0f - strength * std::clamp((dist / (radius * max_dist)), 0.0f, 1.0f);

            int idx = (y * width + x) * channels;
            for (int c = 0; c < std::min(3, channels); ++c) {
                dst[idx + c] = static_cast<uint8_t>(std::clamp(static_cast<int>(src[idx + c] * factor), 0, 255));
            }
            if (channels == 4)
                dst[idx + 3] = src[idx + 3];
        }
    }

    return result;
}
