// normalize: Scale pixel intensities to a standard range (e.g. 0–1 or 0–255)
// clamp: Limit pixel values to lie within a specified min and max range
// histogram_equalization: Enhance overall contrast by spreading out intensity distribution
// adaptive_histogram_equalization: Enhance local contrast via histogram equalization in small regions (CLAHE)
// threshold: Convert to a binary black-and-white image based on a global intensity cutoff
// adaptive_threshold: Generate a binary image using a threshold that adapts to local brightness
/////////////////////////////////////////////////////////////////////////////////////////////////////
// erode: Morphologically erode the image (shrink bright regions and remove small spots)
// dilate: Morphologically dilate the image (expand bright regions and fill in small holes)
// erode_dilate: Apply morphological opening (erode then dilate) to remove small objects/noise
// dilate_erode: Apply morphological closing (dilate then erode) to fill small gaps or holes
// flood_fill: Fill a contiguous region with a given color, starting from a seed pixel

#include "utils.hpp"


std::vector<float> normalizeImage(const Image& image) {
    const uint8_t* src = image.data();
    size_t total = image.getWidth() * image.getHeight() * image.getChannels();

    std::vector<float> out(total);
    for (size_t i = 0; i < total; ++i)
        out[i] = src[i] / 255.0f;

    return out;
}

Image denormalizeImage(const std::vector<float>& data, int width, int height, int channels) {
    Image result(width, height, channels);
    uint8_t* dst = result.data();

    for (size_t i = 0; i < data.size(); ++i)
        dst[i] = std::clamp(int(data[i] * 255.0f), 0, 255);

    return result;
}

Image clampImage(const Image& image, const std::vector<int>& rgb_clamp) {
    if (rgb_clamp.size() != 6) {
        throw std::runtime_error(
            "Arguments must be {Red Min, Red Max, Green Min, Green Max, Blue Min, Blue Max}");
    }

    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    Image result(width, height, channels);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int p = 0; p < width * height; ++p) {
        int idx = p * channels;

        dst[idx + 0] = static_cast<uint8_t>(std::clamp<int>(static_cast<int>(src[idx + 0]), rgb_clamp[0], rgb_clamp[1]));
        dst[idx + 1] = static_cast<uint8_t>(std::clamp<int>(static_cast<int>(src[idx + 1]), rgb_clamp[2], rgb_clamp[3]));
        dst[idx + 2] = static_cast<uint8_t>(std::clamp<int>(static_cast<int>(src[idx + 2]), rgb_clamp[4], rgb_clamp[5]));
        

        if (channels == 4) {
            dst[idx + 3] = src[idx + 3]; // Preserve alpha as-is
        }
    }

    return result;
}

Image histogram_equalization(const Image& image) {
    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    if (channels != 1) {
        throw std::runtime_error("Histogram equalization requires a grayscale image.");
    }

    const uint8_t* src = image.data();
    Image result(width, height, 1);
    uint8_t* dst = result.data();

    int hist[256] = {0};
    for (int i = 0; i < width * height; ++i)
        hist[src[i]]++;

    int cdf[256] = {0};
    cdf[0] = hist[0];
    for (int i = 1; i < 256; ++i)
        cdf[i] = cdf[i - 1] + hist[i];

    int cdf_min = *std::find_if(cdf, cdf + 256, [](int v) { return v > 0; });
    int total = width * height;
    uint8_t equalize_map[256];
    for (int i = 0; i < 256; ++i) {
        equalize_map[i] = std::clamp((cdf[i] - cdf_min) * 255 / (total - cdf_min), 0, 255);
    }

    for (int i = 0; i < width * height; ++i)
        dst[i] = equalize_map[src[i]];

    return result;
}
Image adaptive_histogram_equalization(const Image& image, int block_size = 8) {
    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    if (channels != 1) {
        throw std::runtime_error("Adaptive histogram equalization requires a grayscale image.");
    }

    Image result(width, height, 1);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int y = 0; y < height; y += block_size) {
        for (int x = 0; x < width; x += block_size) {
            int block_w = std::min(block_size, width - x);
            int block_h = std::min(block_size, height - y);

            int hist[256] = {0};
            for (int j = 0; j < block_h; ++j) {
                for (int i = 0; i < block_w; ++i) {
                    int px = (y + j) * width + (x + i);
                    hist[src[px]]++;
                }
            }

            int cdf[256] = {0};
            cdf[0] = hist[0];
            for (int i = 1; i < 256; ++i)
                cdf[i] = cdf[i - 1] + hist[i];

            int block_total = block_w * block_h;
            int cdf_min = *std::find_if(cdf, cdf + 256, [](int v) { return v > 0; });

            uint8_t equalize_map[256];
            for (int i = 0; i < 256; ++i) {
                equalize_map[i] = std::clamp((cdf[i] - cdf_min) * 255 / (block_total - cdf_min), 0, 255);
            }

            for (int j = 0; j < block_h; ++j) {
                for (int i = 0; i < block_w; ++i) {
                    int px = (y + j) * width + (x + i);
                    dst[px] = equalize_map[src[px]];
                }
            }
        }
    }

    return result;
}

Image threshold(const Image& image, uint8_t cutoff = 128) {
    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    if (channels != 1) {
        throw std::runtime_error("Thresholding requires a grayscale image.");
    }

    Image result(width, height, 1);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int i = 0; i < width * height; ++i)
        dst[i] = src[i] >= cutoff ? 255 : 0;

    return result;
}

Image adaptive_threshold(const Image& image, int block_size = 15, int offset = 10) {
    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    if (channels != 1) {
        throw std::runtime_error("Adaptive thresholding requires a grayscale image.");
    }

    Image result(width, height, 1);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    int half = block_size / 2;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int sum = 0;
            int count = 0;

            for (int j = -half; j <= half; ++j) {
                for (int i = -half; i <= half; ++i) {
                    int nx = std::clamp(x + i, 0, width - 1);
                    int ny = std::clamp(y + j, 0, height - 1);
                    sum += src[ny * width + nx];
                    count++;
                }
            }

            int local_mean = sum / count;
            int idx = y * width + x;
            dst[idx] = src[idx] >= (local_mean - offset) ? 255 : 0;
        }
    }

    return result;
}
