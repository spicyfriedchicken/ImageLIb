// adjust_brightness: Increase or decrease the overall brightness of the image // DONE - NAIVE
// adjust_contrast: Increase or decrease the difference between light and dark areas // DONE - NAIVE
// gamma_correction: Apply gamma correction to adjust brightness in a non-linear way // DONE - NAIVE
// adjust_exposure: Make the image lighter or darker, simulating camera exposure change // DONE - NAIVE
// adjust_saturation: Increase or decrease the intensity of all colors // DONE - NAIVE
// adjust_hue: Shift all colors by a given offset around the color wheel // DONE - NAIVE
// auto_contrast: Automatically stretch image intensities to enhance contrast // DONE - NAIVE
// ---------------------------------------------------------------------------------------------------
// apply_alpha_mask(mask)	Applies an alpha mask (grayscale or binary) to adjust transparency
// extract_alpha()	Extracts the alpha channel as a separate grayscale image
// composite_over(background)	Alpha-blends the image onto another background
// premultiply_alpha()	Converts RGBA to pre-multiplied alpha (for compositing)
// unpremultiply_alpha()	Reverses premultiplied alpha
// alpha_to_mask()	Converts alpha transparency to a binary mask
#include "utils.hpp"

inline Image adjust_brightness(const Image& image, float brightness) {
    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels = image.getChannels();

    if (channels < 1)
        throw std::runtime_error("Image must have at least one channel");

    Image result(width, height, channels);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    int total_pixels = width*height;

    for (int p = 0; p < total_pixels; p++) {
        for (int c = 0; c < channels; c++) {
            if (c < 3) {
                dst[p * channels + c] = std::clamp(int(src[p * channels + c] * brightness), 0, 255);
            }
        }
    }

    return result;
}

inline Image adjust_contrast(const Image& image, const float contrast_const) {
    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels = image.getChannels();

    if (channels < 1)
        throw std::runtime_error("Image must have at least one channel");
    
    Image result(width, height, channels);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    int pixel_size = width*height;

    for (int i = 0; i < pixel_size; i++) {
        for (int c = 0; c < channels; c++) {
            if (c < 3) {
                dst[i * channels  + c] = std::clamp(static_cast<int>((float(src[i * channels + c]) - 128.0f) * contrast_const + 128.0f), 0, 255);
            }
        }
    }
    return result;
}

inline Image adjust_gamma(const Image& image, float gamma_const) {
    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();
    
    if (channels < 1)
        throw std::runtime_error("Image must have at least one channel.\n");
    
    Image result(width, height, channels);

    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    int pixel_count = width * height;

    for (int p = 0; p < pixel_count; p++) {
        for (int c = 0; c < channels; c++) {
            if (c < 3)
                dst[p * channels + c] = std::clamp(int((std::pow((src[p * channels + c] / 255.0), 1.0 / gamma_const))*255.0), 0, 255);
        }
    }

    return result;
}

inline Image adjust_exposure(const Image& image, int exposure_factor) {
    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels = image.getChannels();

    if (channels < 1)
        throw std::runtime_error("Image must have at least one channel.\n");

    Image result(width, height, channels);
    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    int total_pixels = width*height;
    float exposure_multiplier = std::pow(2.0f, exposure_factor);

    for (int p = 0; p < total_pixels; p++) {
        for (int c = 0; c < channels; c++) {
            if (c < 3) {
                dst[p * channels + c] = std::clamp(int(src[p * channels + c] * exposure_multiplier), 0, 255);
            }
        }
    }

    return result;
}

inline Image adjust_saturation(const Image& image, float saturation_const) {
    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels =  image.getChannels();

    if (channels < 3) 
        throw std::runtime_error("Image must have at least three channels.\n");

    Image result(width, height, channels);

    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    int pixel_count = width * height;

    for (int p = 0; p < pixel_count; p++) {
        float gray = 0.299*src[p*channels] + 0.587*src[p*channels+1] + 0.114*src[p*channels+2];
        dst[p * channels] = std::clamp(static_cast<int>(gray + (src[p*channels] - gray) * saturation_const), 0, 255);
        dst[p * channels + 1] = std::clamp(static_cast<int>(gray + (src[p*channels + 1] - gray) * saturation_const), 0, 255);
        dst[p * channels + 2] = std::clamp(static_cast<int>(gray + (src[p*channels + 2] - gray) * saturation_const), 0, 255);
        if (channels == 4) dst[p * channels + 3] = src[p * channels + 3];
    }

    return result;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline uint8_t clamp(float x) {
    return static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, x)));
}

Image adjust_hue(Image& image, float hue_shift_degrees) {
    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels = image.getChannels();

    if (channels < 3)
        throw std::runtime_error("Image must have at least 3 channels for hue adjustment.");

    Image result(width, height, channels);

    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    for (int i = 0; i < width * height; ++i) {
        float r = src[i * channels]     / 255.0f;
        float g = src[i * channels + 1] / 255.0f;
        float b = src[i * channels + 2] / 255.0f;

        // RGB → HSV
        float cmax = std::max({r, g, b});
        float cmin = std::min({r, g, b});
        float delta = cmax - cmin;

        float h = 0;
        if (delta > 0.00001f) {
            if (cmax == r)
                h = fmodf((60 * ((g - b) / delta) + 360), 360.0f);
            else if (cmax == g)
                h = fmodf((60 * ((b - r) / delta) + 120), 360.0f);
            else
                h = fmodf((60 * ((r - g) / delta) + 240), 360.0f);
        }

        float s = (cmax == 0.0f) ? 0.0f : delta / cmax;
        float v = cmax;

        h = fmodf(h + hue_shift_degrees, 360.0f);

        float c = v * s;
        float x = c * (1 - fabsf(fmodf(h / 60.0f, 2) - 1));
        float m = v - c;
        float r1, g1, b1;

        if (h < 60)       { r1 = c; g1 = x; b1 = 0; }
        else if (h < 120) { r1 = x; g1 = c; b1 = 0; }
        else if (h < 180) { r1 = 0; g1 = c; b1 = x; }
        else if (h < 240) { r1 = 0; g1 = x; b1 = c; }
        else if (h < 300) { r1 = x; g1 = 0; b1 = c; }
        else              { r1 = c; g1 = 0; b1 = x; }

        dst[i * channels]     = clamp((r1 + m) * 255.0f);
        dst[i * channels + 1] = clamp((g1 + m) * 255.0f);
        dst[i * channels + 2] = clamp((b1 + m) * 255.0f);

        if (channels == 4)
            dst[i * channels + 3] = src[i * channels + 3];
    }

    return result;
}


Image auto_contrast(const Image& image) {
    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels = image.getChannels();

    Image result(width, height, channels);

    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    int min_val[3] = {255, 255, 255};
    int max_val[3] = {0, 0, 0};

    for (int i = 0; i < width * height; ++i) {
        for (int c = 0; c < std::min(3, channels); ++c) {
            uint8_t val = src[i * channels + c];
            min_val[c] = std::min(min_val[c], (int)val);
            max_val[c] = std::max(max_val[c], (int)val);
        }
    }

    for (int i = 0; i < width * height; ++i) {
        for (int c = 0; c < std::min(3, channels); ++c) {
            int min_c = min_val[c];
            int max_c = max_val[c];
            uint8_t val = src[i * channels + c];

            if (max_c != min_c) {
                dst[i * channels + c] = static_cast<uint8_t>(
                    (val - min_c) * 255.0f / (max_c - min_c)
                );
            } else {
                dst[i * channels + c] = val;
            }
        }

        if (channels == 4)
            dst[i * channels + 3] = src[i * channels + 3];
    }

    return result;
}
