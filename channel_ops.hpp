// SUNDAY
// swap_red_blue: Swap the red and blue color channels of the image // DONE - NAIVE
// rotate_rgb: Rotate the color channels (R→G, G→B, B→R cycle) // DONE - NAIVE
// replace_color: Replace all pixels of a specified color with another color // DONE - NAIVE 
// isolate_channel: Extract a single color channel from the image (set other channels to zero) // DONE - NAIVE
// split_channels: Split the image into separate images for each color channel // DONE - NAIVE
// merge_channels: Combine multiple single-channel images into one multi-channel image // DONE - NAIVE
// invert_channel: Invert the values of one color channel (leave other channels unchanged) // DONE - NAIVE
// fill_channel: Set all pixels of a given channel to a constant value // DONE - NAIVE
 
#include "utils.hpp"

inline Image invert_colors(const Image& image) {
    if (image.getChannels() < 3) {
        throw std::runtime_error("Image must have at least 3 channels for invert_colors transformation.");
    }

    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    Image result(width, height, channels);
    
    const uint8_t* source = image.data();
    uint8_t* destination = result.data();

    const int total_pixels = width * height;

    for (int i = 0; i < total_pixels; i++) {
        int idx = i * channels;
        destination[idx] = 255 - source[idx];
        if (channels >= 2) destination[idx+1] = 255 - source[idx+1];
        if (channels >= 3) destination[idx+2] = 255 - source[idx+2];
        if (channels == 4) destination[idx+3] = 255 - source[idx+3];
    }

    return result;
}


inline Image swap_red_blue(const Image& image) {
    if (image.getChannels() < 3) {
        throw std::runtime_error("Image must have at least 3 channels for swap red/blue transformation.");
    }

    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    Image result(width, height, channels);
    
    const uint8_t* source = image.data();
    uint8_t* destination = result.data();

    const int total_pixels = width * height;

    for (int i = 0; i < total_pixels; i++) {
        int idx = i * channels;
        destination[idx] = source[idx + 2];
        destination[idx + 1] = source[idx + 1];
        destination[idx + 2] = source[idx];
        if (channels == 4) destination[idx + 3] = source[idx + 3];
    }

    return result;
}

inline Image swap_green_blue(const Image& image) {
    if (image.getChannels() < 3) {
        throw std::runtime_error("Image must have at least 3 channels for swap red/blue transformation.");
    }

    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    Image result(width, height, channels);
    
    const uint8_t* source = image.data();
    uint8_t* destination = result.data();

    const int total_pixels = width * height;

    for (int i = 0; i < total_pixels; i++) {
        int idx = i * channels;
        destination[idx] = source[idx];
        destination[idx + 1] = source[idx + 2];
        destination[idx + 2] = source[idx + 1];
        if (channels == 4) destination[idx + 3] = source[idx + 3];
    }

    return result;
}

inline Image swap_red_green(const Image& image) {
    if (image.getChannels() < 3) {
        throw std::runtime_error("Image must have at least 3 channels for swap red/blue transformation.");
    }

    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    Image result(width, height, channels);
    
    const uint8_t* source = image.data();
    uint8_t* destination = result.data();

    const int total_pixels = width * height;

    for (int i = 0; i < total_pixels; i++) {
        int idx = i * channels;
        destination[idx] = source[idx + 1];
        destination[idx + 1] = source[idx];
        destination[idx + 2] = source[idx + 2];
        if (channels == 4) destination[idx + 3] = source[idx + 3];
    }

    return result;
}


inline Image rotate_rgb_1st(const Image& image) {
    if (image.getChannels() < 3) {
        throw std::runtime_error("Image must have at least 3 channels for swap red/blue transformation.");
    }

    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    Image result(width, height, channels);
    
    const uint8_t* source = image.data();
    uint8_t* destination = result.data();

    const int total_pixels = width * height;

    for (int i = 0; i < total_pixels; i++) {
        int idx = i * channels;
        destination[idx] = source[idx + 1];
        destination[idx + 1] = source[idx + 2];
        destination[idx + 2] = source[idx];
        if (channels == 4) destination[idx + 3] = source[idx + 3];
    }

    return result;
}

inline Image rotate_rgb_2nd(const Image& image) {
    if (image.getChannels() < 3) {
        throw std::runtime_error("Image must have at least 3 channels for swap red/blue transformation.");
    }

    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    Image result(width, height, channels);
    
    const uint8_t* source = image.data();
    uint8_t* destination = result.data();

    const int total_pixels = width * height;

    for (int i = 0; i < total_pixels; i++) {
        int idx = i * channels;
        destination[idx] = source[idx + 2];
        destination[idx + 1] = source[idx];
        destination[idx + 2] = source[idx + 1];
        if (channels == 4) destination[idx + 3] = source[idx + 3];
    }

    return result;
}


inline Image replace_color(const Image& image, const std::vector<uint8_t>& from_color, const std::vector<uint8_t>& to_color) {
    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    if (from_color.size() != channels || to_color.size() != channels) {
        throw std::runtime_error("Color vectors must match image channel count.");
    }

    Image result(width, height, channels);

    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    const int total_pixels = width * height;

    for (int i = 0; i < total_pixels; i++) {
        int idx = i * channels;

        bool match = true;
        for (int c = 0; c < channels; ++c) {
            if (src[idx + c] != from_color[c]) {
                match = false;
                break;
            }
        }

        if (match) {
            for (int c = 0; c < channels; ++c)
                dst[idx + c] = to_color[c];
        } else {
            for (int c = 0; c < channels; ++c)
                dst[idx + c] = src[idx + c];
        }
    }

    return result;
}

inline Image isolate_channel(const Image& image, int channel_idx) {
    if (image.getChannels() < 3) {
        throw std::runtime_error("Image must have at least 3 channels to isolate a color channel.");
    }

    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    if (channel_idx < 0 || channel_idx >= 3) {
        throw std::runtime_error("channel_idx must be 0 (Red), 1 (Green), or 2 (Blue).");
    }

    Image result(width, height, channels);

    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    const int total_pixels = width * height;

    for (int i = 0; i < total_pixels; ++i) {
        int idx = i * channels;

        for (int c = 0; c < 3; ++c) {
            dst[idx + c] = (c == channel_idx) ? src[idx + c] : 0;
        }

        if (channels == 4) {
            dst[idx + 3] = src[idx + 3]; 
        }
    }

    return result;
}
inline std::vector<Image> split_channels(const Image& image) {
    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    std::vector<Image> channel_images;
    channel_images.reserve(channels);

    const uint8_t* src = image.data();

    for (int c = 0; c < channels; ++c) {
        Image channel_image(width, height, 1);
        uint8_t* dst = channel_image.data();

        for (int i = 0; i < width * height; ++i) {
            dst[i] = src[i * channels + c];
        }

        channel_images.push_back(std::move(channel_image));
    }

    return channel_images;
}

inline Image merge_channels(const std::vector<Image>& channels) {
    if (channels.empty())
        throw std::runtime_error("No input channels to merge.");

    int width = channels[0].getWidth();
    int height = channels[0].getHeight();
    int num_channels = channels.size();

    for (const auto& ch : channels) {
        if (ch.getWidth() != width || ch.getHeight() != height || ch.getChannels() != 1)
            throw std::runtime_error("All channels must be single-channel and have the same dimensions.");
    }

    Image result(width, height, num_channels);
    uint8_t* dst = result.data();

    for (int i = 0; i < width * height; ++i) {
        for (int c = 0; c < num_channels; ++c) {
            dst[i * num_channels + c] = channels[c].data()[i];
        }
    }

    return result;
}

inline Image invert_channel(const Image& image, int channel_idx) {
    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    if (channel_idx < 0 || channel_idx >= channels)
        throw std::runtime_error("Invalid channel index for inversion.");

    Image result(width, height, channels);

    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    const int total_pixels = width * height;

    for (int i = 0; i < total_pixels; ++i) {
        int idx = i * channels;

        for (int c = 0; c < channels; ++c) {
            dst[idx + c] = (c == channel_idx) ? 255 - src[idx + c] : src[idx + c];
        }
    }

    return result;
}

inline Image fill_channel(const Image& image, int channel_idx, uint8_t value) {
    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    if (channel_idx < 0 || channel_idx >= channels)
        throw std::runtime_error("Invalid channel index for fill.");

    Image result(width, height, channels);

    const uint8_t* src = image.data();
    uint8_t* dst = result.data();

    const int total_pixels = width * height;

    for (int i = 0; i < total_pixels; ++i) {
        int idx = i * channels;

        for (int c = 0; c < channels; ++c) {
            dst[idx + c] = (c == channel_idx) ? value : src[idx + c];
        }
    }

    return result;
}
