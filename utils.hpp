#include <cstdint>
#include <vector>
#include "deps/stb_image.h"
#include "deps/stb_image_write.h"
#include <cstring> 

class Image {
private:
    int width_, height_, channels_;
    std::vector<uint8_t> data_;

public:
    Image(int width, int height, int channels, std::vector<uint8_t> data) 
    : width_(width), height_(height), channels_(channels), data_(data) {}

    Image(int width, int height, int channels) : width_(width), height_
    (height), channels_(channels), data_(width * height * channels, 0) {}

    Image(int width, int height, int channels, const uint8_t* data_ptr)
    : width_(width), height_(height), channels_(channels),
      data_(data_ptr, data_ptr + width * height * channels) {}

    Image(const Image& other) : width_(other.width_), height_(other.height_), channels_(other.channels_), data_(other.data_) {} 
    Image(Image&& other) noexcept : width_(other.width_), height_(other.height_), channels_(other.channels_), data_(std::move(other.data_)) {}
    
    Image& operator=(const Image& other) {
        if (this != &other) {
            width_ = other.width_;
            height_ = other.height_;
            channels_ = other.channels_;
            data_ = other.data_;
        }
        return *this;
    }

    Image& operator=(Image&& other) noexcept {
        if (this != &other) {
            width_ = other.width_;
            height_ = other.height_;
            channels_ = other.channels_;
            data_ = std::move(other.data_);
        }
        return *this;
    }
    
    int getWidth() const { return width_; }
    int getHeight() const { return height_; }
    int getChannels() const { return channels_; }
    uint8_t* data() {return data_.data(); }
    const uint8_t* data() const { return data_.data(); }

    static Image load(const std::string& location, int desired_channels = 3) {
        int width, height, channels;
        uint8_t* image = stbi_load(location.c_str(), &width, &height, &channels, desired_channels);
        if (!image)
            throw std::runtime_error("Error! Failed to load image at " + location + ". Reason: " + stbi_failure_reason());
        std::vector<uint8_t> image_copy(image, image + (width * height * desired_channels));
        Image result(width, height, desired_channels, image_copy); 
        stbi_image_free(image);
        return result;
    }

    void save(const std::string& location) const {
        std::size_t dot_pos = location.rfind('.');
        if (dot_pos == std::string::npos)
            throw std::runtime_error("No file extension found in: " + location);

        std::string ext = location.substr(dot_pos);
        int stride_in_bytes = width_ * channels_; 

        if (ext == ".png") {
            if (!stbi_write_png(location.c_str(), width_, height_, channels_, data_.data(), stride_in_bytes))
                throw std::runtime_error("Failed to save PNG to " + location);
        }
        else if (ext == ".jpg" || ext == ".jpeg") {
            int quality = 100;
            if (!stbi_write_jpg(location.c_str(), width_, height_, channels_, data_.data(), quality))
                throw std::runtime_error("Failed to save JPG to " + location);
        }
        else if (ext == ".bmp") {
            if (!stbi_write_bmp(location.c_str(), width_, height_, channels_, data_.data()))
                throw std::runtime_error("Failed to save BMP to " + location);
        }
        else {
            throw std::runtime_error("Unsupported file extension: " + ext);
        }
    }

};