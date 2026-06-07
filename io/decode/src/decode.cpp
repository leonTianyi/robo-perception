#include "roboperc/io/decode.hpp"

#include <stdexcept>

#include <opencv2/imgcodecs.hpp>

namespace roboperc::io {

cv::Mat decode_image(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);  // always BGR 3-channel
    if (img.empty())
        throw std::runtime_error("decode_image: cannot read image: " + path);
    return img;
}

}  // namespace roboperc::io
