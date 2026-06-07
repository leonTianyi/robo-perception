// The one source of truth for image decode, shared C++/Python (via bindings).
//
// Keeping decode in a single place is what lets the Python and C++ paths claim
// they ran on the *same* pixels. Adapters and apps must go through here rather
// than calling cv::imread directly.
#pragma once

#include <string>

#include <opencv2/core.hpp>

namespace roboperc::io {

// Decode an image file from disk into a BGR uint8 HWC cv::Mat.
// Throws std::runtime_error if the file cannot be read/decoded.
cv::Mat decode_image(const std::string& path);

}  // namespace roboperc::io
