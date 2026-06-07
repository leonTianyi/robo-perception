// RF-DETR preprocessing — owned by the adapter (each detector owns its pre/post).
//
// Must match python/roboperc/imageproc.py (which calls THIS code via the binding)
// so every backend feeds the network identical pixels:
//   resize(W,H, INTER_LINEAR) -> BGR2RGB -> /255 -> (x-mean)/std -> HWC2CHW
#pragma once

#include <vector>

#include <opencv2/imgproc.hpp>

namespace roboperc::detection::rfdetr {

struct PreprocCfg {
    int input_w = 560;
    int input_h = 560;
    float mean[3] = {0.485f, 0.456f, 0.406f};   // ImageNet, RGB order
    float std[3]  = {0.229f, 0.224f, 0.225f};   // ImageNet, RGB order
};

// BGR uint8 HWC image -> contiguous CHW float32 buffer of size 3*H*W.
inline std::vector<float> preprocess(const cv::Mat& img_bgr,
                                     const PreprocCfg& c) {
    cv::Mat resized;
    cv::resize(img_bgr, resized, cv::Size(c.input_w, c.input_h), 0, 0,
               cv::INTER_LINEAR);

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    const int H = c.input_h, W = c.input_w;
    std::vector<float> chw(static_cast<std::size_t>(3) * H * W);
    for (int y = 0; y < H; ++y) {
        const cv::Vec3f* row = rgb.ptr<cv::Vec3f>(y);
        for (int x = 0; x < W; ++x) {
            for (int ch = 0; ch < 3; ++ch) {
                chw[static_cast<std::size_t>(ch) * H * W + y * W + x] =
                    (row[x][ch] - c.mean[ch]) / c.std[ch];
            }
        }
    }
    return chw;
}

}  // namespace roboperc::detection::rfdetr
