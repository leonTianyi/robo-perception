// run_inference — single-image detection via the registry + Detector contract.
//
//   run_inference <engine> <image> [out.jpg] [detector=rfdetr] [conf=0.35]
//
// Builds the detector by name from a DetectorConfig (the model-swap lever), runs
// it, draws boxes, and prints detections. Decode goes through io/decode so the
// pixels match the Python paths.
#include <iostream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "roboperc/core/coco_labels.hpp"
#include "roboperc/core/config.hpp"
#include "roboperc/core/registry.hpp"
#include "roboperc/io/decode.hpp"

using namespace roboperc;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0]
                  << " <engine> <image> [out.jpg] [detector=rfdetr] [conf]\n"
                  << "registered detectors:";
        for (const auto& n : DetectorRegistry::instance().names())
            std::cerr << " " << n;
        std::cerr << "\n";
        return 1;
    }
    DetectorConfig cfg;
    cfg.engine_path = argv[1];
    const std::string image_path = argv[2];
    const std::string out_path = argc > 3 ? argv[3] : "run_inference_out.jpg";
    const std::string name = argc > 4 ? argv[4] : "rfdetr";
    if (argc > 5) cfg.conf_threshold = std::stof(argv[5]);

    cv::Mat img = io::decode_image(image_path);

    auto detector = DetectorRegistry::instance().create(name, cfg);
    Detections dets = detector->detect(img);

    for (const auto& d : dets.items) {
        cv::rectangle(img, cv::Point((int)d.x1, (int)d.y1),
                      cv::Point((int)d.x2, (int)d.y2), cv::Scalar(0, 255, 0), 2);
        const std::string label =
            coco_label(d.class_id) + " " + cv::format("%.2f", d.score);
        cv::putText(img, label, cv::Point((int)d.x1, std::max(0, (int)d.y1 - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite(out_path, img);

    std::cout << "[" << name << "] " << dets.size() << " detections -> "
              << out_path << "\n";
    for (const auto& d : dets.items)
        std::cout << "  " << coco_label(d.class_id) << "  "
                  << cv::format("%.3f", d.score) << "\n";
    return 0;
}
