// Minimal assert-based test for the detector registry. No test framework needed
// — kept dependency-free so `ctest` works out of the box on the Jetson.
#include <cassert>
#include <iostream>
#include <memory>

#include "roboperc/core/coco_labels.hpp"
#include "roboperc/core/registry.hpp"

using namespace roboperc;

namespace {

// A trivial Detector used only to exercise the registry.
class DummyDetector : public Detector {
public:
    explicit DummyDetector(const DetectorConfig& cfg) : cfg_(cfg) {}
    Detections detect(const cv::Mat&) override {
        Detections d;
        d.items.push_back(Detection{0, 0, 1, 1, cfg_.conf_threshold, 1});
        return d;
    }
private:
    DetectorConfig cfg_;
};

}  // namespace

REGISTER_DETECTOR("dummy", DummyDetector)

int main() {
    auto& reg = DetectorRegistry::instance();

    assert(reg.contains("dummy"));
    assert(!reg.contains("does-not-exist"));

    DetectorConfig cfg;
    cfg.conf_threshold = 0.5f;
    auto det = reg.create("dummy", cfg);
    assert(det != nullptr);

    cv::Mat dummy_img;  // detect() ignores it
    Detections out = det->detect(dummy_img);
    assert(out.size() == 1);
    assert(out.items[0].score == 0.5f);
    assert(out.items[0].class_id == 1);

    // Unknown name must throw.
    bool threw = false;
    try {
        reg.create("nope", cfg);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);

    // Label table sanity (DETR COCO indexing).
    assert(coco_label(1) == "person");
    assert(coco_label(6) == "bus");

    std::cout << "test_registry: OK (" << reg.names().size()
              << " detector(s) registered)\n";
    return 0;
}
