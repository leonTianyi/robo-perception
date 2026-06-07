// roboperc._native — pybind11 bridge exposing the C++ core to Python.
//
// This is the parity seam: the Python PyTorch / ONNX / TensorRT paths all call
// these functions for decode + pre/post, so they compare models, not pipelines.
// It also exposes the registry so a full C++ detector (engine + pre/post) can be
// driven from Python.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <opencv2/core.hpp>

#include "roboperc/core/coco_labels.hpp"
#include "roboperc/core/config.hpp"
#include "roboperc/core/detections.hpp"
#include "roboperc/core/detector.hpp"
#include "roboperc/core/registry.hpp"
#include "roboperc/detection/rfdetr/postprocess.hpp"
#include "roboperc/detection/rfdetr/preprocess.hpp"
#include "roboperc/io/decode.hpp"

namespace py = pybind11;
using namespace roboperc;

namespace {

// numpy (H,W,3) uint8 BGR -> cv::Mat (shares no ownership; copies into Mat).
cv::Mat numpy_to_mat_bgr(const py::array& arr) {
    auto a = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>(arr);
    if (a.ndim() != 3 || a.shape(2) != 3)
        throw std::invalid_argument("expected an (H,W,3) uint8 BGR array");
    const int h = static_cast<int>(a.shape(0));
    const int w = static_cast<int>(a.shape(1));
    cv::Mat m(h, w, CV_8UC3);
    std::memcpy(m.data, a.data(), static_cast<std::size_t>(h) * w * 3);
    return m;
}

// cv::Mat (H,W,3) uint8 BGR -> numpy (H,W,3) uint8.
py::array_t<uint8_t> mat_to_numpy_bgr(const cv::Mat& m) {
    py::array_t<uint8_t> out({m.rows, m.cols, 3});
    std::memcpy(out.mutable_data(), m.data,
                static_cast<std::size_t>(m.rows) * m.cols * 3);
    return out;
}

py::array_t<uint8_t> decode_image_py(const std::string& path) {
    return mat_to_numpy_bgr(io::decode_image(path));
}

// BGR HWC uint8 -> NCHW float32 (1,3,H,W), normalised exactly as the network expects.
py::array_t<float> preprocess_py(const py::array& image_bgr, int input_w,
                                 int input_h) {
    cv::Mat img = numpy_to_mat_bgr(image_bgr);
    detection::rfdetr::PreprocCfg cfg;
    cfg.input_w = input_w;
    cfg.input_h = input_h;
    std::vector<float> chw = detection::rfdetr::preprocess(img, cfg);

    py::array_t<float> out({1, 3, input_h, input_w});
    std::memcpy(out.mutable_data(), chw.data(), chw.size() * sizeof(float));
    return out;
}

// (.., num_queries, num_classes) logits + (.., num_queries, 4) boxes -> detections.
std::vector<Detection> postprocess_py(const py::array& logits_arr,
                                      const py::array& boxes_arr, int orig_w,
                                      int orig_h, float conf) {
    auto logits = py::array_t<float, py::array::c_style | py::array::forcecast>(
        logits_arr);
    auto boxes = py::array_t<float, py::array::c_style | py::array::forcecast>(
        boxes_arr);
    if (logits.ndim() < 2 || boxes.ndim() < 2)
        throw std::invalid_argument("logits/boxes must be at least 2-D");

    const int num_classes = static_cast<int>(logits.shape(logits.ndim() - 1));
    const int num_queries = static_cast<int>(logits.shape(logits.ndim() - 2));
    Detections d = detection::rfdetr::postprocess(
        logits.data(), boxes.data(), num_queries, num_classes, orig_w, orig_h,
        conf);
    return d.items;
}

}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "roboperc native core (decode + RF-DETR pre/post + registry)";

    py::class_<Detection>(m, "Detection")
        .def_readonly("x1", &Detection::x1)
        .def_readonly("y1", &Detection::y1)
        .def_readonly("x2", &Detection::x2)
        .def_readonly("y2", &Detection::y2)
        .def_readonly("score", &Detection::score)
        .def_readonly("class_id", &Detection::class_id)
        .def("__repr__", [](const Detection& d) {
            return "<Detection cls=" + std::to_string(d.class_id) + " score=" +
                   std::to_string(d.score) + ">";
        });

    py::class_<Detections>(m, "Detections")
        .def_readonly("items", &Detections::items)
        .def_readonly("image_w", &Detections::image_w)
        .def_readonly("image_h", &Detections::image_h)
        .def("__len__", &Detections::size);

    py::class_<DetectorConfig>(m, "DetectorConfig")
        .def(py::init<>())
        .def_readwrite("engine_path", &DetectorConfig::engine_path)
        .def_readwrite("conf_threshold", &DetectorConfig::conf_threshold)
        .def_readwrite("input_w", &DetectorConfig::input_w)
        .def_readwrite("input_h", &DetectorConfig::input_h);

    py::class_<Detector>(m, "Detector")
        .def("detect", [](Detector& self, const py::array& image_bgr) {
            cv::Mat img = numpy_to_mat_bgr(image_bgr);
            return self.detect(img);
        });

    // ── free functions: the shared pre/post + decode ──────────────────────────
    m.def("decode_image", &decode_image_py, py::arg("path"),
          "Decode an image file to a BGR HWC uint8 array.");
    m.def("preprocess", &preprocess_py, py::arg("image_bgr"),
          py::arg("input_w") = 560, py::arg("input_h") = 560,
          "BGR HWC uint8 -> NCHW float32 (1,3,H,W), RF-DETR normalisation.");
    m.def("postprocess", &postprocess_py, py::arg("logits"), py::arg("boxes"),
          py::arg("orig_w"), py::arg("orig_h"), py::arg("conf") = 0.35f,
          "Decode RF-DETR raw outputs into a list of Detection.");

    // ── registry ──────────────────────────────────────────────────────────────
    m.def("registered_detectors",
          [] { return DetectorRegistry::instance().names(); });
    m.def(
        "make_detector",
        [](const std::string& name, const DetectorConfig& cfg) {
            return DetectorRegistry::instance().create(name, cfg);
        },
        py::arg("name"), py::arg("config"),
        "Build a registered detector (e.g. 'rfdetr') from a DetectorConfig.");

    m.def("coco_label", &coco_label, py::arg("class_id"));
    m.attr("__version__") = "0.1.0";
}
