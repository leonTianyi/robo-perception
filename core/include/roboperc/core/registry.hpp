// Detector registry — the model-swap lever.
//
// Each adapter registers a factory under a name (e.g. "rfdetr"); a config then
// selects which one to build. This is what lets you hold dataset + pre/post
// constant and swap the model/runtime without touching call sites.
//
// Adapters self-register at static-init time via the REGISTER_DETECTOR macro, so
// linking the adapter library is all it takes to make a name available. NOTE:
// because registration happens in a translation unit that may carry no other
// referenced symbols, link adapter libraries with whole-archive (see the
// detection CMakeLists) or depend on them from an executable directly.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "roboperc/core/config.hpp"
#include "roboperc/core/detector.hpp"

namespace roboperc {

class DetectorRegistry {
public:
    using Factory =
        std::function<std::unique_ptr<Detector>(const DetectorConfig&)>;

    // Process-wide singleton.
    static DetectorRegistry& instance();

    // Register a factory under `name`. Returns true (so it can be used to
    // initialise a static bool). Last registration for a name wins.
    bool register_factory(const std::string& name, Factory factory);

    // Build a detector by name. Throws std::out_of_range if name is unknown.
    std::unique_ptr<Detector> create(const std::string& name,
                                     const DetectorConfig& cfg) const;

    bool contains(const std::string& name) const;
    std::vector<std::string> names() const;

private:
    DetectorRegistry() = default;
    std::vector<std::pair<std::string, Factory>> factories_;
};

}  // namespace roboperc

// Convenience: register a Detector subclass constructible from DetectorConfig.
// The variable name is made unique via __LINE__ so `Type` may be namespace-
// qualified (token-pasting a qualified name would be ill-formed).
#define ROBOPERC_CAT_(a, b) a##b
#define ROBOPERC_CAT(a, b) ROBOPERC_CAT_(a, b)
#define REGISTER_DETECTOR(name_literal, Type)                                  \
    namespace {                                                                \
    const bool ROBOPERC_CAT(_roboperc_reg_, __LINE__) =                        \
        ::roboperc::DetectorRegistry::instance().register_factory(             \
            name_literal, [](const ::roboperc::DetectorConfig& cfg) {          \
                return std::unique_ptr<::roboperc::Detector>(new Type(cfg));   \
            });                                                                \
    }
