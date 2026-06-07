#include "roboperc/core/registry.hpp"

#include <algorithm>
#include <stdexcept>

namespace roboperc {

DetectorRegistry& DetectorRegistry::instance() {
    static DetectorRegistry reg;
    return reg;
}

bool DetectorRegistry::register_factory(const std::string& name,
                                        Factory factory) {
    for (auto& kv : factories_) {
        if (kv.first == name) {       // last registration wins
            kv.second = std::move(factory);
            return true;
        }
    }
    factories_.emplace_back(name, std::move(factory));
    return true;
}

std::unique_ptr<Detector> DetectorRegistry::create(
    const std::string& name, const DetectorConfig& cfg) const {
    for (const auto& kv : factories_) {
        if (kv.first == name) return kv.second(cfg);
    }
    throw std::out_of_range("no detector registered under name: " + name);
}

bool DetectorRegistry::contains(const std::string& name) const {
    return std::any_of(factories_.begin(), factories_.end(),
                       [&](const auto& kv) { return kv.first == name; });
}

std::vector<std::string> DetectorRegistry::names() const {
    std::vector<std::string> out;
    out.reserve(factories_.size());
    for (const auto& kv : factories_) out.push_back(kv.first);
    return out;
}

}  // namespace roboperc
