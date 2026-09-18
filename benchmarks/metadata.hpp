#pragma once
#include "kestrel/model.hpp"
#include <ctime>
#include <set>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <sys/utsname.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
namespace kestrel {
inline std::string timestamp() {
    auto now = std::time(nullptr);
    std::tm utc{};
    gmtime_r(&now, &utc);
    std::ostringstream out;
    out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}
#ifdef __APPLE__
inline std::string sys_string(const char *key) {
    size_t size = 0;
    if (sysctlbyname(key, nullptr, &size, nullptr, 0) || !size) return "unknown";
    std::string out(size, '\0');
    if (sysctlbyname(key, out.data(), &size, nullptr, 0)) return "unknown";
    while (!out.empty() && out.back() == '\0') out.pop_back();
    return out;
}
#endif
inline Json environment() {
    Json j{{"os", "unknown"}, {"machine", "unknown"}, {"chip", "unknown"}};
    struct utsname u{};
    if (!uname(&u)) {
        j["os"] = std::string(u.sysname) + " " + u.release;
        j["machine"] = u.machine;
    }
#ifdef __APPLE__
    j["os"] = "macOS " + sys_string("kern.osproductversion");
    j["os_build"] = sys_string("kern.osversion");
    j["machine"] = sys_string("hw.model");
    j["chip"] = sys_string("machdep.cpu.brand_string");
    uint64_t memory = 0;
    size_t size = sizeof(memory);
    if (!sysctlbyname("hw.memsize", &memory, &size, nullptr, 0)) j["memory_bytes"] = memory;
    for (auto key : {"hw.physicalcpu", "hw.logicalcpu", "hw.perflevel0.physicalcpu", "hw.perflevel1.physicalcpu"}) {
        int cores = 0;
        size = sizeof(cores);
        if (!sysctlbyname(key, &cores, &size, nullptr, 0)) j[key] = cores;
    }
#endif
    return j;
}
inline Json model_metadata(const std::filesystem::path &dir, const Config &c) {
    Json dtypes = Json::array();
    size_t parameters = 0;
    // Read the exact tensor headers, including sharded checkpoints; no dtype inference from config.
    std::set<std::string> shards;
    auto index = dir / "model.safetensors.index.json";
    if (std::filesystem::exists(index)) {
        auto mapping = read_json(index).at("weight_map");
        for (auto &[name, file] : mapping.items()) shards.insert(file.get<std::string>());
    } else shards.insert("model.safetensors");
    for (const auto &shard : shards) {
        SafeTensors file(dir / shard);
        for (auto &[name, tensor] : file.tensors()) {
            parameters += tensor.size();
            std::string dtype = tensor.dtype() == DType::f32 ? "F32" : tensor.dtype() == DType::f16 ? "F16" : "BF16";
            if (std::find(dtypes.begin(), dtypes.end(), dtype) == dtypes.end()) dtypes.push_back(dtype);
        }
    }
    std::string name = dir.filename().string();
    if (std::filesystem::exists(dir / "download-manifest.json")) name = read_json(dir / "download-manifest.json").value("repository", name);
    return {{"name", name}, {"path", dir.string()}, {"hidden_size", c.hidden},
        {"layers", c.layers}, {"attention_heads", c.heads}, {"kv_heads", c.kv_heads},
        {"vocab_size", c.vocab}, {"intermediate_size", c.intermediate},
        {"stored_parameter_count", parameters}, {"weight_dtypes", dtypes},
        {"weight_dtype", dtypes.size() == 1 ? dtypes[0].get<std::string>() : "mixed"}};
}
} // namespace kestrel
