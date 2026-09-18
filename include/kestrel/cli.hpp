#pragma once
#include "kestrel/sampling.hpp"
#include <charconv>
#include <map>
#include <set>
#include <sstream>
namespace kestrel {
class Args {
  public:
    Args(int argc, char **argv, std::set<std::string> values, std::set<std::string> flags) {
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (options_.contains(key))
                throw std::invalid_argument("duplicate option: " + key);
            if (flags.contains(key))
                options_[key] = "true";
            else if (values.contains(key)) {
                if (++i >= argc)
                    throw std::invalid_argument("missing value for " + key);
                options_[key] = argv[i];
            } else
                throw std::invalid_argument("unknown option: " + key);
        }
    }
    bool has(const std::string &key) const {
        return options_.contains(key);
    }
    std::string get(const std::string &key, std::string fallback = "") const {
        auto i = options_.find(key);
        return i == options_.end() ? fallback : i->second;
    }
    size_t number(const std::string &key, size_t fallback) const {
        if (!has(key))
            return fallback;
        auto str = get(key);
        size_t value = 0;
        auto [end, ec] = std::from_chars(str.data(), str.data() + str.size(), value);
        if (ec != std::errc() || end != str.data() + str.size())
            throw std::invalid_argument("invalid unsigned integer for " + key);
        return value;
    }
    float real(const std::string &key, float fallback) const {
        if (!has(key))
            return fallback;
        auto str = get(key);
        size_t n = 0;
        float value = std::stof(str, &n);
        if (n != str.size())
            throw std::invalid_argument("invalid number for " + key);
        return value;
    }

  private:
    std::map<std::string, std::string> options_;
};
inline std::vector<int> parse_tokens(const std::string &text) {
    std::vector<int> result;
    std::istringstream in(text);
    std::string s;
    while (std::getline(in, s, ',')) {
        int value;
        auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
        if (ec != std::errc() || end != s.data() + s.size() || value < 0)
            throw std::invalid_argument("invalid comma-separated token IDs");
        result.push_back(value);
    }
    if (result.empty() || text.back() == ',')
        throw std::invalid_argument("empty token list or trailing comma");
    return result;
}
} // namespace kestrel
