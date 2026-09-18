#pragma once
#include <filesystem>
#include <json.hpp>
namespace kestrel {
using Json = nlohmann::json;
Json parse_json(std::string_view text);
Json read_json(const std::filesystem::path &path);
size_t json_size(const Json &value);
} // namespace kestrel
