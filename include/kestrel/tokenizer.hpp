#pragma once
#include "kestrel/json.hpp"
namespace kestrel {
// Explicit subprocess boundary. Python is used only for the tokenizer library.
class Tokenizer {
  public:
    Tokenizer(std::string model, std::string python, std::string helper)
        : model_(std::move(model)), python_(std::move(python)), helper_(std::move(helper)) {}
    std::vector<int> encode(const std::string &text) const;
    std::string decode(const std::vector<int> &ids) const;

  private:
    Json request(const Json &input) const;
    std::string model_, python_, helper_;
};
} // namespace kestrel
