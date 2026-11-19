#include "kestrel/config.hpp"
#include "kestrel/tensor.hpp"

#include <cmath>
#include <fstream>
#include <set>
#include <stdexcept>
namespace kestrel {
Json parse_json(std::string_view text) {
    std::vector<std::set<std::string>> keys;
    return Json::parse(text, [&](int depth, Json::parse_event_t event, Json &parsed) {
        if (depth > 128)
            throw std::runtime_error("JSON nesting exceeds 128");
        if (event == Json::parse_event_t::object_start)
            keys.emplace_back();
        if (event == Json::parse_event_t::key &&
            !keys.back().insert(parsed.get<std::string>()).second)
            throw std::runtime_error("duplicate JSON key: " + parsed.get<std::string>());
        if (event == Json::parse_event_t::object_end)
            keys.pop_back();
        return true;
    });
}
Json read_json(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open JSON: " + path.string());
    if (std::filesystem::file_size(path) > 100'000'000)
        throw std::runtime_error("JSON file exceeds 100 MB");
    return parse_json(std::string(std::istreambuf_iterator<char>(in), {}));
}
size_t json_size(const Json &j) {
    if (!j.is_number_unsigned() && !(j.is_number_integer() && j.get<int64_t>() >= 0))
        throw std::runtime_error("expected nonnegative integer");
    const auto n = j.get<uint64_t>();
    if (n > SIZE_MAX)
        throw std::runtime_error("integer out of range");
    return size_t(n);
}
Config Config::parse(const Json &j) {
    if (j.value("model_type", "") != "llama")
        throw std::runtime_error("only model_type=llama is supported");
    if (j.contains("architectures") && j.at("architectures") != Json::array({"LlamaForCausalLM"}))
        throw std::runtime_error("unsupported Llama architecture");
    for (auto key : {"quantization_config", "sliding_window", "rope_parameters"})
        if (j.contains(key) && !j.at(key).is_null())
            throw std::runtime_error(std::string("unsupported config: ") + key);
    if (j.value("hidden_act", "silu") != "silu" || j.value("attention_bias", false) ||
        j.value("mlp_bias", false) || j.value("pretraining_tp", 1) != 1 ||
        j.value("partial_rotary_factor", 1.f) != 1.f)
        throw std::runtime_error("requires SiLU, bias-free projections, pretraining_tp=1");
    Config c{json_size(j.at("hidden_size")),
             json_size(j.at("intermediate_size")),
             json_size(j.at("num_hidden_layers")),
             json_size(j.at("num_attention_heads")),
             json_size(j.value("num_key_value_heads", j.at("num_attention_heads"))),
             json_size(j.at("vocab_size")),
             json_size(j.at("max_position_embeddings")),
             j.value("rms_norm_eps", 1e-6f),
             j.value("rope_theta", 10000.f)};
    if (!c.hidden || !c.intermediate || !c.layers || !c.heads || !c.kv_heads || !c.vocab ||
        !c.max_positions || c.hidden % c.heads || c.heads % c.kv_heads || c.head_dim() % 2)
        throw std::runtime_error("invalid model dimensions or GQA head mapping");
    if (j.contains("head_dim") && json_size(j.at("head_dim")) != c.head_dim())
        throw std::runtime_error("custom head_dim is unsupported");
    if (!(c.epsilon > 0) || !std::isfinite(c.epsilon) || !(c.theta > 0) || !std::isfinite(c.theta))
        throw std::runtime_error("invalid norm epsilon or RoPE theta");
    if (j.contains("rope_scaling") && !j.at("rope_scaling").is_null()) {
        const auto &rs = j.at("rope_scaling");
        std::string type = rs.value("rope_type", rs.value("type", std::string()));
        if (type != "llama3")
            throw std::runtime_error("unsupported rope_scaling rope_type: " + type);
        c.rope_llama3 = true;
        c.rope_factor = rs.value("factor", 8.f);
        c.rope_low_freq = rs.value("low_freq_factor", 1.f);
        c.rope_high_freq = rs.value("high_freq_factor", 4.f);
        c.rope_original_ctx = json_size(rs.at("original_max_position_embeddings"));
        if (!(c.rope_factor > 0) || !(c.rope_high_freq > c.rope_low_freq) || !c.rope_original_ctx)
            throw std::runtime_error("invalid llama3 rope_scaling parameters");
    }
    c.bos = j.value("bos_token_id", 1);
    c.eos = j.value("eos_token_id", 2);
    c.tied = j.value("tie_word_embeddings", false);
    if (c.bos < 0 || c.eos < 0 || size_t(c.bos) >= c.vocab || size_t(c.eos) >= c.vocab)
        throw std::runtime_error("invalid BOS/EOS token IDs");
    checked_mul(c.hidden, c.intermediate);
    return c;
}
} // namespace kestrel
