#include "kestrel/backend.hpp"
#include "kestrel/config.hpp"
#include "kestrel/generation.hpp"
#include "kestrel/model.hpp"
#include "kestrel/safetensors.hpp"
#include "kestrel/sampling.hpp"
#include "kestrel/tensor.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unistd.h>
using namespace kestrel;
static int checks = 0;
void check(bool ok, const char *message) {
    ++checks;
    if (!ok)
        throw std::runtime_error(message);
}
void near(float a, float b, float eps = 1e-5f) {
    check(std::isfinite(a) && std::abs(a - b) <= eps, "numeric mismatch");
}
template <class F> void throws(F f) {
    bool caught = false;
    try {
        f();
    } catch (const std::exception &) {
        caught = true;
    }
    check(caught, "expected exception");
}
void tensor_tests() {
    Tensor t({2, 3});
    t.values()[4] = 7;
    check(t.view().strides() == std::vector<size_t>({3, 1}), "strides");
    near(t.view().at(std::vector<size_t>{1, 1}), 7);
    throws([&] { t.view().at(6); });
    throws([&] { t.view().at(std::vector<size_t>{0}); });
    throws([] { checked_mul(std::numeric_limits<size_t>::max(), 2); });
    throws([] { TensorView({}, DType::f32, {2}); });
    check(Tensor({}).values().size() == 1, "scalar");
    check(Tensor({2, 0}).values().empty(), "empty shape");
    std::byte one_half[] = {std::byte{0}, std::byte{0x3c}};
    std::byte one_bfloat[] = {std::byte{0x80}, std::byte{0x3f}};
    near(decode_scalar(one_half, DType::f16), 1);
    near(decode_scalar(one_bfloat, DType::bf16), 1);
    throws([] { parse_dtype("I8"); });
    const auto half = [](uint16_t bits) {
        std::byte bytes[] = {std::byte(bits & 255), std::byte(bits >> 8)};
        return decode_scalar(bytes, DType::f16);
    };
    near(half(1), std::ldexp(1.f, -24), 0);
    near(half(0x7bff), 65504.f, 0);
    near(half(0xbc00), -1, 0);
    check(std::signbit(half(0x8000)), "FP16 negative zero");
    check(std::isinf(half(0x7c00)), "FP16 infinity");
    check(std::isnan(half(0x7e00)), "FP16 NaN");
}
void loading_tests() {
    auto j = Json{{"model_type", "llama"},    {"hidden_size", 8},
                  {"intermediate_size", 16},  {"num_hidden_layers", 2},
                  {"num_attention_heads", 2}, {"num_key_value_heads", 1},
                  {"vocab_size", 32},         {"max_position_embeddings", 64}};
    check(Config::parse(j).head_dim() == 4, "config dimensions");
    auto bad = j;
    bad["num_key_value_heads"] = 3;
    throws([&] { Config::parse(bad); });
    bad = j;
    bad["rope_scaling"] = {{"factor", 2}};
    throws([&] { Config::parse(bad); });
    bad = j;
    bad["hidden_size"] = -1;
    throws([&] { Config::parse(bad); });
    throws([] { parse_json(R"({"x":1,"x":2})"); });
    auto path = std::filesystem::temp_directory_path() /
                ("kestrel-test-" + std::to_string(getpid()) + ".safetensors");
    auto write = [&](Json header, size_t payload) {
        std::ofstream f(path, std::ios::binary);
        auto h = header.dump();
        uint64_t n = h.size();
        for (int i = 0; i < 8; ++i)
            f.put(char((n >> (8 * i)) & 255));
        f << h;
        for (size_t i = 0; i < payload; ++i)
            f.put(0);
    };
    Json header = {{"weight", {{"dtype", "F32"}, {"shape", {2, 3}}, {"data_offsets", {0, 24}}}}};
    write(header, 24);
    {
        SafeTensors st(path);
        check(st.tensors().at("weight").size() == 6, "load tensor");
    }
    write(header, 23);
    throws([&] { SafeTensors st(path); });
    header["weight"]["data_offsets"] = {1, 25};
    write(header, 25);
    throws([&] { SafeTensors st(path); });
    header["weight"]["data_offsets"] = {24, 0};
    write(header, 24);
    throws([&] { SafeTensors st(path); });
    {
        std::ofstream f(path);
        f << "bad";
    }
    throws([&] { SafeTensors st(path); });
    std::filesystem::remove(path);
}
void math_tests(Backend &b) {
    Tensor w({2, 3});
    for (size_t i = 0; i < 6; ++i)
        w.values()[i] = float(i + 1);
    Vec x{1, 2, 3, 3, 2, 1}, y(4);
    b.matmul(w.view(), x, y, 2);
    near(y[0], 14);
    near(y[1], 32);
    near(y[2], 10);
    near(y[3], 28);
    Vec a{1, 2, 3, 4}, unit(4, 1), out(4);
    b.rmsnorm(a, unit, out, 1e-6f);
    for (size_t i = 0; i < 4; ++i)
        near(out[i], a[i] / std::sqrt(7.5f + 1e-6f));
    b.add(a, a, out);
    near(out[2], 6);
    b.multiply(a, out, out);
    near(out[2], 18);
    b.silu(a, out);
    near(out[0], 1 / (1 + std::exp(-1.f)));
    b.softmax(Vec{1000, 1001, 1002, 1003}, out);
    float sum = 0;
    for (float f : out)
        sum += f;
    near(sum, 1);
    near(out[3], 1 / (1 + std::exp(-1.f) + std::exp(-2.f) + std::exp(-3.f)));
    Vec rot = a;
    b.rope(rot, 1, 4, 0, 10000);
    for (size_t i = 0; i < 4; ++i)
        near(rot[i], a[i]);
    b.rope(rot, 1, 4, 1, 10000);
    near(rot[0], std::cos(1.f) - 3 * std::sin(1.f));
    near(rot[1], 2 * std::cos(.01f) - 4 * std::sin(.01f));
    // Four Q heads share two KV heads. Zero queries make averaging exact.
    Vec q(8, 0), k(8, 0), v{1, 2, 10, 20, 3, 4, 30, 40}, att(8);
    b.attention(q, k, v, att, 4, 2, 2, 2);
    Vec expected{2, 3, 2, 3, 20, 30, 20, 30};
    for (size_t i = 0; i < 8; ++i)
        near(att[i], expected[i]);
    throws([&] { b.attention(q, k, v, att, 3, 2, 2, 2); });
    throws([&] { b.matmul(w.view(), a, out, 1); });
}
void model_tests(const std::filesystem::path &dir, Backend &b) {
    Model model(dir, b);
    Session s(model, 16);
    std::vector<int> ids{1, 5, 9};
    auto logits = s.prefill(ids);
    check(logits.size() == 32, "logit shape");
    for (float f : logits)
        check(std::isfinite(f), "finite logits");
    auto again = s.prefill(ids);
    for (size_t i = 0; i < 32; ++i)
        near(logits[i], again[i]);
    auto next = s.decode(4);
    ids.push_back(4);
    auto full = s.prefill(ids);
    for (size_t i = 0; i < 32; ++i)
        near(next[i], full[i]);
    throws([&] { s.decode(100); });
    check(s.position() == 0, "failed step resets session");
    throws([&] { s.decode(1); });
    throws([&] { s.prefill({}); });
}
void cache_generation_tests(const std::filesystem::path &dir) {
    auto b = make_cpu_backend();
    Model m(dir, *b);
    ContiguousKVCache cache(m.config(), 2);
    check(cache.bytes() == 2 * 2 * 2 * 2 * 4 * 4, "KV memory accounting");
    Vec k(8, 3), v(8, 5);
    cache.append(0, 0, k, v);
    near(cache.keys(0)[0], 3);
    near(cache.values(0)[7], 5);
    check(cache.length(1) == 0, "layer isolation");
    throws([&] { cache.append(0, 0, k, v); });
    cache.append(0, 1, k, v);
    throws([&] { cache.append(0, 2, k, v); });
    cache.clear();
    check(cache.keys(0).empty(), "cache reset");
    Session s(m, 16);
    std::vector<int> ids{1, 5, 9};
    auto a = generate(s, ids, 5, 2, greedy, {}, false);
    auto bgen = generate(s, ids, 5, 2, greedy, {}, false);
    check(a.tokens == bgen.tokens && a.tokens.size() == 5, "deterministic greedy generation");
    check(a.inter_token_ms.size() == 4, "decode interval count");
    auto eos = generate(s, ids, 5, 2, [](Input) { return 2; });
    check(eos.tokens.size() == 1, "EOS termination");
    auto zero = generate(s, ids, 0, 2);
    check(zero.tokens.empty(), "zero generation");
}
void sampling_tests() {
    Vec logits{1, 3, 2};
    Sampler greedy_sampler({0, 0, 1, 1});
    check(greedy_sampler(logits) == 1, "greedy sampling");
    Sampler topk({1, 1, 1, 1}), topp({1, 0, .01f, 1});
    for (int i = 0; i < 100; ++i) {
        check(topk(logits) == 1, "top-k filter");
        check(topp(logits) == 1, "top-p filter");
    }
    Sampler a({.7f, 0, .9f, 42}), b({.7f, 0, .9f, 42});
    for (int i = 0; i < 50; ++i)
        check(a(logits) == b(logits), "seed reproducibility");
    Sampler uniform({1, 0, 1, 42});
    int counts[3] = {};
    for (int i = 0; i < 6000; ++i)
        ++counts[uniform(Vec{0, 0, 0})];
    for (int n : counts)
        check(n > 1800 && n < 2200, "sampling distribution");
    throws([] { Sampler s({-1, 0, 1, 1}); });
    throws([] { Sampler s({1, 0, 0, 1}); });
    throws([&] { a(Vec{NAN, 1}); });
}
int main(int argc, char **argv) {
    try {
        Profiler profiler;
        profiler.enabled = true;
        {
            auto scope = profiler.scope("test");
        }
        check(profiler.json()["test"]["calls"] == 1, "profiler call count");
        std::ostringstream profile_text;
        profiler.print(profile_text);
        check(profile_text.str().find("test") != std::string::npos, "profiler table");
        sampling_tests();
        tensor_tests();
        loading_tests();
        auto cpu = make_cpu_backend();
        math_tests(*cpu);
        if (argc > 1) {
            model_tests(argv[1], *cpu);
            cache_generation_tests(argv[1]);
        }
        if (argc > 2) {
            auto gpu = make_backend(argv[2]);
            math_tests(*gpu);
            model_tests(argv[1], *gpu);
        }
        std::cout << checks << " checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL after " << checks << " checks: " << e.what() << '\n';
        return 1;
    }
}
