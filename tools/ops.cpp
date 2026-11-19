#include "kestrel/backend.hpp"
#include "kestrel/json.hpp"
#include <cmath>
#include <iostream>
using namespace kestrel;
int main(int argc, char **argv) {
    try {
        auto b = make_backend(argc > 1 ? argv[1] : "cpu");
        Json j;
        Vec x(16), w(16), y(16);
        for (size_t i = 0; i < 16; ++i) {
            x[i] = std::sin(float(i));
            w[i] = .5f + float(i) / 16;
        }
        j["x"] = x;
        j["w"] = w;
        b->rmsnorm(x, w, y, 1e-5f);
        j["rmsnorm"] = y;
        b->silu(x, y);
        j["silu"] = y;
        b->multiply(y, w, y);
        j["swiglu"] = y;
        b->softmax(x, y);
        j["softmax"] = y;
        y = x;
        Vec rope_freq{1.f, 0.1f, 0.01f, 0.001f}; // theta=10000, dim=8: pow(10000, -2i/8)
        b->rope(y, 2, 8, 7, rope_freq);
        j["rope"] = y;
        b->add(x, w, y);
        j["add"] = y;
        Tensor matrix({3, 16});
        for (size_t i = 0; i < matrix.values().size(); ++i)
            matrix.values()[i] = std::cos(float(i));
        j["matrix"] = Vec(matrix.values().begin(), matrix.values().end());
        y.resize(3);
        b->matmul(matrix.view(), x, y);
        j["matmul"] = y;
        Vec k(3 * 2 * 4), v(k.size());
        for (size_t i = 0; i < k.size(); ++i) {
            k[i] = std::cos(float(i));
            v[i] = std::sin(float(i) * .7f);
        }
        j["k"] = k;
        j["v"] = v;
        y.resize(16);
        b->attention(x, k, v, y, 4, 2, 4, 3);
        j["attention"] = y;
        std::cout << j.dump() << '\n';
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
