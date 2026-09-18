#include "kestrel/model.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
using namespace kestrel;
int main(int argc, char **argv) {
    try {
        if (argc != 5)
            throw std::invalid_argument(
                "usage: kestrel_trace MODEL BACKEND TOKEN_IDS_CSV OUTPUT.json");
        auto backend = make_backend(argv[2]);
        Model model(argv[1], *backend);
        std::vector<int> ids;
        std::istringstream stream(argv[3]);
        std::string part;
        while (std::getline(stream, part, ',')) {
            size_t n = 0;
            int v = std::stoi(part, &n);
            if (n != part.size())
                throw std::invalid_argument("invalid token");
            ids.push_back(v);
        }
        Json output = Json::object();
        Trace trace = [&](const std::string &name, size_t pos, Input values) {
            output[name][pos] = Vec(values.begin(), values.end());
        };
        Session session(model, model.config().max_positions);
        // Deliberately cross the prefill/decode boundary in every multi-token trace.
        if (ids.empty())
            throw std::invalid_argument("empty token list");
        session.prefill(std::span(ids).first(1), trace);
        for (size_t i = 1; i < ids.size(); ++i)
            session.decode(ids[i], trace);
        std::ofstream file(argv[4]);
        if (!file)
            throw std::runtime_error("cannot open trace output");
        file << output.dump(2) << '\n';
        if (!file)
            throw std::runtime_error("trace write failed");
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
