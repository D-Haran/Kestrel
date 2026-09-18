#include "kestrel/cli.hpp"
#include "metadata.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sys/resource.h>
using namespace kestrel;
static double percentile(std::vector<double> values, double p) {
    if (values.empty())
        return 0;
    std::sort(values.begin(), values.end());
    return values[size_t(std::ceil(p * values.size())) - 1];
}
int main(int argc, char **argv) {
    try {
        Args a(argc, argv,
               {"--model", "--backend", "--tokens", "--prompt-tokens", "--max-tokens",
                "--iterations", "--warmup", "--json"},
               {"--help", "--profile"});
        if (a.has("--help")) {
            std::cout
                << "kestrel_bench --model DIR --backend cpu --prompt-tokens 32 --max-tokens 32 "
                   "--iterations 3 --warmup 1 --json results.json [--profile]\nSynthetic "
                   "BOS-filled prompt unless --tokens CSV is supplied. EOS stopping is disabled.\n";
            return 0;
        }
        if (!a.has("--model"))
            throw std::invalid_argument("--model is required");
        if (a.has("--tokens") && a.has("--prompt-tokens"))
            throw std::invalid_argument("choose --tokens or --prompt-tokens");
        auto backend = make_backend(a.get("--backend", "cpu"));
        Model m(a.get("--model"), *backend);
        auto n = a.number("--max-tokens", 32), iterations = a.number("--iterations", 3),
             warmup = a.number("--warmup", 1);
        if (n < 2 || !iterations)
            throw std::invalid_argument(
                "benchmark requires >=2 generated tokens and >=1 measured iteration");
        std::vector<int> ids =
            a.has("--tokens") ? parse_tokens(a.get("--tokens"))
                              : std::vector<int>(a.number("--prompt-tokens", 32), m.config().bos);
        if (ids.empty() || ids.size() > m.config().max_positions ||
            n - 1 > m.config().max_positions - ids.size())
            throw std::invalid_argument("benchmark exceeds context limit");
        auto system = environment();
        auto metadata = model_metadata(a.get("--model"), m.config());
        auto started = timestamp();
        std::cout << "Kestrel Benchmark\n-----------------\nOS: " << system["os"]
                  << "\nMachine: " << system["machine"] << "\nChip: " << system["chip"]
                  << "\nMemory bytes: " << system.value("memory_bytes", Json("unknown"))
                  << "\nCompiler: " << KESTREL_COMPILER << "\nBuild: " << KESTREL_BUILD_TYPE
                  << "\nModel: " << metadata["name"] << "\nLayers: " << m.config().layers
                  << "\nHidden size: " << m.config().hidden << "\nKV heads: " << m.config().kv_heads
                  << "\nWeight dtype: " << metadata["weight_dtype"] << "\nBackend: " << backend->name()
                  << "\nPrompt / generated tokens: " << ids.size() << " / " << n
                  << "\nBatch size: 1\n";
        Session session(m, ids.size() + n - 1);
        for (size_t i = 0; i < warmup; ++i) {
            generate(session, ids, n, m.config().eos, greedy, {}, false);
            std::cerr << "warm-up " << i + 1 << '/' << warmup << " (excluded)\n";
        }
        m.profiler.enabled = a.has("--profile");
        m.profiler.clear();
        backend->diagnostics.enabled = m.profiler.enabled;
        backend->diagnostics.clear();
        Json runs = Json::array();
        double prefill = 0, ttft = 0, measured = 0;
        std::vector<double> latencies;
        for (size_t i = 0; i < iterations; ++i) {
            auto begin = Profiler::Clock::now();
            auto r = generate(session, ids, n, m.config().eos, greedy, {}, false);
            auto elapsed = std::chrono::duration<double, std::milli>(Profiler::Clock::now() - begin).count();
            measured += elapsed;
            prefill += r.prefill_ms;
            ttft += r.first_token_ms;
            latencies.insert(latencies.end(), r.inter_token_ms.begin(), r.inter_token_ms.end());
            runs.push_back({{"prefill_ms", r.prefill_ms},
                            {"elapsed_ms", elapsed},
                            {"decode_elapsed_ms", std::accumulate(r.inter_token_ms.begin(), r.inter_token_ms.end(), 0.0)},
                            {"generated_tokens", r.tokens.size()},
                            {"decode_tokens", r.inter_token_ms.size()},
                            {"time_to_first_token_ms", r.first_token_ms},
                            {"inter_token_ms", r.inter_token_ms},
                            {"generated_ids", r.tokens}});
        }
        double decode = std::accumulate(latencies.begin(), latencies.end(), 0.0);
        struct rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage))
            throw std::runtime_error("getrusage failed");
        size_t peak = size_t(usage.ru_maxrss);
#ifndef __APPLE__
        peak *= 1024;
#endif
        Json j = {
            {"backend", backend->name()},
            {"schema_version", 2},
            {"timestamp", started}, {"git_commit", KESTREL_GIT_COMMIT},
            {"build_type", KESTREL_BUILD_TYPE}, {"compiler", KESTREL_COMPILER},
            {"cmake_version", KESTREL_CMAKE_VERSION},
            {"system", system}, {"model", metadata},
            {"benchmark", {{"batch_size", 1}, {"prompt_tokens", ids.size()},
                {"prompt_ids", ids}, {"generated_tokens", n}, {"warmup_runs", warmup},
                {"measured_runs", iterations}, {"sampling", "greedy"}, {"stop_on_eos", false}}},
            {"prefill_elapsed_ms_total", prefill},
            {"decode_elapsed_ms_total", decode},
            {"measured_elapsed_ms_total", measured},
            {"decode_elapsed_ms_mean", decode / iterations},
            {"inter_token_sample_count", latencies.size()},
            {"percentile_method", "nearest rank over pooled inter-token intervals"},
            {"p99_note", latencies.size() < 100 ? "fewer than 100 intervals; p99 is not a reliable tail estimate" : "descriptive sample percentile, not a tail-latency guarantee"},
            {"min_inter_token_ms", *std::min_element(latencies.begin(), latencies.end())},
            {"max_inter_token_ms", *std::max_element(latencies.begin(), latencies.end())},
            {"prompt_tokens", ids.size()},
            {"generated_tokens", n},
            {"warmup_runs", warmup},
            {"measured_runs", iterations},
            {"prompt_kind", a.has("--tokens") ? "explicit token IDs" : "synthetic repeated BOS"},
            {"profiling_enabled", m.profiler.enabled},
            {"prefill_tokens_per_second", 1000 * double(ids.size()) * iterations / prefill},
            {"prefill_elapsed_ms_mean", prefill / iterations},
            {"decode_tokens_per_second", 1000 * latencies.size() / decode},
            {"decode_steps_per_run", n - 1},
            {"time_to_first_token_ms", ttft / iterations},
            {"mean_inter_token_ms", decode / latencies.size()},
            {"p50_inter_token_ms", percentile(latencies, .5)},
            {"p95_inter_token_ms", percentile(latencies, .95)},
            {"p99_inter_token_ms", percentile(latencies, .99)},
            {"model_weight_bytes", m.weight_bytes()},
            {"backend_resident_weight_bytes", backend->resident_weight_bytes()},
            {"kv_cache_bytes", session.cache_bytes()},
            {"peak_process_memory_bytes", peak},
            {"runs", runs},
            {"profile", m.profiler.json(measured)},
            {"backend_diagnostics", backend->diagnostics.json(measured)}};
        auto manifest = std::filesystem::path(a.get("--model")) / "download-manifest.json";
        if (std::filesystem::exists(manifest))
            j["checkpoint"] = read_json(manifest);
        std::cout << "Backend: " << backend->name()
                  << "\nPrefill: " << j["prefill_tokens_per_second"]
                  << " tokens/s\nDecode: " << j["decode_tokens_per_second"]
                  << " tokens/s\nTTFT: " << j["time_to_first_token_ms"]
                  << " ms\nInter-token p50/p95/p99: " << j["p50_inter_token_ms"] << " / "
                  << j["p95_inter_token_ms"] << " / " << j["p99_inter_token_ms"]
                  << " ms\nWeights / KV / peak RSS: " << m.weight_bytes() << " / "
                  << session.cache_bytes() << " / " << peak << " bytes\n";
        if (m.profiler.enabled) {
            m.profiler.print(std::cout);
            std::cout << "Nested backend diagnostics (overlap operation times):\n"
                      << j["backend_diagnostics"].dump(2) << '\n';
        }
        if (a.has("--json")) {
            std::ofstream f(a.get("--json"));
            if (!f)
                throw std::runtime_error("cannot open benchmark JSON");
            f << j.dump(2) << '\n';
            if (!f)
                throw std::runtime_error("benchmark JSON write failed");
        } else
            std::cout << j.dump(2) << '\n';
    } catch (const std::exception &e) {
        std::cerr << "kestrel_bench: " << e.what() << '\n';
        return 1;
    }
}
