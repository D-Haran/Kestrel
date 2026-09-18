#include "kestrel/cli.hpp"
#include "kestrel/tokenizer.hpp"
#include <fstream>
#include <iostream>
using namespace kestrel;
int main(int argc, char **argv) {
    try {
        Args args(argc, argv,
                  {"--model", "--backend", "--prompt", "--tokens", "--max-tokens", "--temperature",
                   "--top-k", "--top-p", "--seed", "--context", "--tokenizer-python",
                   "--tokenizer-helper", "--json"},
                  {"--help", "--interactive", "--profile"});
        if (args.has("--help")) {
            std::cout << "Kestrel native Llama inference\n"
                         "  --model DIR --backend cpu|metal|cuda\n"
                         "  --prompt TEXT | --tokens 1,5,9 | --interactive\n"
                         "  --max-tokens 128 --temperature 0.7 --top-k 40 --top-p 0.9 --seed 42\n"
                         "  --context N --json FILE --profile\n"
                         "  --tokenizer-python python3 --tokenizer-helper PATH\n"
                         "Text tokenization uses a separate Python tokenizer library; inference is "
                         "native.\n"
                         "Interactive mode starts a fresh context for each line; /quit exits.\n";
            return 0;
        }
        if (!args.has("--model"))
            throw std::invalid_argument("--model DIR is required (see --help)");
        if (int(args.has("--prompt")) + int(args.has("--tokens")) +
                int(args.has("--interactive")) !=
            1)
            throw std::invalid_argument("choose exactly one of --prompt, --tokens, --interactive");
        if (args.has("--interactive") && args.has("--json"))
            throw std::invalid_argument("--json is unavailable in interactive mode");
        auto backend = make_backend(args.get("--backend", "cpu"));
        Model model(args.get("--model"), *backend);
        model.profiler.enabled = args.has("--profile");
        size_t capacity = args.number("--context", model.config().max_positions),
               max_tokens = args.number("--max-tokens", 128);
        Session session(model, capacity);
        auto seed = args.number("--seed", 42);
        if (seed > UINT32_MAX)
            throw std::invalid_argument("seed must fit uint32");
        Sampler sampler({args.real("--temperature", 0), args.number("--top-k", 0),
                         args.real("--top-p", 1), uint32_t(seed)});
        Tokenizer tokenizer(args.get("--model"), args.get("--tokenizer-python", "python3"),
                            args.get("--tokenizer-helper", KESTREL_TOKENIZER_HELPER));
        auto run = [&](const std::string &prompt) {
            auto ids = args.has("--tokens") ? parse_tokens(args.get("--tokens"))
                                            : tokenizer.encode(prompt);
            if (ids.empty())
                throw std::invalid_argument("tokenizer produced an empty prompt");
            if (ids.size() > capacity || (max_tokens && max_tokens - 1 > capacity - ids.size()))
                throw std::invalid_argument(
                    "prompt plus decode steps exceed --context; shorten the prompt or generation");
            auto result = generate(session, ids, max_tokens, model.config().eos,
                                   [&](Input x) { return sampler(x); });
            std::string text;
            if (args.has("--tokens")) {
                for (int id : result.tokens)
                    std::cout << id << ' ';
                std::cout << '\n';
            } else {
                text = tokenizer.decode(result.tokens);
                std::cout << text << '\n';
            }
            if (args.has("--json")) {
                std::ofstream file(args.get("--json"));
                if (!file)
                    throw std::runtime_error("cannot open JSON output");
                file << Json{{"backend", backend->name()},
                             {"prompt_ids", ids},
                             {"generated_ids", result.tokens},
                             {"text", text},
                             {"prefill_ms", result.prefill_ms},
                             {"time_to_first_token_ms", result.first_token_ms},
                             {"kv_cache_bytes", session.cache_bytes()}}
                            .dump(2)
                     << '\n';
                if (!file)
                    throw std::runtime_error("JSON output write failed");
            }
        };
        if (args.has("--interactive")) {
            std::string line;
            while (std::cout << "> " && std::getline(std::cin, line)) {
                if (line == "/quit")
                    break;
                try {
                    run(line);
                } catch (const std::exception &e) {
                    std::cerr << "error: " << e.what() << '\n';
                }
            }
        } else
            run(args.get("--prompt"));
        if (model.profiler.enabled)
            model.profiler.print(std::cerr);
    } catch (const std::exception &e) {
        std::cerr << "kestrel: " << e.what() << '\n';
        return 1;
    }
}
