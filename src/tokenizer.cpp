#include "kestrel/tokenizer.hpp"
#include <cerrno>
#include <cstring>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
namespace kestrel {
namespace {
struct Temp {
    char path[64] = "/tmp/kestrel-tokenizer-XXXXXX";
    int fd;
    Temp() : fd(mkstemp(path)) {
        if (fd < 0)
            throw std::runtime_error("cannot create tokenizer temporary file");
    }
    ~Temp() {
        close(fd);
        unlink(path);
    }
    Temp(const Temp &) = delete;
};
} // namespace
Json Tokenizer::request(const Json &input) const {
    Temp in, out;
    auto text = input.dump();
    size_t offset = 0;
    while (offset < text.size()) {
        auto n = write(in.fd, text.data() + offset, text.size() - offset);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            throw std::runtime_error("tokenizer input write failed");
        }
        offset += size_t(n);
    }
    if (lseek(in.fd, 0, SEEK_SET) < 0)
        throw std::runtime_error("tokenizer seek failed");
    posix_spawn_file_actions_t actions;
    int status = posix_spawn_file_actions_init(&actions);
    if (status)
        throw std::runtime_error("cannot initialize tokenizer process actions");
    status = posix_spawn_file_actions_adddup2(&actions, in.fd, STDIN_FILENO);
    if (!status)
        status = posix_spawn_file_actions_adddup2(&actions, out.fd, STDOUT_FILENO);
    pid_t pid = 0;
    char *argv[] = {const_cast<char *>(python_.c_str()), const_cast<char *>(helper_.c_str()),
                    const_cast<char *>(model_.c_str()), nullptr};
    if (!status)
        status = posix_spawnp(&pid, python_.c_str(), &actions, nullptr, argv, environ);
    posix_spawn_file_actions_destroy(&actions);
    if (status)
        throw std::runtime_error("cannot start tokenizer Python: " +
                                 std::string(std::strerror(status)));
    int child = 0;
    while (waitpid(pid, &child, 0) < 0) {
        if (errno == EINTR)
            continue;
        throw std::runtime_error("tokenizer wait failed");
    }
    if (!WIFEXITED(child) || WEXITSTATUS(child) != 0)
        throw std::runtime_error(
            "tokenizer helper failed; install tokenizers/sentencepiece or use --tokens");
    return read_json(out.path);
}
std::vector<int> Tokenizer::encode(const std::string &text) const {
    return request({{"operation", "encode"}, {"text", text}}).get<std::vector<int>>();
}
std::string Tokenizer::decode(const std::vector<int> &ids) const {
    return request({{"operation", "decode"}, {"ids", ids}}).get<std::string>();
}
} // namespace kestrel
