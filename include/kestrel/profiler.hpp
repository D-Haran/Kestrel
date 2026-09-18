#pragma once
#include "kestrel/json.hpp"
#include <chrono>
#include <map>
#include <ostream>
namespace kestrel {
class Profiler {
  public:
    using Clock = std::chrono::steady_clock;
    bool enabled = false;
    struct Entry {
        size_t calls = 0;
        double ms = 0;
    };
    class Scope {
      public:
        Scope(Profiler &p, std::string name)
            : p_(p), name_(std::move(name)),
              start_(p.enabled ? Clock::now() : Clock::time_point{}) {}
        ~Scope() {
            if (p_.enabled) {
                auto &e = p_.entries_[name_];
                ++e.calls;
                e.ms += std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
            }
        }
        Scope(const Scope &) = delete;

      private:
        Profiler &p_;
        std::string name_;
        Clock::time_point start_;
    };
    Scope scope(std::string name) {
        return Scope(*this, std::move(name));
    }
    template <class F> void measure(const char *name, F &&f) {
        auto s = scope(name);
        f();
    }
    void clear() {
        entries_.clear();
    }
    Json json(double runtime_ms = 0) const;
    void record(const std::string &name, double ms) {
        if (enabled) { auto &e = entries_[name]; ++e.calls; e.ms += ms; }
    }
    void print(std::ostream &out) const;

  private:
    std::map<std::string, Entry> entries_;
};
} // namespace kestrel
