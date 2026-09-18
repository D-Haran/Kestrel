#include "kestrel/profiler.hpp"
#include <iomanip>
namespace kestrel {
Json Profiler::json(double runtime_ms) const {
    Json j = Json::object();
    double total = entries_.contains("__forward") ? entries_.at("__forward").ms : 0;
    double runtime = runtime_ms > 0 ? runtime_ms :
        (entries_.contains("__generation") ? entries_.at("__generation").ms : total);
    for (auto &[name, e] : entries_)
        if (!name.starts_with("__"))
            j[name] = {{"calls", e.calls},
                       {"total_ms", e.ms},
                       {"mean_ms", e.calls ? e.ms / e.calls : 0},
                       {"percent_measured_runtime", runtime ? 100 * e.ms / runtime : 0},
                       {"percent_forward_runtime", total ? 100 * e.ms / total : 0}};
    return j;
}
void Profiler::print(std::ostream &out) const {
    out << "Operation                Calls     Total ms      Mean ms    % Runtime\n";
    auto data = json();
    for (auto &[name, v] : data.items())
        out << std::left << std::setw(25) << name << std::right << std::setw(6)
            << v["calls"].get<size_t>() << std::setw(13) << std::fixed << std::setprecision(3)
            << v["total_ms"].get<double>() << std::setw(13)
            << v["mean_ms"].get<double>() << std::setw(13)
            << v["percent_measured_runtime"].get<double>() << '\n';
}
} // namespace kestrel
