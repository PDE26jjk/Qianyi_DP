#include "common/perf_timing.h"

PerfTiming& PerfTiming::global_timer() {
    static PerfTiming instance;
    return instance;
}

PerfTiming::Node::Node(const std::string& n) : name(n) {}

PerfTiming::PerfTiming()
    : root_(std::make_unique<Node>("")) {}

void PerfTiming::start(const std::string& name) {
    Node* current = stack_.empty() ? root_.get() : stack_.back().node;
    auto it = current->children.find(name);
    if (it == current->children.end()) {
        auto node = std::make_unique<Node>(name);
        // Node* raw = node.get();
        current->children[name] = std::move(node);
        it = current->children.find(name);
    }
    Node* child = it->second.get();
    child->call_count++;
    stack_.push_back({child, Clock::now()});
}

double PerfTiming::stop() {
    if (stack_.empty())
        throw std::runtime_error("No running timer to stop");
    auto& entry = stack_.back();
    auto end = Clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - entry.start_time).count();
    entry.node->total_time_ms += elapsed;
    stack_.pop_back();
    return elapsed;
}

void PerfTiming::clear() {
    stack_.clear();
    root_->children.clear();
}
PerfTiming::TimingResult PerfTiming::getResults() const {
    TimingResult result;
    for (const auto& [name, child] : root_->children) {
        result.children[name] = buildResult(child.get());
    }
    return result;
}

PerfTiming::TimingResult PerfTiming::buildResult(const Node* node) {
    TimingResult res;
    res.duration = node->total_time_ms;
    res.call_count = node->call_count;
    for (const auto& [name, child] : node->children) {
        res.children[name] = buildResult(child.get());
    }
    return res;
}