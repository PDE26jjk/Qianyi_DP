#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>

class PerfTiming {
public:
    using Clock = std::chrono::steady_clock;

    // 供外部使用的嵌套结果结构
    struct TimingResult {
        double duration = 0.0;          // mm
        size_t call_count = 0;
        std::unordered_map<std::string, TimingResult> children;
    };

    PerfTiming();
    ~PerfTiming() = default;
    PerfTiming(const PerfTiming&) = delete;
    // 开始计时一个名为 name 的块（支持嵌套）
    void start(const std::string& name);

    double stop();
    void clear();

    TimingResult getResults() const;

    static PerfTiming& global_timer();

private:
    struct Node {
        std::string name;
        double total_time_ms = 0.0;
        size_t call_count = 0;
        std::unordered_map<std::string, std::unique_ptr<Node>> children;

        explicit Node(const std::string& n);
    };

    struct StackEntry {
        Node* node;
        Clock::time_point start_time;
    };

    std::unique_ptr<Node> root_;
    std::vector<StackEntry> stack_;

    // 递归构建 TimingResult 的辅助函数
    static TimingResult buildResult(const Node* node);
};

// RAII 辅助类，进入作用域开始计时，退出时自动停止
class ScopedTimer {
public:
    ScopedTimer(PerfTiming& timing, const std::string& name)
    : timing_(timing), name_(name) {
        timing_.start(name_);
    }

    ~ScopedTimer() {
        try { timing_.stop(); }
        catch ( ... ) {}
    }

private:
    PerfTiming& timing_;
    std::string name_;
};
