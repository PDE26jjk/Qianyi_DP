#pragma once

#include <cuda_runtime.h>

#include <cstdint>

// Per-frame stage timing for the simulation loop (see the
// `frame-stage-timing` capability).
//
// One CUDA event per stage *boundary*, created once and re-recorded every frame
// on the stream the work runs on: the stages are consecutive, so `StageCount+1`
// boundaries describe them all (and the total). Recording is asynchronous and
// the frame path never synchronizes; `resolve()` reports the most recent frame
// whose events have completed and falls back to the previous sample while they
// have not, so a caller polling for timings can never stall the simulation.
//
// The stages are the boundaries of `Simulator::update`, which makes the timer
// solver-agnostic: the substep stage covers whatever the selected solver does
// inside its loop.
class FrameTiming {
public:
    enum Stage : int {
        FrameUpdate = 0, // normals, pick, pin, sewing bookkeeping
        Collision,       // broad and narrow phase
        Substeps,        // external forces, plastic state, Newton loop, seams
        EndFrame,        // end-of-frame work
        StageCount,
    };

    // Flat readback key of a stage, in the same order as `Stage`.
    static const char* stage_key(int stage);

    struct Snapshot {
        int frame = -1;
        bool enabled = false;
        // True while the reported numbers predate the most recent frame.
        bool stale = true;
        float total_ms = 0.f;
        float stage_ms[StageCount] = {};
    };

    FrameTiming() = default;
    ~FrameTiming();
    FrameTiming(const FrameTiming&) = delete;
    FrameTiming& operator=(const FrameTiming&) = delete;

    void set_enabled(bool on) { enabled_ = on; }
    bool enabled() const { return enabled_; }

    // Frame brackets. `start_frame` picks this frame's slot; the first
    // `begin(FrameUpdate)` records the leading boundary and each `end(stage)`
    // records the boundary that closes that stage, so `end(EndFrame)` closes the
    // frame as well. `finish_frame` then makes the frame the pending sample.
    void start_frame(int frame);
    void finish_frame();

    // Stage brackets. No-ops while the timer is disabled, and `begin` only has
    // to record the leading boundary (every other stage starts where the
    // previous one ended).
    void begin(Stage stage);
    void end(Stage stage);

    // Accumulated sub-intervals. A solver brackets a piece of work that repeats
    // inside another stage (the per-substep contact queries inside the substep
    // loop, say) with these; the time is charged to `label` and taken out of
    // whichever stage is open around it, so the reported stages still partition
    // the frame.
    void begin_accum(Stage label);
    void end_accum(Stage label);

    const Snapshot& resolve();

private:
    static constexpr int kSlots = 2; // recorded and pending never share a slot
    // Sub-intervals per frame and slot. A frame with more substeps than this
    // reports the first `kMaxIntervals` of them.
    static constexpr int kMaxIntervals = 64;

    bool create_events();
    bool create_accum_events();
    float elapsed_ms(cudaEvent_t from, cudaEvent_t to) const;

    // boundary_[i] is the instant stage i-1 ends and stage i starts;
    // boundary_[0] opens the frame and boundary_[StageCount] closes it.
    cudaEvent_t boundary_[StageCount + 1][kSlots] = {};
    cudaEvent_t accum_start_[kSlots][kMaxIntervals] = {};
    cudaEvent_t accum_stop_[kSlots][kMaxIntervals] = {};
    int8_t accum_label_[kSlots][kMaxIntervals] = {};
    int8_t accum_owner_[kSlots][kMaxIntervals] = {};
    int accum_count_[kSlots] = {};

    bool created_ = false;
    bool accum_created_ = false;
    bool enabled_ = false;
    int slot_ = 0;
    int next_slot_ = 0;
    int frame_ = -1;
    int open_stage_ = -1;
    int open_accum_ = -1;
    int pending_frame_ = -1;
    int pending_slot_ = 0;
    bool pending_enabled_ = false;
    Snapshot current_;
};
