#include "frame_timing.cuh"

#include <cmath>

// See `frame_timing.cuh` and the `frame-stage-timing` capability.
//
// Nothing here synchronizes: the frame path only records events (about a
// microsecond each), and the elapsed-time queries happen inside `resolve()`.

const char* FrameTiming::stage_key(int stage) {
    switch ( stage ) {
    case FrameUpdate: return "frame_update";
    case Collision: return "collision";
    case Substeps: return "substeps";
    case EndFrame: return "end_frame";
    default: return "unknown";
    }
}

FrameTiming::~FrameTiming() {
    if ( !created_ ) return;
    for ( int boundary = 0; boundary <= StageCount; ++boundary ) {
        for ( int slot = 0; slot < kSlots; ++slot ) {
            cudaEventDestroy(boundary_[boundary][slot]);
        }
    }
    if ( accum_created_ ) {
        for ( int slot = 0; slot < kSlots; ++slot ) {
            for ( int k = 0; k < kMaxIntervals; ++k ) {
                cudaEventDestroy(accum_start_[slot][k]);
                cudaEventDestroy(accum_stop_[slot][k]);
            }
        }
    }
}

bool FrameTiming::create_events() {
    if ( created_ ) return true;
    // Timing-enabled events: `cudaEventElapsedTime` needs them, and the flags
    // only affect what the event records, not the frame path.
    auto create = [](cudaEvent_t& event) {
        return cudaEventCreateWithFlags(&event, cudaEventDefault) == cudaSuccess;
    };
    bool ok = true;
    for ( int boundary = 0; boundary <= StageCount && ok; ++boundary ) {
        for ( int slot = 0; slot < kSlots && ok; ++slot ) {
            ok = create(boundary_[boundary][slot]);
        }
    }
    created_ = ok;
    return ok;
}

bool FrameTiming::create_accum_events() {
    if ( accum_created_ ) return true;
    auto create = [](cudaEvent_t& event) {
        return cudaEventCreateWithFlags(&event, cudaEventDefault) == cudaSuccess;
    };
    bool ok = true;
    for ( int slot = 0; slot < kSlots && ok; ++slot ) {
        for ( int k = 0; k < kMaxIntervals && ok; ++k ) {
            ok = create(accum_start_[slot][k]) && create(accum_stop_[slot][k]);
        }
    }
    accum_created_ = ok;
    return ok;
}

void FrameTiming::start_frame(int frame) {
    if ( !enabled_ ) return;
    if ( !create_events() ) {
        // Out of events: report the disabled contract rather than pretending.
        enabled_ = false;
        return;
    }
    frame_ = frame;
    slot_ = next_slot_;
    next_slot_ = (next_slot_ + 1) % kSlots;
    accum_count_[slot_] = 0;
    open_stage_ = -1;
    open_accum_ = -1;
}

void FrameTiming::finish_frame() {
    if ( !enabled_ ) return;
    pending_frame_ = frame_;
    pending_slot_ = slot_;
    pending_enabled_ = true;
}

void FrameTiming::begin(Stage stage) {
    if ( !enabled_ ) return;
    // Only the first stage opens a boundary of its own; the rest start where
    // the previous stage ended.
    if ( stage == FrameUpdate ) cudaEventRecord(boundary_[0][slot_], 0);
    open_stage_ = stage;
}

void FrameTiming::end(Stage stage) {
    if ( !enabled_ ) return;
    cudaEventRecord(boundary_[stage + 1][slot_], 0);
    open_stage_ = -1;
}

void FrameTiming::begin_accum(Stage label) {
    if ( !enabled_ ) return;
    if ( !create_accum_events() ) return;
    const int k = accum_count_[slot_];
    if ( k >= kMaxIntervals ) return; // report the intervals that fit
    cudaEventRecord(accum_start_[slot_][k], 0);
    accum_label_[slot_][k] = (int8_t)label;
    accum_owner_[slot_][k] = (int8_t)(open_stage_ < 0 ? label : open_stage_);
    open_accum_ = k;
}

void FrameTiming::end_accum(Stage label) {
    if ( !enabled_ || open_accum_ < 0 ) return;
    const int k = open_accum_;
    cudaEventRecord(accum_stop_[slot_][k], 0);
    open_accum_ = -1;
    accum_count_[slot_] = k + 1;
}

float FrameTiming::elapsed_ms(cudaEvent_t from, cudaEvent_t to) const {
    float ms = 0.f;
    cudaEventElapsedTime(&ms, from, to);
    return ms;
}

const FrameTiming::Snapshot& FrameTiming::resolve() {
    if ( pending_frame_ < 0 ) {
        // Nothing recorded since the last resolve. While the caller is still
        // collecting, the last sample stands (a second readback of the same
        // frame reports the same numbers); once collection stops, the readback
        // reports the disabled contract instead of a sample from an earlier,
        // enabled stretch.
        if ( !enabled_ ) {
            current_ = Snapshot();
        }
        return current_;
    }
    if ( cudaEventQuery(boundary_[StageCount][pending_slot_]) != cudaSuccess ) {
        // The sample is not complete: keep the previous numbers and say so.
        current_.stale = true;
        return current_;
    }
    current_.frame = pending_frame_;
    current_.enabled = pending_enabled_;
    current_.stale = false;
    // The stages are consecutive boundaries, so the total is their sum.
    current_.total_ms = elapsed_ms(boundary_[0][pending_slot_],
        boundary_[StageCount][pending_slot_]);
    // Sub-intervals are charged to their label and taken out of the stage that
    // was open around them, which keeps the four stages a partition of the
    // frame while letting a solver report a piece of its own loop separately.
    float charged[StageCount] = {};
    float removed[StageCount] = {};
    for ( int k = 0; k < accum_count_[pending_slot_]; ++k ) {
        const float ms = elapsed_ms(accum_start_[pending_slot_][k],
            accum_stop_[pending_slot_][k]);
        charged[accum_label_[pending_slot_][k]] += ms;
        removed[accum_owner_[pending_slot_][k]] += ms;
    }
    for ( int stage = 0; stage < StageCount; ++stage ) {
        const float span =
            elapsed_ms(boundary_[stage][pending_slot_], boundary_[stage + 1][pending_slot_]);
        current_.stage_ms[stage] = fmaxf(0.f, span - removed[stage] + charged[stage]);
    }
    // Resolved once; the next recorded frame becomes the pending sample.
    pending_frame_ = -1;
    return current_;
}
