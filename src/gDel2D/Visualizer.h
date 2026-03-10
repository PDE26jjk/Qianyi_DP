#ifndef VISUALIZER
#define VISUALIZER

// Stub Visualizer for compilation without OpenGL/GLEW/GLUT
// Original file renamed to Visualizer.h.orig

#include "gDel2D/GpuDelaunay.h"

class Visualizer {
public:
    static Visualizer* instance() {
        static Visualizer inst;
        return &inst;
    }

    bool isEnable() const {
        return false; // Visualization disabled
    }

    void disable() {
        // Do nothing
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5>
    void addFrame(T1&, T2&, T3&, T4&, T5&) {
        // Do nothing - visualization disabled
    }

    template<typename T1, typename T2, typename T3, typename T4>
    void addFrame(T1&, T2&, T3&, T4&) {
        // Do nothing - visualization disabled
    }

    template<typename T1, typename T2, typename T3>
    void addFrame(T1&, T2&, T3&) {
        // Do nothing - visualization disabled
    }
};

#endif
