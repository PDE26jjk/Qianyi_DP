#pragma once
#include "solver_base.cuh"

#include <thrust/device_vector.h>

struct SolverXPBD : SolverBase {
    // ~SolverExplicit() = default;
    explicit SolverXPBD(Simulator* simulator):SolverBase(simulator){}

    void init() override;
    void step(float h) override;
    // XPBD survives much larger substeps than the explicit family but its
    // accuracy does not: on the t1 garment the motion is 3x too large at the
    // frontend's 4.5 ms default and settles at 1 ms. Set `xpbd_max_step_h = 0`
    // to switch the guard off.
    float max_stable_step_h() const override;
private:
    thrust::device_vector<float3> delta;
    thrust::device_vector<float> lambdas;
    // Size of the membrane's range inside `lambdas`; bending hinges and
    // stitches accumulate into the ranges that follow it.
    int membrane_lambda_slots = 0;
};
