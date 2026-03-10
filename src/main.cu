#include <stdio.h>
#include <cuda_runtime.h>

#include "common/cuda_utils.h"
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// #include "simulation/simulator.h"
namespace py = pybind11;
#include "common/py_utils.h"
std::vector<int3> Delaunay2D(std::vector<float2>& pointVecIn, std::vector<int2>& constraintVec);

py::array_t<int> delaunay2D(py::array_t<float> pointVecIn, py::array_t<int> constraintVec) {
    auto pointVecIn_ = to_vector_cast<float2>(pointVecIn);
    auto constraintVec_ = to_vector_cast<int2>(constraintVec);

    auto res = Delaunay2D(pointVecIn_, constraintVec_);
    return to_py_vector<int3, int>(res, ShapeContainer({ (long long)res.size(), 3 }));
}
