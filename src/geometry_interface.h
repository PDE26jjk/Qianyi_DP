#pragma once
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class GeometryInterface {
public:
    static py::tuple sample_points(py::array_t<float> boundy, py::array_t<int> next_pt, float radius);
    static py::tuple find_map_weight(py::array_t<float> map_points, py::array_t<int> map_tris, py::array_t<float> query_points);
    static py::array_t<unsigned char> sample_points_dbg(py::array_t<float> boundy, py::array_t<int> next_pt, float radius);

    static py::array_t<int> delaunay_2d(py::array_t<float> pointVecIn, py::array_t<int> constraintVec);
};

