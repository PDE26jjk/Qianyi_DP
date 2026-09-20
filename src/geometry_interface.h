#pragma once
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class GeometryInterface {
public:
    static py::tuple sample_points(py::array_t<float> boundary, py::array_t<int> edge_indices,
        py::array_t<int> curve_sizes, py::array_t<int> is_holes, float radius, int triangulator,
        float relax_gain1, int relax_iters1, float relax_gain2, int relax_iters2,
        float boundary_margin_cells);
    static py::tuple find_map_weight(py::array_t<float> map_points, py::array_t<int> map_tris, py::array_t<float> query_points,
        bool map_bounds);

    static py::array_t<int> delaunay_2d(py::array_t<float> pointVecIn, py::array_t<int> constraintVec);
};
