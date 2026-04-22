#pragma once

#include <pybind11/numpy.h>

namespace py = pybind11;

class PatternHelperInterface {
public:
    static void update_edges(py::array_t<float> edge_points, py::array_t<int> loop_sizes, py::array_t<float> loop_transforms); // (N,2), (M,1), (M,16)
    static py::dict find_nearest_edge(std::array<float, 2> query_point);
    static py::dict check_edge_intersection(py::array_t<float> edge_points);
};

