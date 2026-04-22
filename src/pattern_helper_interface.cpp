#include "pattern_helper_interface.h"

#include "common/py_utils.h"
#include "geometry/pattern_helper.h"

void PatternHelperInterface::update_edges(py::array_t<float> edge_points, py::array_t<int> loop_sizes,py::array_t<float> loop_transforms) {
    auto edge_points_ = to_vector(edge_points);
    auto loop_sizes_ = to_vector(loop_sizes);
    auto loop_transforms_ = to_vector(loop_transforms);
    PatternHelper::Instance().update_edges(edge_points_, loop_sizes_, loop_transforms_);
}
py::dict PatternHelperInterface::find_nearest_edge(std::array<float, 2> query_point) {
    int res_index = 0;
    float res_weight = 0.f;
    PatternHelper::Instance().find_nearest_edge(query_point.data(),res_index,res_weight);
    py::dict result;
    result["res_index"] = res_index;
    result["res_weight"] = res_weight;
    return result;
}
py::dict PatternHelperInterface::check_edge_intersection(py::array_t<float> edge_points) {
    int res_index = 0;
    float res_weight = 0.f;
    auto edge_points_ = to_vector(edge_points);
    bool is_intersected = PatternHelper::Instance().check_edge_intersection(edge_points_,res_index,res_weight);
    py::dict result;
    result["intersected"] = is_intersected;
    result["res_index"] = res_index;
    result["res_weight"] = res_weight;
    return result;
}
