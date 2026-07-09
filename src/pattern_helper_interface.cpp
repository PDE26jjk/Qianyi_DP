#include "pattern_helper_interface.h"

#include "common/py_utils.h"
#include "geometry/pattern_helper.h"

void PatternHelperInterface::update_edges(py::array_t<float> edge_points, py::array_t<int> loop_sizes,
    py::array_t<float> loop_transforms) {
    auto edge_points_ = to_vector(edge_points);
    auto loop_sizes_ = to_vector(loop_sizes);
    auto loop_transforms_ = to_vector(loop_transforms);
    PatternHelper::Instance().update_edges(edge_points_, loop_sizes_, loop_transforms_);
}
py::dict PatternHelperInterface::find_nearest_edge(std::array<float, 2> query_point) {
    int res_index = 0;
    float res_weight = 0.f;
    PatternHelper::Instance().find_nearest_edge(query_point.data(), res_index, res_weight);
    py::dict result;
    result["res_index"] = res_index;
    result["res_weight"] = res_weight;
    return result;
}
py::dict PatternHelperInterface::check_edge_intersection(py::array_t<float> edge_points) {
    int res_index = 0;
    float res_weight = 0.f;
    auto edge_points_ = to_vector(edge_points);
    bool is_intersected = PatternHelper::Instance().check_edge_intersection(edge_points_, res_index, res_weight);
    py::dict result;
    result["intersected"] = is_intersected;
    result["res_index"] = res_index;
    result["res_weight"] = res_weight;
    return result;
}

py::list PatternHelperInterface::get_all_intersections(py::array_t<float> edge_points, py::array_t<int> curve_sizes,
    py::array_t<int8_t> is_loops, py::array_t<int> num_sections_per_curve,  // 新增：每个曲线的 section 数量
    py::array_t<int>& section_sizes) {

    auto edge_points_ = to_vector(edge_points);
    auto curve_sizes_ = to_vector(curve_sizes);
    auto is_loops_ = to_vector(is_loops);
    auto num_sections_per_curve_ = to_vector(num_sections_per_curve);
    auto section_sizes_ = to_vector(section_sizes);
    auto data = PatternHelper::Instance().get_all_intersections(edge_points_, curve_sizes_, is_loops_, num_sections_per_curve_,
        section_sizes_);
    py::list result;

    for ( auto& [curve_a,section_a, t_a, curve_b,section_b, t_b, state] : data ) {
        py::dict d;
        d["curve_a"] = curve_a;
        d["section_a"] = section_a;
        d["t_a"] = t_a;
        d["curve_b"] = curve_b;
        d["section_b"] = section_b;
        d["t_b"] = t_b;
        d["state"] = state;
        result.append(d);
    }

    return result;
}

py::tuple PatternHelperInterface::deduplicate_points(py::array_t<float> point_data, float threshold) {
    auto point_data_ = to_vector(point_data);
    std::vector<float> out_point_data;
    std::vector<unsigned int> out_map;
    PatternHelper::Instance().deduplicate_points(
        point_data_, threshold, out_point_data, out_map);
    return py::make_tuple(to_py_vector(out_point_data), to_py_vector(out_map));
}
