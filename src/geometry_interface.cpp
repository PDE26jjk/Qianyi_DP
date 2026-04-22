#include "geometry_interface.h"

#include "common/py_utils.h"
#include "geometry/triangulator.h"

void sample_points_impl(std::vector<float>& boundary, std::vector<int>& next_pt, float radius,
    std::vector<float>& output_points, std::vector<int>& output_tris);
std::vector<unsigned char> sample_points_debug(std::vector<float>& boundary, std::vector<int>& next_pt, float radius);
std::vector<int> sample_points_debug2(std::vector<float>& boundary, std::vector<float>& points);
std::vector<int> sample_points_debug3(std::vector<float>& boundary, std::vector<int>& next_pt, float radius);

py::tuple GeometryInterface::sample_points(py::array_t<float> boundy, py::array_t<int> next_pt, float radius) {
    //test(boundy.)
    // py::object print = py::module::import("builtins").attr("print");
    auto buf = boundy.request();
    auto buf2 = next_pt.request();
    if ( buf.ndim != 2 ) {
        throw std::runtime_error("Points array must be 2-dimensional");
    }

    if ( buf.shape[1] != 2 ) {
        throw std::runtime_error("Points array must have shape (N, 2)");
    }

    auto boundy_ = to_vector(boundy);
    auto nextpt_ = to_vector(next_pt);
    std::vector<float> points;
    std::vector<int> tris;
    sample_points_impl(boundy_, nextpt_, radius, points, tris);
    return py::make_tuple(
        to_py_vector(points, ShapeContainer({ (long long)points.size() / 2, 2 })),
        to_py_vector(tris, ShapeContainer({ (long long)tris.size() / 3, 3 }))
        );
}

py::array_t<unsigned char> GeometryInterface::sample_points_dbg(py::array_t<float> boundy, py::array_t<int> next_pt,
    float radius) {
    auto buf = boundy.request();
    auto buf2 = next_pt.request();

    auto boundy_ = to_vector(boundy);
    auto next_pt_ = to_vector(next_pt);
    // auto res = sample_points_debug2(boundy_, points_);
    auto res = sample_points_debug(boundy_, next_pt_, radius);
    return to_py_vector(res);
}

py::array_t<int> GeometryInterface::delaunay_2d(py::array_t<float> pointVecIn, py::array_t<int> constraintVec) {
    auto pointVecIn_ = to_vector(pointVecIn);
    auto constraintVec_ = to_vector(constraintVec);

    auto res = triangulator::delaunay(pointVecIn_, constraintVec_);
    return to_py_vector(res, ShapeContainer({ (long long)res.size() / 3, 3 }));
}
void find_map_weight_impl(
    const std::vector<float>& map_points,
    const std::vector<int>& map_tris,
    const std::vector<float>& query_points,
    std::vector<int>& res_index, std::vector<float>& res_weight
);
py::tuple GeometryInterface::find_map_weight(py::array_t<float> map_points, py::array_t<int> map_tris,
    py::array_t<float> query_points) {
    auto map_points_ = to_vector(map_points);
    auto map_tris_ = to_vector(map_tris);
    auto query_points_ = to_vector(query_points);
    std::vector<int> res_index;
    std::vector<float> res_weight;
    find_map_weight_impl(map_points_, map_tris_, query_points_, res_index, res_weight);
    return py::make_tuple(
        to_py_vector(res_index),
        to_py_vector(res_weight, ShapeContainer({ (long long)res_weight.size() / 3, 3 }))
        );
}
