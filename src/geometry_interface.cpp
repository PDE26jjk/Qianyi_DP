#include "geometry_interface.h"

#include "common/py_utils.h"
#include "geometry/triangulator.h"

void sample_points_impl(std::vector<float>& boundary, std::vector<int>& edge_indices_flat,
      std::vector<int>& curve_sizes, std::vector<int>& is_holes_int,
      float radius, std::vector<float>& output_points,
      std::vector<int>& output_tris, int triangulator,
      float relax_gain1, int relax_iters1, float relax_gain2, int relax_iters2,
      float boundary_margin_cells);
  
  py::tuple GeometryInterface::sample_points(py::array_t<float> boundary, py::array_t<int> edge_indices,
      py::array_t<int> curve_sizes, py::array_t<int> is_holes, float radius, int triangulator,
      float relax_gain1, int relax_iters1, float relax_gain2, int relax_iters2,
      float boundary_margin_cells) {
    auto buf = boundary.request();
    if ( buf.ndim != 2 ) {
        throw std::runtime_error("Points array must be 2-dimensional");
    }

    if ( buf.shape[1] != 2 ) {
        throw std::runtime_error("Points array must have shape (N, 2)");
    }

    auto boundary_ = to_vector(boundary);
    auto edge_indices_ = to_vector(edge_indices);
    auto curve_sizes_ = to_vector(curve_sizes);
    auto is_holes_ = to_vector(is_holes);
    std::vector<float> points;
    std::vector<int> tris;
      sample_points_impl(boundary_, edge_indices_, curve_sizes_, is_holes_, radius, points, tris,
          triangulator, relax_gain1, relax_iters1, relax_gain2, relax_iters2,
          boundary_margin_cells);
    return py::make_tuple(
        to_py_vector(points, ShapeContainer({ (long long)points.size() / 2, 2 })),
        to_py_vector(tris, ShapeContainer({ (long long)tris.size() / 3, 3 }))
        );
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
    std::vector<int>& res_index, std::vector<float>& res_weight,
    bool map_bounds
);
py::tuple GeometryInterface::find_map_weight(py::array_t<float> map_points, py::array_t<int> map_tris,
    py::array_t<float> query_points, bool map_bounds) {
    auto map_points_ = to_vector(map_points);
    auto map_tris_ = to_vector(map_tris);
    auto query_points_ = to_vector(query_points);
    std::vector<int> res_index;
    std::vector<float> res_weight;
    find_map_weight_impl(map_points_, map_tris_, query_points_, res_index, res_weight, map_bounds);
    return py::make_tuple(
        to_py_vector(res_index),
        to_py_vector(res_weight, ShapeContainer({ (long long)res_weight.size() / 3, 3 }))
        );
}
