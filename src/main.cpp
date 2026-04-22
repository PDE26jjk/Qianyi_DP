#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
#include <filesystem>

#include "geometry_interface.h"
#include "pattern_helper_interface.h"
#include "common/device.h"
// #include <vector_types.h>
// #include <simulation/simulator.cuh>


int add(int i, int j) {
    return i + j;
}

#include <iostream>
#include <vector>

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "simulator_interface.h"
namespace py = pybind11;

std::string module_dir;
std::string get_module_dir(py::module_& m) {
    py::object file_attr = m.attr("__file__");
    if ( file_attr.is_none() ) {
        return ".";
    }
    std::string module_path = py::cast<std::string>(file_attr);
    return std::filesystem::path(module_path).parent_path().string();
}
PYBIND11_MODULE(Qianyi_DP, m) {
    m.doc() = R"pbdoc(
        Qianyi data processing
        -----------------------

        .. currentmodule:: Qianyi_DP

    )pbdoc";
    module_dir = get_module_dir(m);
    // std::string shader_path = (std::filesystem::path(module_dir) / "resources").string();

// 	m.def("add", &add, R"pbdoc(
//         Add two numbers
//
//         Some other explanation about the add function.
//     )pbdoc");
//
// 	m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
//         Subtract two numbers
//
//         Some other explanation about the subtract function.
//     )pbdoc");
//
// 	m.def("print_array", [](py::array_t<float> x) {
// 		auto buf = x.request();
// 		float* ptr1 = static_cast<float*>(buf.ptr);
// 		for (size_t i = 0; i < (int)buf.shape[0]; i++) {
// 			ptr1[i] += 2;
// 		}
// 	}, py::arg().noconvert());
    m.def("print_test", []() {
        py::print("Hello World!2");
    });
    py::class_<GeometryInterface>(m, "geometry")
        .def_static("sample_points", &GeometryInterface::sample_points,
            py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert())
        .def_static("sample_points_dbg", &GeometryInterface::sample_points_dbg,
            py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert())
        .def_static("delaunay_2d", &GeometryInterface::delaunay_2d,
            py::arg("pointVecIn").noconvert(), py::arg("constraintVec").noconvert())
        .def_static("find_map_weight", &GeometryInterface::find_map_weight,
            py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert());
    
    py::class_<PatternHelperInterface>(m, "pattern_helper")
        .def_static("update_edges", &PatternHelperInterface::update_edges,
            py::arg("edge_points").noconvert(), py::arg("loop_sizes").noconvert(), py::arg("loop_sizes").noconvert())
        .def_static("find_nearest_edge", &PatternHelperInterface::find_nearest_edge,
            py::arg("query_point"))
        .def_static("check_edge_intersection", &PatternHelperInterface::check_edge_intersection,
            py::arg("edge_points").noconvert());

    py::class_<SimulatorInterface>(m, "simulator")
        .def_static("print", &SimulatorInterface::print)
        .def_static("get_all_solver", &SimulatorInterface::get_all_solver)
        .def_static("set_solver", &SimulatorInterface::set_solver)
        .def_static("input_data", &SimulatorInterface::input_data)
        .def_static("update", &SimulatorInterface::update)
        .def_static("update_world_matrix", &SimulatorInterface::update_world_matrix)
        .def_static("update_local_vertices", &SimulatorInterface::update_local_vertices)
        .def_static("get_simulation_data", &SimulatorInterface::get_simulation_data,
            py::arg("world_space") = false)
        .def_static("get_debug_colors", &SimulatorInterface::get_debug_colors)
        .def_static("pick_triangle", &SimulatorInterface::pick_triangle)
        .def_static("pick_triangle_update", &SimulatorInterface::pick_triangle_update)
        .def_static("pick_triangle_remove", &SimulatorInterface::pick_triangle_remove)
        .def_static("add_picker", &SimulatorInterface::add_picker)
        .def_static("picker_update", &SimulatorInterface::picker_update)
        .def_static("picker_remove", &SimulatorInterface::picker_remove)
        .def_static("set_parameter", &SimulatorInterface::set_parameter,
            py::arg("key"), py::arg("value"))
        .def_static("set_parameters", &SimulatorInterface::set_parameters,
            py::arg("params"))
        .def_static("on_exit", &SimulatorInterface::on_exit);


    // m.def("sample_points", [](py::array_t<float> boundy, py::array_t<int> next_pt, float radius) {
    //     auto buf = boundy.request();
    //     auto buf2 = next_pt.request();
    //     if ( buf.ndim != 2 ) {
    //         throw std::runtime_error("Points array must be 2-dimensional");
    //     }
    //
    //     if ( buf.shape[1] != 2 ) {
    //         throw std::runtime_error("Points array must have shape (N, 2)");
    //     }
    //
    //     auto boundy_ = to_vector(boundy);
    //     auto nextpt_ = to_vector(next_pt);
    //     auto res = test(boundy_, nextpt_, radius);
    //     return to_py_vector(res, ShapeContainer({ (long long)res.size() / 2, 2 }));
    //
    // }, py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert());
    // m.def("delaunay2D", &delaunay2D, py::arg("pointVecIn").noconvert(),
    //     py::arg("constraintVec").noconvert());


#ifdef VERSION_INFO
	m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
