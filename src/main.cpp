#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
// #include <vector_types.h>
// #include <simulation/simulator.cuh>


int add(int i, int j) {
    return i + j;
}

#include <iostream>
#include <vector>

std::vector<float> test(std::vector<float>& boundary, std::vector<int>& next_pt, float radius);
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "simulator_interface.h"
namespace py = pybind11;
py::array_t<int> delaunay2D(py::array_t<float> pointVecIn, py::array_t<int> constraintVec);

#include "common/py_utils.h"

PYBIND11_MODULE(Qianyi_DP, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: Qianyi_DP

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

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

    py::class_<SimulatorInterface>(m, "simulator")
        .def_static("print", &SimulatorInterface::print)
        .def_static("input_data", &SimulatorInterface::input_data)
        .def_static("update", &SimulatorInterface::update)
        .def_static("get_simulation_data", &SimulatorInterface::get_simulation_data,
            py::arg("world_space") = false)
        .def_static("pick_triangle", &SimulatorInterface::pick_triangle)
        .def_static("pick_triangle_update", &SimulatorInterface::pick_triangle_update)
        .def_static("pick_triangle_remove", &SimulatorInterface::pick_triangle_remove)
        .def_static("add_picker", &SimulatorInterface::add_picker)
        .def_static("picker_update", &SimulatorInterface::picker_update)
        .def_static("picker_remove", &SimulatorInterface::picker_remove);


    m.def("sample_points", [](py::array_t<float> boundy, py::array_t<int> next_pt, float radius) {

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
        auto res = test(boundy_, nextpt_, radius);
        return to_py_vector(res, ShapeContainer({ (long long)res.size() / 2, 2 }));

    }, py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert());
    m.def("delaunay2D", &delaunay2D, py::arg().noconvert(), py::arg().noconvert());


#ifdef VERSION_INFO
	m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
