#include "test.h"

#include <pybind11/iostream.h>

#include "benchmarks.h"
namespace py = pybind11;

void register_test_bindings(pybind11::module_& m) {
    py::add_ostream_redirect(m, "ostream_redirect");
    auto test_ = m.def_submodule("test", "test module");
    test_.def("version", []() { return "v1"; });
    test_.def("test_sort", &sort_benchmark,
        py::arg("sizes"), py::arg("warmup") = 2, py::arg("runs") = 10, py::arg("verify") = false,
        "Execute Thrust and CUB sort benchmarks. Results are stored internally.");

    test_.def("bvh_benchmark", &bvh_benchmark,
        py::arg("vertices_in"), py::arg("faces_in"), py::arg("query_points"), py::arg("warmup") = 2, py::arg("runs") = 10,
        py::arg("verify") = false);

    test_.def("bvh2_benchmark", &bvh2_benchmark,
        py::arg("vertices_in"), py::arg("faces_in"), py::arg("query_points"), py::arg("warmup") = 2, py::arg("runs") = 10,
        py::arg("verify") = false);

    test_.def("bvh_edge_benchmark", &bvh_edge_benchmark,
        py::arg("vertices_in"), py::arg("edges_in"), py::arg("query_points"), py::arg("warmup") = 2, py::arg("runs") = 10,
        py::arg("verify") = false);

    test_.def("sdf_benchmark", &sdf_benchmark,
        py::arg("vertices_in"), py::arg("faces_in"), py::arg("use_parity") = true, py::arg("warmup") = 2, py::arg("runs") = 10);

    test_.def("sdf_check_inside", &sdf_check_inside,
        py::arg("vertices_in"), py::arg("faces_in"), py::arg("query_points"), py::arg("warmup") = 2, py::arg("runs") = 10);

    test_.def("graph_coloring_benchmark", &graph_coloring_benchmark,
        py::arg("num_nodes"), py::arg("edges"), py::arg("algorithm"),
        py::arg("balance") = false, py::arg("target_max_min_ratio") = 1.1,
        py::arg("warmup") = 2, py::arg("runs") = 10,
        py::arg("verify") = true);
}
