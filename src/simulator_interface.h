#pragma once
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class SimulatorInterface {
public:
    static void print();
    
    static py::tuple get_all_solver();
    static void set_solver(const std::string& solver_name);
    
    static void input_data(py::dict input);
    static void update(float dt);
    static void update_world_matrix(int index, py::array_t<float> matrix);
    static void update_local_vertices(int index, py::array_t<float> vertices);
    static py::array_t<float> get_simulation_data(bool world_space);
    static py::array_t<float> get_debug_colors();
    
    static int pick_triangle(int mesh_index, int tri_index, py::array_t<float> position);
    static void pick_triangle_update(int index, py::array_t<float> position);
    static void pick_triangle_remove(int index);

    static int add_picker(py::array_t<float> position);
    static void picker_update(int index, py::array_t<float> position);
    static void picker_remove(int index);
    
    static void set_parameter(const std::string& key, float value);
    static void set_parameters(const std::unordered_map<std::string, float>& params);
    
    static void on_exit();
};
