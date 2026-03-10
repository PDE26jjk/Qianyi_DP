#pragma once
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class SimulatorInterface {
public:
    static void print();
    static void input_data(py::dict input);
    static void update(float dt);
    static py::array_t<float> get_simulation_data(bool world_space);
    
    static int pick_triangle(int mesh_index, int tri_index, py::array_t<float> position);
    static void pick_triangle_update(int index, py::array_t<float> position);
    static void pick_triangle_remove(int index);

    static int add_picker(py::array_t<float> position);
    static void picker_update(int index, py::array_t<float> position);
    static void picker_remove(int index);
};
