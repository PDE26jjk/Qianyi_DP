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

    // Frame-stage timings of the last completed frame (see the
    // `frame-stage-timing` capability): a flat dict of milliseconds, one frame,
    // no averaging. `profile_timing` turns it on.
    static py::dict get_timing();

    // Cloth plasticity (see the `cloth-plasticity` capability): adopt the
    // current configuration as the rest shape, drop the accumulated plastic and
    // friction state, and read the state back. The readback is one row per bend
    // entry: rest angle, anchor angle, yield angle, stick timer, plastic timer.
    static void freeze_rest_shape();
    static void reset_plasticity();
    static py::array_t<float> get_plasticity_state();
    
    static py::dict check_point_attributes(int index);
    // Observability: solver convergence metrics of the last substep.
    // newton_initial / newton_final are the squared norms of the force
    // residual at the first and last outer iteration; linear_initial /
    // linear_final are the preconditioned residual of the last linear solve
    // and linear_relative their ratio.
    static py::dict get_residual_metrics();
    static py::dict check_edge_attributes(int p0, int p1);
    static py::dict check_edge_collision_data(int p0, int p1);
    static void on_exit();
};
