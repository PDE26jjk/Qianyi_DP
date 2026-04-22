#include "simulator_interface.h"
#include <vector_types.h>
#include <simulation/simulator.h>

#include "common/py_utils.h"

void SimulatorInterface::print() {
    std::cout << "SimulatorInterface::print" << std::endl;
}
py::tuple SimulatorInterface::get_all_solver() {
    auto solvers = Simulator::instance().get_all_solver();
    py::tuple tup(solvers.size());
    for ( int i = 0; i < solvers.size(); ++i ) {
        tup[i] = solvers[i];
    }
    return tup;
}
void SimulatorInterface::set_solver(const std::string& solver_name) {
    Simulator::instance().set_solver(solver_name);
}
template<typename T>
void copy_and_add(std::vector<T>& dst, size_t offset, const py::array_t<T>& src, T to_add = T{}) {
    auto ptr = dst.data();
    auto buf = src.request();
    auto src_ptr = static_cast<T*>(buf.ptr);
    auto src_size = buf.size;
    for ( size_t i = offset, j = 0; i < offset + src_size; i++, j++ ) {
        ptr[i] = src_ptr[j] + to_add;
    }
}
template<typename T>
void copy_and_add(std::vector<T>& dst, size_t offset, const std::vector<T>& src, T to_add = T{}) {
    auto ptr = dst.data();
    auto src_ptr = src.data();
    auto src_size = src.size();
    for ( size_t i = offset, j = 0; i < offset + src_size; i++, j++ ) {
        ptr[i] = src_ptr[j] + to_add;
    }
}
static float3 to_float3(py::array_t<float> src) {
    float3 v;
    memcpy(&v, src.data(), sizeof(float3));
    return v;
}
void SimulatorInterface::input_data(py::dict input) {
    int nb_all_v{}, nb_all_e{}, nb_all_f{};
    auto mesh_list = input["mesh_list"].cast<py::list>();
    for ( auto mesh : mesh_list ) {
        nb_all_v += static_cast<int>(mesh["vertices"].cast<py::array>().size());
        nb_all_e += static_cast<int>(mesh["edges"].cast<py::array>().size());
        nb_all_f += static_cast<int>(mesh["triangles"].cast<py::array>().size());
    }
    int nb_all_o = (int)mesh_list.size();
    std::vector<float> vertices_init(nb_all_v);
    std::vector<float> vertices_simulation(nb_all_v);
    std::vector<int> edges(nb_all_e);
    std::vector<int> triangles(nb_all_f);
    std::vector<float> normals(nb_all_f);
    std::vector<int> object_types(nb_all_o);
    std::vector<float> pin_fixed(nb_all_v);
    std::vector<float> pin_attached(nb_all_v);
    // std::vector<float> mass_densitys(nb_all_o);
    std::vector<ObjectDataInput> obj_data(nb_all_o);
    std::vector<Mat4> world_matrixs(nb_all_o);

    int nb_all_cloth_v{ nb_all_v }, nb_all_cloth_e{ nb_all_e }, nb_all_cloth_f{ nb_all_f };
    nb_all_v = nb_all_e = nb_all_f = 0;
    std::vector<int> vertex_index_offsets(mesh_list.size());
    std::vector<int> triangle_index_offsets(mesh_list.size());

    for ( size_t i = 0; i < nb_all_o; i++ ) {
        auto mesh = mesh_list[i];
        auto _vertices = mesh["vertices"].cast<py::array_t<float>>();
        copy_and_add(vertices_init, nb_all_v, _vertices);
        if ( mesh.contains("vertices_sim") ) {
            _vertices = mesh["vertices_sim"].cast<py::array_t<float>>();
        }
        copy_and_add(vertices_simulation, nb_all_v, _vertices);
        vertex_index_offsets[i] = nb_all_v / 3;
        triangle_index_offsets[i] = nb_all_f / 3;
        int object_type = mesh["object_type"].cast<int>();
        // pybind11::print(nb_all_v);
        object_types[i] = object_type;
        if ( i > 0 && object_types[i - 1] == 0 && object_type != 0 ) {
            nb_all_cloth_v = nb_all_v;
            nb_all_cloth_e = nb_all_e;
            nb_all_cloth_f = nb_all_f;
            // local normal of cloth, should be (0,0,1)
            for ( size_t j = 2; j < nb_all_cloth_f; j += 3 ) {
                normals[j] = 1.f;
            }
        }
        if ( object_type != 0 ) {
            auto _normals = mesh["normals"].cast<py::array_t<float>>();
            copy_and_add(normals, nb_all_f, _normals);
        }
        else {
            auto _pin_fixed = mesh["fixed_vertices"].cast<py::array_t<float>>();
            copy_and_add(pin_fixed, nb_all_v / 3, _pin_fixed);
            auto _pin_attached = mesh["attached_vertices"].cast<py::array_t<float>>();
            copy_and_add(pin_attached, nb_all_v / 3, _pin_attached);
        }
        auto _edges = mesh["edges"].cast<py::array_t<int>>();
        copy_and_add(edges, nb_all_e, _edges, vertex_index_offsets[i]);
        auto _triangles = mesh["triangles"].cast<py::array_t<int>>();
        copy_and_add(triangles, nb_all_f, _triangles, vertex_index_offsets[i]);
        nb_all_v += (int)_vertices.size();
        nb_all_e += (int)_edges.size();
        nb_all_f += (int)_triangles.size();
        if ( object_type == 0 ) {
            obj_data[i].mass_densitys = mesh["mass"].cast<float>();
            // mass_densitys[i] = mesh["mass"].cast<float>();
            obj_data[i].granularity = mesh["granularity"].cast<float>();
            obj_data[i].friction = mesh["friction"].cast<float>();
            obj_data[i].thickness = mesh["thickness"].cast<float>();
            obj_data[i].stretch = to_float3(mesh["stretch"].cast<py::array_t<float>>());
            obj_data[i].shear = to_float3(mesh["shear"].cast<py::array_t<float>>());
            obj_data[i].bending = to_float3(mesh["bending"].cast<py::array_t<float>>());

        }
        else {
            obj_data[i].mass_densitys = 1.f;
        }
        auto world_matrix = mesh["world_matrix"].cast<py::array_t<float>>();
        Mat4 mat;
        memcpy(&mat.r, world_matrix.data(), sizeof(float) * 16);
        world_matrixs[i] = mat;
    }
    int nb_all_s{};
    auto sewings_dict = input["sewings"].cast<py::list>();

    for ( auto sewing : sewings_dict ) {
        auto value = sewing["stitches"].cast<py::array_t<int>>();
        assert(value.size() >= 2);
        nb_all_s += (int)value.size();
    }
    std::vector<int2> stitches(nb_all_s / 2);
    std::vector<SewingData> sewings(sewings_dict.size());
    nb_all_s = 0;
    int i = 0;
    for ( auto sewing : sewings_dict ) {
        auto stitches_dict = sewing["stitches"].cast<py::dict>();
        sewings[i].start_idx = nb_all_s;
        sewings[i].angle = sewing.contains("angle") ? sewing["angle"].cast<float>() : 0.f;
        sewings[i].compress = sewing.contains("compress") ? sewing["compress"].cast<float>() : 1.f;
        auto key = sewing["patterns"].cast<py::list>();
        int p1 = key[0].cast<int>();
        int p2 = key[1].cast<int>();
        auto value = to_vector_cast<int2>(sewing["stitches"].cast<py::array_t<int>>());
        copy_and_add(stitches, nb_all_s, value, make_int2(vertex_index_offsets[p1], vertex_index_offsets[p2]));
        nb_all_s += (int)value.size();
        sewings[i].count = nb_all_s - sewings[i].start_idx;
        i++;
    }
    auto& simulator = Simulator::instance();
    simulator.init(vertices_init, vertices_simulation,
        edges, triangles, normals, object_types, obj_data, world_matrixs,
        vertex_index_offsets,
        triangle_index_offsets, pin_fixed, pin_attached,
        sewings, stitches,
        nb_all_cloth_v, nb_all_cloth_e, nb_all_cloth_f);
}
void SimulatorInterface::update(float dt) {
    py::gil_scoped_release release;
    Simulator::instance().update(dt);
}
void SimulatorInterface::update_world_matrix(int index, py::array_t<float> matrix) {
    auto matrix_ = to_vector(matrix);
    Simulator::instance().update_world_matrix(index, matrix_);
}
void SimulatorInterface::update_local_vertices(int index, py::array_t<float> vertices) {
    auto vertices_ = to_vector(vertices);
    Simulator::instance().update_local_vertices(index, vertices_);
}
py::array_t<float> SimulatorInterface::get_simulation_data(bool world_space = false) {
    auto& simulator = Simulator::instance();
    auto result = py::array_t<float>(
        { (py::ssize_t)simulator.get_params()->nb_all_cloth_vertices, (py::ssize_t)3 });
    py::buffer_info buf = result.request();
    float* ptr = static_cast<float*>(buf.ptr);
    simulator.copy_vertices(ptr, world_space);
    return result;
}
py::array_t<float> SimulatorInterface::get_debug_colors() {
    auto& simulator = Simulator::instance();
    auto result = py::array_t<float>(
        { (py::ssize_t)simulator.get_params()->nb_all_cloth_vertices, (py::ssize_t)3 });
    py::buffer_info buf = result.request();
    float* ptr = static_cast<float*>(buf.ptr);
    simulator.copy_debug_colors(ptr);
    return result;
}
int SimulatorInterface::pick_triangle(int mesh_index, int tri_index, py::array_t<float> position) {
    py::buffer_info buf = position.request();
    float* ptr = static_cast<float*>(buf.ptr);
    auto pos = make_float3(ptr[0], ptr[1], ptr[2]);
    return Simulator::instance().add_pick_triangle(mesh_index, tri_index, pos);
}
void SimulatorInterface::pick_triangle_update(int index, py::array_t<float> position) {
    py::buffer_info buf = position.request();
    float* ptr = static_cast<float*>(buf.ptr);
    auto pos = make_float3(ptr[0], ptr[1], ptr[2]);
    Simulator::instance().update_pick_triangle(index, pos);
}
void SimulatorInterface::pick_triangle_remove(int index) {
    Simulator::instance().remove_pick_triangle(index);
}
int SimulatorInterface::add_picker(py::array_t<float> position) {
    py::buffer_info buf = position.request();
    float* ptr = static_cast<float*>(buf.ptr);
    auto pos = make_float3(ptr[0], ptr[1], ptr[2]);
    return Simulator::instance().add_picker(pos);
}
void SimulatorInterface::picker_update(int index, py::array_t<float> position) {
    py::buffer_info buf = position.request();
    float* ptr = static_cast<float*>(buf.ptr);
    auto pos = make_float3(ptr[0], ptr[1], ptr[2]);
    Simulator::instance().update_picker(index, pos);
}
void SimulatorInterface::picker_remove(int index) {
    Simulator::instance().remove_picker(index);
}
void SimulatorInterface::set_parameter(const std::string& key, float value) {
    Simulator::instance().set_parameter(key, value);
}
void SimulatorInterface::set_parameters(const std::unordered_map<std::string, float>& params) {
    for ( const auto& [fst, snd] : params ) {
        Simulator::instance().set_parameter(fst, snd);
    }
}
void SimulatorInterface::on_exit() {
    Simulator::instance().reset();
}
