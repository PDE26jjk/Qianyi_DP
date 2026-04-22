#pragma once
#include <unordered_map>
#include <vector>
#include <thrust/device_vector.h>

#include "common/device.h"
#include "common/vec_math.h"

struct SimulatorParams;
struct SolverBase;

enum class MaskBit: char {
    fix_mask = 1 << 0,  // 1
    pick_mesh_mask = 1 << 1, // 2
    attach_mask = 1 << 2,  // 4
    proxy_mask = 1 << 3,  // 8
};

struct ObjectDataInput {
    float granularity; // mm
    float thickness;
    float friction;
    float3 stretch;
    float3 shear;
    float3 bending;
    float mass_densitys; // mass density per object, kg/m^2
    bool vertices_updated;
    bool matrix_updated;
    Mat4 new_matrix;
};

struct SewingData {
    int start_idx; // Start index of stitches
    int count; // Count of stitches
    float angle; // Dihedral angel of sewing line
    float compress; // Compress lenght of sewing line
};

struct SimulatorParams {
    int nb_all_objects;
    int nb_all_vertices;
    int nb_all_edges;
    int nb_all_triangles;
    int nb_all_stitches;
    int nb_all_cloth_vertices;
    int nb_all_cloth_edges;
    int nb_all_cloth_triangles;
    float cloth_edge_mean_length;
};
class Simulator {
private:
    Simulator();
    Simulator(const Simulator&) = delete;
    Simulator& operator=(const Simulator&) = delete;
    ~Simulator();

public:
    static Simulator& instance() {
        static Simulator s{};
        init_device();
        return s;
    }
    void init(const std::vector<float>& vertices, const std::vector<float>& vertices_sim,
        const std::vector<int>& edges,
        const std::vector<int>& triangles,
        const std::vector<float>& normals,
        const std::vector<int>& object_types,
        // const std::vector<float>& mass_densitys,
        const std::vector<ObjectDataInput>& object_data_inputs,
        const std::vector<Mat4>& world_matrixs,
        const std::vector<int>& _vertices_size,
        const std::vector<int>& triangle_index_offsets,
        const std::vector<float>& pin_fixed,
        const std::vector<float>& pin_attached,
        const std::vector<SewingData>& sewings,
        const std::vector<int2>& stitches,
        int nb_all_cloth_v, int nb_all_cloth_e, int nb_all_cloth_f);
    void update(float h);
    void copy_vertices(float*, bool world_space = false);
    void copy_debug_colors(float*);
    const SimulatorParams* get_params() const;
    void reset();

    int add_pick_triangle(int mesh_index, int tri_index, float3 position);
    void update_pick_triangle(int index, float3 position);
    void remove_pick_triangle(int index);
    void clear_pick_triangle();

    // return picker index
    int add_picker(float3 position);
    // return picked triangle index or -1
    void update_picker(int index, float3 position);
    void remove_picker(int index);
    void clear_picker();
    // set parameters
    void set_parameter(const std::string& key, float value);
    float get_parameter(const std::string& key, float default_value) const;
    void update_world_matrix(int obj_index, const std::vector<float>& matrix);
    std::vector<std::string> get_all_solver();
    void update_local_vertices(int obj_index, const std::vector<float>& vertices);
    void set_solver(const std::string& string);

private:
    SolverBase* m_solver;
    std::unordered_map<std::string, float> m_parameters;
    std::string m_last_solver_name;
    std::string m_solver_name = "Explicit";
    void create_solver();

};
