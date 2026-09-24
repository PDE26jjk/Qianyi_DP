#pragma once
#include <unordered_map>
#include <vector>
// #include <thrust/device_vector.h>

#include "common/device.h"
#include "common/vec_math.h"
#include "frame_timing.cuh"

struct SimulatorParams;
struct SolverBase;
struct Geometry;

enum class MaskBit: char {
    fix_mask = 1 << 0,  // 1
    pick_mesh_mask = 1 << 1, // 2
    attach_mask = 1 << 2,  // 4
    proxy_mask = 1 << 3,  // 8
};

struct ObjectDataInput {
    float granularity; // m
    float thickness; // m
    float friction;
    int collision_layer;
    float grain_dir; // pattern grain direction
    float3 stretch;
    // float3 shear;
    float3 bending;
    float mass_densitys; // mass density per object, kg/m^2
    // External forces (see the `external-forces` capability). `pressure` is a
    // constant load along the surface normal in Pa: positive pushes along the
    // stored normal, negative pulls inward, and 0 is inert. The aerodynamic
    // coefficients override the global parameters per object; a negative value
    // means "use the global parameter".
    float pressure;
    float wind_drag;
    float wind_lift;
    // Plasticity opt-in (see the `cloth-plasticity` capability). A panel that
    // does not set it keeps its input rest angle: no plastic rest-angle offset
    // and no internal-friction anchor offset are applied to its bend entries.
    bool plastic;
    bool kinetic; // The solver does not update its position, todo set mass for it
    bool vertices_updated;
    bool matrix_updated;
    Mat4 new_matrix;
};


struct SewingData {
    int start_idx; // Start index of stitches
    int count; // Count of stitches
};

struct GeoDataInput {
    std::vector<float> vertices;
    std::vector<float> vertices_sim;
    std::vector<int> edges;
    std::vector<int> triangles;
    std::vector<float> normals;
    std::vector<int> object_types;
    std::vector<ObjectDataInput> obj_data_input;
    std::vector<Mat4> world_matrices;
    std::vector<int> vertex_index_offsets;
    std::vector<int> edge_index_offsets;
    std::vector<int> triangle_index_offsets;
    std::vector<float> pin_fixed;
    std::vector<float> pin_attached;
    std::vector<SewingData> sewings;
    std::vector<int2> stitches;
    // Rest-shape input, per edge over the whole scene (0 where an object does
    // not provide a value). `edge_rest_angle` is the rest dihedral angle of the
    // edge's bend entry (0 = flat); `edge_compress` is the relative change of
    // the edge's rest length (0 = keep the pattern length).
    std::vector<float> edge_rest_angle;
    std::vector<float> edge_compress;
    int nb_all_cloth_v; int nb_all_cloth_e; int nb_all_cloth_f;
    int nb_all_cloth_o;
};

struct SimulatorParams {
    int nb_all_objects;
    int nb_all_cloth_objects;
    int nb_all_vertices;
    int nb_all_edges;
    int nb_all_triangles;
    int nb_all_stitches;
    int nb_all_cloth_vertices;
    int nb_all_cloth_edges;
    int nb_all_cloth_triangles;
    float cloth_edge_mean_length;
};

struct CheckPointData {
    float mass;
    float3 force;
    float3 force_elastic;
    float3 pos_prev;
    float3 pos_world;
    float3 pos_pred;
    std::vector<int> nearby_faces;
};
struct CheckEdgeData {
    std::vector<int> nearby_edges;
    std::vector<int> nearby_faces;
    float3 normal;
    int2 tris;
};
struct CheckEdgeCollisionData {
    std::vector<int> nearby_edges;
    std::vector<int> valid;
    std::vector<float3> forces;
    std::vector<float2> st;
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
    void init(const GeoDataInput& geo);
    void update(float h);
    // Frame-stage timings of the most recent completed frame (see the
    // `frame-stage-timing` capability). Enabled by the `profile_timing`
    // parameter; returns a disabled, empty sample when it is off.
    FrameTiming::Snapshot get_timing();
    // The frame timer itself, for a solver that wants to bracket a piece of its
    // own loop (see `FrameTiming::begin_accum`).
    FrameTiming& timing() { return m_timing; }
    void copy_vertices(float*, bool world_space = false);
    void copy_debug_colors(float*);
    const SimulatorParams* get_geo_params() const;
    Geometry* get_geo() const;
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
    // Bumped by every set_parameter call. Callers that bake host parameters
    // into a captured CUDA graph (see pd_cuda_graph) watch this to know when
    // the capture has to be rebuilt.
    uint64_t parameter_version() const { return m_parameter_version; }
    void update_world_matrix(int obj_index, const std::vector<float>& matrix);
    std::vector<std::string> get_all_solver();
    // Cloth plasticity (see the `cloth-plasticity` capability): adopt the
    // current configuration as the rest shape, drop the accumulated plastic and
    // friction state, and read the state back per bend entry.
    void freeze_rest_shape();
    void reset_plasticity();
    // Number of bend entries the plasticity state covers (0 before a scene is
    // loaded).
    int plasticity_state_size() const;
    void copy_plasticity_state(float* out) const;
    // Observability: the solver's convergence metrics (see
    // SimulatorInterface::get_residual_metrics for the named layout).
    std::vector<float> get_residual_metrics();
    void update_local_vertices(int obj_index, const std::vector<float>& vertices);
    void set_solver(const std::string& string);

    CheckPointData get_check_point_data(int index) const;
    CheckEdgeData get_check_edge_data(int p0, int p1) const;
    CheckEdgeCollisionData get_check_edge_collision_data(int p0, int p1) const;

    float dt;
    int frame;
private:
    SolverBase* m_solver;
    Geometry* m_geo;
    std::unordered_map<std::string, float> m_parameters;
    uint64_t m_parameter_version = 0;
    std::string m_last_solver_name;
    std::string m_solver_name = "PDNewton";
    FrameTiming m_timing;
    void create_solver();
    
};
