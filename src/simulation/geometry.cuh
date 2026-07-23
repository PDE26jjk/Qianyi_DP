#pragma once
#include <mutex>

#include "simulator.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "contact/contact.cuh"
// #include <thrust/universal_vector.h>

struct Picker {
    int tri_idx;
};

#define MAX_PICKER_POINTS_PER_CELL 15
#define HASH_TABLE_SIZE 1007     // The hash table can be made relatively small because there are few points.

struct PickerHashCell {
    int count;
    int point_indices[MAX_PICKER_POINTS_PER_CELL];
};
struct CollisionResult_TP_Picker {
    int tri_idx;
    int vert_idx;
    float min_dist_sq;
};
struct AttachInfo {
    int face_idx;
    float u;
    float v;
    float d;
};
struct Geometry {
    virtual ~Geometry() = default;
    Geometry(Simulator* simulator): simulator(simulator), m_contact(Contact(this)) {}
    Simulator* simulator;

public:
    thrust::device_vector<float3> pos_2D; // vertex from pattern, z=0
    thrust::device_vector<float3> pos_local;
    thrust::device_vector<float3> pos_local_new_frame; // input every frame
    thrust::device_vector<float3> normals_input;
    thrust::device_vector<float3> edge_normals;
    thrust::device_vector<float3> vertex_normals;
    thrust::device_vector<int2> edges; // edge index [p0_idx,p1_idx]
    thrust::device_vector<int3> triangles; // triangle by edges index [e0_idx,e1_idx,e2_idx]
    thrust::device_vector<int3> triangle_indices; // triangle by vertex index [p0_idx,p1_idx,p2_idx]
    thrust::device_vector<int> object_types;
    thrust::device_vector<int> vertex_index_offsets;
    thrust::device_vector<int> edge_index_offsets;
    thrust::device_vector<int> triangle_index_offsets;
    thrust::device_vector<float> pin_fixed;
    thrust::device_vector<float> pin_attached;
    thrust::device_vector<AttachInfo> attach_data;
    thrust::device_vector<int> vertices_obj;
    thrust::device_vector<ObjectDataInput> obj_data;
    thrust::device_vector<float> mass_densitys; // mass density per object, kg/m^2
    thrust::device_vector<float> masses;
    thrust::device_vector<float> mass_inv;
    thrust::device_vector<Mat4> world_matrices;
    thrust::device_vector<Mat4> world_matrices_inv;
    thrust::device_vector<int2> dir_edges; // size = 2 * nb_all_edges,  [target, edge_id] per element
    thrust::device_vector<int2> edge_lookup; // size = nb_all_vertices, [offset, count] per vertex
    thrust::device_vector<int2> e2t;
    thrust::device_vector<int2> edge_opposite_points;
    thrust::device_vector<Mat2> Dms;
    thrust::device_vector<float> areas;
    thrust::device_vector<float3> pos_inertia;
    thrust::device_vector<float3> pos_world;
    thrust::device_vector<float3> pos_temp;
    thrust::device_vector<float3> pos_step_prev;
    thrust::device_vector<float3> pos_interpolation_old;
    thrust::device_vector<float3> pos_interpolation_new;
    thrust::device_vector<float3> debug_colors;

    thrust::device_vector<float3> forces;
    thrust::device_vector<float> edge_lengths;
    thrust::device_vector<float> static_diags;
    thrust::device_vector<float3> velocities;
    thrust::device_vector<float3> vel_prev;
    thrust::device_vector<char> vertices_mask;

    // sewing
    thrust::device_vector<SewingData> sewing_lines;
    thrust::device_vector<int2> stitches;
    thrust::device_vector<int> stitch_sewing; // stitch to sewing
    thrust::device_vector<int> vertex_proxy; // if proxy is valid, copy position from proxy on the end of updating.
    thrust::device_vector<int2> sewing_edges;
    thrust::device_vector<int2> sewing_e2t;
    thrust::device_vector<int2> sewing_edge_opposite_points;
    thrust::device_vector<int> stitches_done_count;
    thrust::device_vector<char> stitches_status;
    bool sewing_done;

    bool need_update_inv_mass;

    thrust::device_vector<float3> temp_vertices_f3;
    
    // bending
    thrust::device_vector<float4> IBM_q;
    bool need_record_interpolation_this_frame;
    bool need_update_interpolation_vertices_this_frame;
    bool has_pin_attached;
    bool has_pick_triangles_this_frame;
    
    float3 gravity;
    bool ground;

    // subspace
    thrust::device_vector<float> basis_weights; // save by 9*vertex
    thrust::device_vector<int> basis_indices; // save by 9*vertex
    thrust::device_vector<int> basis_index_offsets; // save by obj
    int basis_size;
    thrust::device_vector<float> H_red; // H Reduced dense save by obj, sum(M_obj^2)
    thrust::device_vector<float> M_red; // mass. The complete matrix is H_red + M_red / (h*h) , h may change every step.
    thrust::device_vector<int> H_red_offsets; // save by obj
    thrust::device_vector<int2> H2y;
    int H_red_total_sizes;
    
    void init(const GeoDataInput& geo);
    void compute_normals();
    void update_for_frame();
    void update_for_step(float h, float time_factor);
    void end_for_frame();
    void collision_detect() { m_contact.collision_detect(); }

    void upload_world_matrix(int obj_index, const std::vector<float>& matrix);
    void upload_local_vertices(int obj_index, const std::vector<float>& vertices);
    void copy_vertices(float* ptr, bool world_space);
    void copy_debug_colors(float* ptr);
    // void* get_device_temp_memory();

    SimulatorParams params;

    // init
    void init_vertex_data();
    void init_edge_data();
    void calc_edge_length();
    void init_triangle_data();
    void average_mass_by_cloth();
    void init_subspace();
    void precompute_subspace_H(const float* Jx_diag_pd, const float* Jx_nondiag_pd);

    // void collision_LCP_postprocess(float3* points_y);
    // void collision_LCP_postprocess_unified(float3* points_y);
    void collision_collect_near_pairs(float3* points, float max_dist, bool update_hash,
        bool collect_pp = false, bool collect_tp = false, bool collect_ee = false);
    // int color_constraints(int);

    float get_global_parameter(const std::string& key, float default_value) const;


    void check_update_pick();
    void reset_pick_mask();

    // void copy_debug_colors(float* ptr);

    // private: // can not be private due to cuda lambda
    void init_sewing();
    void init_pin();
    void init_picker();

    int add_pick_triangle(int mesh_index, int tri_index, float3 position);
    void update_pick_triangle(int index, float3 position);
    void remove_pick_triangle(int index);
    void clear_pick_triangle();

    // return picker index
    int add_picker(float3 position);
    void update_picker(int index, float3 position);
    void remove_picker(int index);
    void clear_picker();

    // run in update function
    void check_picker();

    void check_sewing(bool forced_connect = false);
    void accumulate_sewing_force();

    void update_pin(float3* q);

    // color graph for vbd
    void color_graph();
    thrust::device_vector<int> node_colors;
    thrust::device_vector<int> colors_index_offsets;
    std::vector<int> h_colors_index_offsets;
    thrust::device_vector<int> color_groups;
    
    // build adjacency data for vbd, vertices -> constraints
    void build_adj_data();
    thrust::device_vector<int2> v_adj_bending;
    thrust::device_vector<int> v_adj_bending_offsets;
    thrust::device_vector<int2> v_adj_tris;
    thrust::device_vector<int> v_adj_tris_offsets;
    
    // picker
    std::mutex pick_mutex;
    std::mutex picker_mutex;

    // thrust::universal_vector<int> pickers; // Never use universal_vector in multithreading!
    int max_pick_size = 1024;
    int picker_size;
    int pick_size;
    thrust::device_vector<int> pickers;
    thrust::device_vector<PickerHashCell> picker_hash_table;
    thrust::device_vector<CollisionResult_TP_Picker> picker_collision_result;
    thrust::device_vector<thrust::pair<int, float3>> pick_triangles;
    thrust::device_vector<Mat3> pick_triangle_offsets;

    Contact& get_contact(){return m_contact;}
    thrust::device_vector<float3> inertial_offset;
    float step_h;
private:
    Contact m_contact;
    size_t temp_mem_size;
    thrust::device_vector<char> temp_mem;
};
__global__ void forward_step(
    const float3* __restrict__ vel,
    const float3* __restrict__ vel_prev,
    const float* __restrict__ inv_mass,
    const float3* __restrict__ external_force,
    const char* __restrict__ mask,
    float3* __restrict__ pos,
    float3* __restrict__ inertia_out,
    float3* __restrict__ dx,
    float* __restrict__ static_diags,
    float dt,
    float mask_stiff,
    float3 gravity,
    bool warm_start,
    int num_vertices
);