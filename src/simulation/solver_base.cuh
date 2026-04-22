#pragma once
#include <mutex>

#include "simulator.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>

#include "contact/collision_type.cuh"


// struct PatternData {
//     float granularity;
//     float thickness;
//     float friction;
// };

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
struct AutoGPUmem {
    SolverBase* pool;
    int* ptr;
    ~AutoGPUmem();
};

struct AttachInfo {
    int face_idx;
    float u;
    float v;
    float d;
};

struct SolverBase {
    virtual ~SolverBase() = default;
    SolverBase(Simulator* simulator):simulator(simulator){}
    Simulator* simulator;
public:
    thrust::device_vector<float3> vertices_2D;
    thrust::device_vector<float3> vertices_local;
    thrust::device_vector<float3> vertices_local_new_frame;
    thrust::device_vector<float3> normals_input;
    thrust::device_vector<int2> edges;
    thrust::device_vector<int3> triangles;
    thrust::device_vector<int> object_types;
    thrust::device_vector<int> vertex_index_offsets;
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
    thrust::device_vector<int2> dir_edges;
    thrust::device_vector<int2> edge_lookup;
    thrust::device_vector<int3> triangle_indices;
    thrust::device_vector<int2> e2t;
    thrust::device_vector<int2> edge_opposite_points;
    thrust::device_vector<Mat2> Dms;
    thrust::device_vector<float3> vertices_old;
    thrust::device_vector<float3> vertices_world;
    thrust::device_vector<float3> vertices_new;
    thrust::device_vector<float3> debug_colors;

    thrust::device_vector<float3> forces;
    thrust::device_vector<float> edge_lengths;
    thrust::device_vector<float3> velocities;
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

    // collision or contact
    uint32_t point_hash_table_size;
    uint32_t edge_hash_table_size;
    thrust::device_vector<PointHashCell> point_hash_table;
    thrust::device_vector<int2> point_hashes;
    thrust::device_vector<int2> edge_hashes;
    thrust::device_vector<CollisionResult_TP> tp_collision_result; // point-triangle pairs
    thrust::device_vector<CollisionResult_EE> ee_collision_result; // edge-edge pairs
    thrust::device_vector<CollisionResult_PP> pp_collision_result; // point-point pairs : p1,p2,color
    // thrust::device_vector<uint64_t> sort_key_temp;
    thrust::device_vector<int> sort_key_temp;
    thrust::device_vector<int> pp_result_size;// one size
    int pp_result_size_h;
    thrust::device_vector<int> tp_result_size;// one size
    int tp_result_size_h;
    thrust::device_vector<int> ee_result_size;// one size
    int ee_result_size_h;

    thrust::device_vector<int> sort_result_size;// one size
    int point_hashes_size_h;
    int edge_hashes_size_h;
    thrust::device_vector<int> point_hash_table_lookup;
    thrust::device_vector<int> edge_hash_table_lookup;

    thrust::device_vector<CollisionConstraint> collision_constraints;
    thrust::device_vector<UnifiedNormalConstraint> normal_constraints;
    std::vector<int> constraint_color_offsets;
    thrust::device_vector<int> vertex_color_claimer;
    thrust::device_vector<int> uncolored_count; // one size
    thrust::device_vector<int> constraint_colors;
    thrust::device_vector<uint64_t> vertex_forbidden_masks;
    cudaGraph_t current_graph = nullptr;
    cudaGraphExec_t current_graph_exec = nullptr;
    cudaStream_t capture_stream = nullptr;

    thrust::device_vector<float3> points_safe;
    thrust::device_vector<float> weight;

    int max_pp_result_size;
    int max_tp_result_size;
    int max_ee_result_size;
    int max_point_hashes_size;
    int max_edge_hashes_size;
    int max_sort_result_size;
    int max_collision_constraints_size;
    
    float dt;
    int frame;

    thrust::device_vector<float3> temp_vertices_f3;
    thrust::device_vector<float3> temp_vertices2_f3;
    thrust::device_vector<float3> temp_vertices3_f3;
    // thrust::device_vector<float3> temp_edge_f3;

    // bending
    thrust::device_vector<float4> IBM_q;

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

    virtual void init();
    virtual void update(float dt_rest);
    
    void check_update_pick();
    void reset_pick_mask();

    void update_world_matrix(int obj_index, const std::vector<float>& matrix);
    void update_local_vertices(int obj_index, const std::vector<float>& vertices);
    void copy_vertices(float* ptr, bool world_space);
    void copy_debug_colors(float* ptr);

    SimulatorParams params;

// private: // can not be private for cuda lambda
    void sort_and_generate_edge_lookup();
    void generate_inverse_matrix();
    void generate_vertex_object();
    void init_triangle_data();
    void init_sewing();
    void init_pin();
    void init_collision();

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
    
    void update_pin(float3* q);

    void collision_LCP_postprocess(float3* points_y);
    void collision_LCP_postprocess_unified(float3* points_y);
    virtual void contact_handle();
    void collision_collect_near_pairs(float3* points, float max_dist, bool update_hash,
        bool collect_pp = false, bool collect_tp = false, bool collect_ee = false);
    int color_constraints(int);
    float3* collision_Wu2021_step(float3* y, bool first, bool& done);
    float alpha_hard;
    
    float vector_field_dot(const float3* a, const float3* b);

    float get_global_parameter(const std::string& key, float default_value) const;
private:
    thrust::device_vector<int> pool;
    std::vector<bool> pool_used;
    AutoGPUmem alloc_pool();
    void dealloc_pool(void* p);
    friend AutoGPUmem;
};
