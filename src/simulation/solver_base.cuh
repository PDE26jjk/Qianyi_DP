#pragma once
#include "simulator.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>


struct Picker {
    int tri_idx;
};

#define MAX_PICKER_POINTS_PER_CELL 15
#define HASH_TABLE_SIZE 1007     // The hash table can be made relatively small because there are few points.

#define MAX_POINTS_PER_CELL 7
struct PickerHashCell {
    int count;
    int point_indices[MAX_PICKER_POINTS_PER_CELL];
};
struct PointHashCell {
    int count;
    int point_indices[MAX_POINTS_PER_CELL];
};

struct CollisionResult_TP {
    int tri_idx;
    int vert_idx;
    float min_dist_sq;
    float3 normal;
};
struct CollisionResult_PP {
    int p1;
    int p2;
    int color;
    int status;
    CollisionResult_PP() = default;
    __device__ __host__ CollisionResult_PP(int vert_idx, int p_idx): p1(vert_idx), p2(p_idx) {
        color = -1;
        status = 0;
    }
};


struct SolverBase {
    ~SolverBase() = default;

public:
    thrust::device_vector<float3> vertices_2D;
    thrust::device_vector<float3> vertices_local;
    thrust::device_vector<int2> edges;
    thrust::device_vector<int3> triangles;
    thrust::device_vector<int> object_types;
    thrust::device_vector<int> vertex_index_offsets;
    thrust::device_vector<int> triangle_index_offsets;
    thrust::device_vector<int> vertices_obj;
    thrust::device_vector<float> mass;
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
    thrust::device_vector<PointHashCell> point_hash_table;
    thrust::device_vector<int2> point_hashes;
    thrust::device_vector<CollisionResult_TP> tp_collision_result; // point-triangle pairs
    thrust::device_vector<int2> ee_collision_result; // edge-edge pairs
    thrust::device_vector<CollisionResult_PP> pp_collision_result; // point-point pairs : p1,p2,color
    thrust::device_vector<uint64_t> sort_key_temp;
    thrust::device_vector<int> sort_key_temp2;
    thrust::device_vector<int> sort_value_temp;
    thrust::device_vector<uint64_t> vertex_colors;
    thrust::device_vector<int> pp_result_size;
    thrust::device_vector<int> tp_result_size;
    thrust::device_vector<int> sort_result_size;
    thrust::device_vector<int> hash_table_lookup;
    int max_pp_result_size;
    int max_tp_result_size;
    int max_sort_result_size;
    
    float dt;
    
    thrust::device_vector<float3> temp_vertices_f3;
    // thrust::device_vector<float3> temp_edge_f3;
    
    // bending
    thrust::device_vector<float4> IBM_q;

    // picker 
    thrust::universal_vector<int> pickers;
    thrust::device_vector<PickerHashCell> picker_hash_table;
    thrust::device_vector<CollisionResult_TP> picker_collision_result;
    thrust::universal_vector<thrust::pair<int, float3>> pick_triangles;
    thrust::universal_vector<Mat3> pick_triangle_offsets;


    virtual void init();
    void check_update_pick();
    void reset_pick_mask();
    virtual void update(float dt);

    void copy_vertices(float* ptr,bool world_space);

    SimulatorParams params;

// private: // can not be private for cuda lambda
    void sort_and_generate_edge_lookup();
    void generate_inverse_matrix();
    void generate_vertex_object();
    void init_triangle_index();
    void init_sewing();
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

    void check_sewing();

    virtual void contact_handle();
    void collision_Wu2021();
};
