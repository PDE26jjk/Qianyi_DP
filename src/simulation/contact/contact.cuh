#pragma once
#include <thrust/device_vector.h>
#include "collision_type.cuh"
#include "lbvh.cuh"

constexpr unsigned int broad_phase_size = 16;
struct Contact {
    
    Geometry* geo;
    
    Contact(Geometry* geo): geo(geo) {}
    void init();
    void collision_detect_broad_phase(const float3* pos, const float3* offset);
    void collision_detect();
    void compute_inertial_offset();
    void rebuild_bvh();
    void refit_bvh();
    void refit_bvh(const float3* pos, const float3* offset);
    void accumulate_contact_force(float3* forces, Mat3* Jx_diag);
    // // contact
    // uint32_t point_hash_table_size;
    // uint32_t edge_hash_table_size;
    // thrust::device_vector<PointHashCell> point_hash_table;
    // thrust::device_vector<int2> point_hashes;
    // thrust::device_vector<int2> edge_hashes;
    // thrust::device_vector<CollisionResult_TP> tp_collision_result; // point-triangle pairs
    // thrust::device_vector<CollisionResult_EE> ee_collision_result; // edge-edge pairs
    // thrust::device_vector<CollisionResult_PP> pp_collision_result; // point-point pairs : p1,p2,color
    // // thrust::device_vector<uint64_t> sort_key_temp;
    // thrust::device_vector<int> sort_key_temp;
    // thrust::device_vector<int> pp_result_size;// one size
    // int pp_result_size_h;
    // thrust::device_vector<int> tp_result_size;// one size
    // int tp_result_size_h;
    // thrust::device_vector<int> ee_result_size;// one size
    // int ee_result_size_h;
    //
    // thrust::device_vector<int> sort_result_size;// one size
    // int point_hashes_size_h;
    // int edge_hashes_size_h;
    // thrust::device_vector<int> point_hash_table_lookup;
    // thrust::device_vector<int> edge_hash_table_lookup;
    //
    // thrust::device_vector<CollisionConstraint> collision_constraints;
    // thrust::device_vector<UnifiedNormalConstraint> normal_constraints;
    // std::vector<int> constraint_color_offsets;
    // thrust::device_vector<int> vertex_color_claimer;
    // thrust::device_vector<int> uncolored_count; // one size
    // thrust::device_vector<int> constraint_colors;
    // thrust::device_vector<uint64_t> vertex_forbidden_masks;
    // cudaGraph_t current_graph = nullptr;
    // cudaGraphExec_t current_graph_exec = nullptr;
    // cudaStream_t capture_stream = nullptr;
    //
    // thrust::device_vector<float3> points_safe;
    // thrust::device_vector<float> weight;
    //
    // int max_pp_result_size;
    // int max_tp_result_size;
    // int max_ee_result_size;
    // int max_point_hashes_size;
    // int max_edge_hashes_size;
    // int max_sort_result_size;
    // int max_collision_constraints_size;


    float alpha_hard;
    float point_radius = 0.00025f;
    float max_dist = 0.0005f;
    lbvh3d::BVH3D tri_bvh;
    lbvh3d::BVH3D edge_bvh;

    thrust::device_vector<unsigned int> point_sorted_indices;
    thrust::device_vector<unsigned int> edge_sorted_indices;
    
    thrust::device_vector<int> broad_phase_ee;
    thrust::device_vector<int> broad_phase_ef;
    thrust::device_vector<int> broad_phase_vf;
};
