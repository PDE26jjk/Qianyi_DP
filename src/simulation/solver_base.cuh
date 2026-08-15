#pragma once
#include <mutex>

#include "simulator.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
// #include <thrust/universal_vector.h>

//

// struct PatternData {
//     float granularity;
//     float thickness;
//     float friction;
// };


struct AutoGPUmem {
    SolverBase* pool;
    int* ptr;
    ~AutoGPUmem();
};

struct SolverBase {
    SolverBase(const SolverBase& other) = delete;
    SolverBase(SolverBase&& other) noexcept = delete;
    SolverBase& operator=(const SolverBase& other) = delete;
    SolverBase& operator=(SolverBase&& other) noexcept = delete;
    virtual ~SolverBase() = default;

    explicit SolverBase(Simulator* simulator) noexcept: simulator(simulator) {}
    Simulator* simulator;

public:
    // thrust::device_vector<float3> vertices_2D;
    // thrust::device_vector<float3> vertices_local;
    // thrust::device_vector<float3> vertices_local_new_frame;
    // thrust::device_vector<float3> normals_input;
    // thrust::device_vector<int2> edges;
    // thrust::device_vector<int3> triangles;
    // thrust::device_vector<int> object_types;
    // thrust::device_vector<int> vertex_index_offsets;
    // thrust::device_vector<int> triangle_index_offsets;
    // thrust::device_vector<float> pin_fixed;
    // thrust::device_vector<float> pin_attached;
    // thrust::device_vector<AttachInfo> attach_data;
    // thrust::device_vector<int> vertices_obj;
    // thrust::device_vector<ObjectDataInput> obj_data;
    // thrust::device_vector<float> mass_densitys; // mass density per object, kg/m^2
    // thrust::device_vector<float> masses; 
    // thrust::device_vector<float> mass_inv;
    // thrust::device_vector<Mat4> world_matrices;
    // thrust::device_vector<Mat4> world_matrices_inv;
    // thrust::device_vector<int2> dir_edges;
    // thrust::device_vector<int2> edge_lookup;
    // thrust::device_vector<int3> triangle_indices;
    // thrust::device_vector<int2> e2t;
    // thrust::device_vector<int2> edge_opposite_points;
    // thrust::device_vector<Mat2> Dms;
    // thrust::device_vector<float3> vertices_old;
    // thrust::device_vector<float3> vertices_world;
    // thrust::device_vector<float3> vertices_new;

    // thrust::device_vector<float3> forces;
    // thrust::device_vector<float> edge_lengths;
    // thrust::device_vector<float3> velocities;
    // thrust::device_vector<char> vertices_mask;

    // sewing
    // thrust::device_vector<SewingData> sewing_lines;
    // thrust::device_vector<int2> stitches;
    // thrust::device_vector<int> stitch_sewing; // stitch to sewing
    // thrust::device_vector<int> vertex_proxy; // if proxy is valid, copy position from proxy on the end of updating.
    // thrust::device_vector<int2> sewing_edges;
    // thrust::device_vector<int2> sewing_e2t;
    // thrust::device_vector<int2> sewing_edge_opposite_points;
    // thrust::device_vector<int> stitches_done_count;
    // thrust::device_vector<char> stitches_status;
    // bool sewing_done;

    // thrust::device_vector<float3> temp_vertices2_f3;
    // thrust::device_vector<float3> temp_vertices3_f3;
    // thrust::device_vector<float3> temp_edge_f3;

    // bending
    // thrust::device_vector<float4> IBM_q;

    virtual void init();
    virtual void begin_frame() {}
    virtual void step(float h) = 0;


    // void collision_LCP_postprocess(float3* points_y);
    // void collision_LCP_postprocess_unified(float3* points_y);
    // void collision_collect_near_pairs(float3* points, float max_dist, bool update_hash,
    //     bool collect_pp = false, bool collect_tp = false, bool collect_ee = false);
    // int color_constraints(int);
    // float3* collision_Wu2021_step(float3* y, bool first, bool& done);

    float get_global_parameter(const std::string& key, float default_value) const;

private:
    thrust::device_vector<int> pool;
    std::vector<bool> pool_used;
    friend AutoGPUmem;
protected:
    AutoGPUmem alloc_pool();
    void dealloc_pool(void* p);
};
