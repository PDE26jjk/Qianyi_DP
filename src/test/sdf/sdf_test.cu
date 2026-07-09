#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <vector>
#include <random>
#include <string>
#include <iostream>
#include "../benchmarks.h"
#include "simulation/sdf/sdf.cuh"

#include "simulation/sdf/solid_angle.cuh"

std::vector<float> sdf_benchmark(
    const std::vector<float>& vertices_in,
    const std::vector<int>& faces_in,
    bool use_parity,
    int warmup, int runs
) {
    auto& timer = globalTimer();
    thrust::device_vector<float3> d_vertices(vertices_in.size() / 3);
    cudaMemcpy(d_vertices.data().get(), vertices_in.data(), d_vertices.size() * sizeof(float3), cudaMemcpyHostToDevice);
    thrust::device_vector<int3> d_faces(faces_in.size() / 3);
    cudaMemcpy(d_faces.data().get(), faces_in.data(), d_faces.size() * sizeof(int3), cudaMemcpyHostToDevice);
    int num_verts = vertices_in.size() / 3;
    int num_faces = faces_in.size() / 3;
    lbvh3d::initialize(num_faces);

    lbvh3d::BVH3D bvh;
    thrust::device_vector<lbvh3d::SolidAngleProps> solid_angle_props;
    if ( !use_parity ) {
        lbvh3d::build_face_bvh_with_solid_angle(d_vertices, d_faces, bvh, solid_angle_props);
    }
    else {
        lbvh3d::build_face_bvh(d_vertices, d_faces, bvh);
    }
    cudaDeviceSynchronize();

    sdf::SDF sdf;
    sdf.build_from_mesh(d_faces.data().get(), d_vertices.data().get(), bvh, num_verts, 64, 0.f, 8,
        use_parity, use_parity ? nullptr : solid_angle_props.data().get());
    auto mesh = sdf.compute_isomesh_from_texture_sdf(0.f);
    auto result = std::vector<float>();
    result.resize(mesh.positions.size() * 3);
    cudaMemcpy(result.data(), mesh.positions.data().get(), result.size() * sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}
std::vector<float> sdf_check_inside(
    const std::vector<float>& vertices_in,
    const std::vector<int>& faces_in,
    const std::vector<float>& query_points,
    int warmup, int runs
) {
    auto& timer = globalTimer();

    int num_verts = vertices_in.size() / 3;
    int num_faces = faces_in.size() / 3;
    thrust::device_vector<float3> d_vertices(num_verts);
    thrust::device_vector<int3> d_faces(num_faces);
    cudaMemcpy(thrust::raw_pointer_cast(d_vertices.data()), vertices_in.data(),
        num_verts * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(thrust::raw_pointer_cast(d_faces.data()), faces_in.data(),
        num_faces * sizeof(int3), cudaMemcpyHostToDevice);

    int num_queries = query_points.size() / 3;
    thrust::device_vector<float3> d_points(num_queries);
    cudaMemcpy(thrust::raw_pointer_cast(d_points.data()), query_points.data(),
        num_queries * sizeof(float3), cudaMemcpyHostToDevice);
    lbvh3d::initialize(num_faces);

    lbvh3d::BVH3D bvh;
    lbvh3d::build_face_bvh(d_vertices, d_faces, bvh);
    cudaDeviceSynchronize();

    thrust::device_vector<float> d_res(num_queries);
    sdf::check_inside(d_faces.data().get(), d_vertices.data().get(), bvh, d_points.data().get(), num_queries,
        d_res.data().get());
    auto result = std::vector<float>();
    result.resize(num_queries);
    cudaMemcpy(result.data(), d_res.data().get(), result.size() * sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}
