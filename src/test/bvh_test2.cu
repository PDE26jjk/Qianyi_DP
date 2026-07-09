#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <random>
#include <iostream>
#include <vector>
#include "bvh/bvh_query.cuh"
#include "benchmarks.h"

std::vector<int> bvh2_benchmark(
    const std::vector<float>& vertices_in,
    const std::vector<int>& faces_in,
    const std::vector<float>& query_points,
    int warmup, int runs, bool verify) {
    auto& timer = globalTimer();

    // Upload data
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

    lbvh3dtest::BVH bvh;  // raw BVH object

    // Build (warmup)
    for ( int i = 0; i < warmup; ++i ) {
        lbvh3dtest::bvh_destroy(0, bvh);
        lbvh3dtest::build_face_bvh_test(d_vertices, d_faces, bvh);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Benchmark builds
    for ( int i = 0; i < runs; ++i ) {
        lbvh3dtest::bvh_destroy(0, bvh);
        timer.start("build_bvh");
        lbvh3dtest::build_face_bvh_test(d_vertices, d_faces, bvh);
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();
    }

    int root_idx = lbvh3dtest::get_root_index(bvh);

    thrust::device_vector<int> nearest(num_queries);
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_queries + threadsPerBlock - 1) / threadsPerBlock;


    // Warmup queries
    for ( int i = 0; i < warmup; ++i ) {
        cudaMemset(thrust::raw_pointer_cast(nearest.data()), -1, num_queries * sizeof(int));
        lbvh3dtest::query_nearest_face_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(d_points.data()),
            num_queries,
            bvh.node_lowers,
            bvh.node_uppers,
            bvh.primitive_indices,
            root_idx,
            thrust::raw_pointer_cast(d_vertices.data()),
            thrust::raw_pointer_cast(d_faces.data()),
            thrust::raw_pointer_cast(nearest.data()));
        CUDA_CHECK(cudaDeviceSynchronize());
    }


    // Benchmark queries
    for ( int i = 0; i < runs; ++i ) {
        cudaMemset(thrust::raw_pointer_cast(nearest.data()), -1, num_queries * sizeof(int));
        timer.start("bvh_query");
        lbvh3dtest::query_nearest_face_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(d_points.data()),
            num_queries,
            bvh.node_lowers,
            bvh.node_uppers,
            bvh.primitive_indices,
            root_idx,
            thrust::raw_pointer_cast(d_vertices.data()),
            thrust::raw_pointer_cast(d_faces.data()),
            thrust::raw_pointer_cast(nearest.data()));
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();
    }

    std::vector<int> result;
    if ( verify ) {
        result.resize(num_queries);
        cudaMemcpy(result.data(), thrust::raw_pointer_cast(nearest.data()),
            num_queries * sizeof(int), cudaMemcpyDeviceToHost);
    }
    return result;
}
