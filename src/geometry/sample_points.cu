#include <cuda_runtime.h>

#include "common/cuda_utils.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <thrust/host_vector.h>

#include "lbvh_2d.h"
#include "sampler.h"

#include "common/atomic_utils.cuh"
#include "common/device.h"
#include "graphics/graphics.h"
std::vector<int3> delaunay_2d_cuda_type_impl(std::vector<float2>& pointVecIn,
    std::vector<int2>& constraintVecIn);


static __device__ __forceinline__ float2 normalize_safe(const float2& a) {
    float l = sqrtf(dot(a, a));
    if ( l < 1e-8f ) return make_float2(0, 0);
    return a * (1.0f / l);
}


static __device__ int2 grid_index(float2 p, float one_grid_length) {
    return make_int2((int)(p.x / one_grid_length + 0.5f), (int)(p.y / one_grid_length + 0.5f));
}
static __device__ bool try_test(int2 index, unsigned char* grid_status, int max_size) {
    int idx = index.y * max_size + index.x;
    unsigned char old = atomicOr(&grid_status[idx], (unsigned char)(1 << 1));
    return old == unsigned char(0);
}

// ==========================================
// Kernels
// ==========================================

static __global__ void k_init_rng(curandState* states, long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < n ) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

static __global__ void k_fill_boundary(
    const float2* points,
    const int* next_point,
    int num_input_points,
    float2* final_pts,
    unsigned char* grid_status,
    int* grid_point,
    int* d_nb_points,
    Params p
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_input_points ) return;

    float2 point = points[i];
    int2 index = grid_index(point, p.one_grid_length);

    if ( index.x < 0 || index.x >= p.max_size || index.y < 0 || index.y >= p.max_size ) return;

    int idx_flat = index.y * p.max_size + index.x;

    if ( grid_status[idx_flat] == 0 ) {
        if ( try_test(index, grid_status, p.max_size) ) {
            int j = atomicAdd(d_nb_points, 1);
            final_pts[j] = point;
            grid_point[idx_flat] = i;
        }
    }
    float2 next_p = points[next_point[i]];
    float2 mid_point = (point + next_p) * 0.5f;
    int2 mid_index = grid_index(mid_point, p.one_grid_length);

    if ( mid_index.x >= 0 && mid_index.x < p.max_size && mid_index.y >= 0 && mid_index.y < p.max_size ) {
        int mid_idx_flat = mid_index.y * p.max_size + mid_index.x;
        if ( grid_status[mid_idx_flat] == 0 ) {
            if ( try_test(mid_index, grid_status, p.max_size) ) {
                int k = atomicAdd(d_nb_points, 1);
                final_pts[k] = mid_point;
                grid_point[mid_idx_flat] = i;
            }
        }
    }

}

static __global__ void k_scan_line_fill(
    float2* final_pts,
    const float2* points,
    const int* next_point,
    int* grid_point,
    unsigned char* grid_status,
    int nb_current_points,
    Params p
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= nb_current_points ) return;

    float2 point = final_pts[i];
    int2 index = grid_index(point, p.one_grid_length);

    if ( index.x < 0 || index.x >= p.max_size || index.y < 0 || index.y >= p.max_size ) return;

    int idx_flat = index.y * p.max_size + index.x;
    int grid_pt_idx = grid_point[idx_flat];

    float2 next_p = points[next_point[grid_pt_idx]];

    int d = (next_p.x - point.x > 0) ? 1 : -1;

    int x = index.x;
    int y = index.y;

    while ( true ) {
        y = y + d;
        if ( y < 0 || y >= p.max_size ) break;

        int curr_idx = y * p.max_size + x;
        if ( grid_status[curr_idx] != 0 ) {
            break;
        }
        else {
            grid_status[curr_idx] = 4;
        }
    }
}
static __global__ void k_copy_fill(
    unsigned char* grid_status,
    float* rasterization_result,
    Params p
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ( x >= p.grid_size || y >= p.grid_size ) return;

    int idx_flat = y * p.max_size + x;
    if ( grid_status[idx_flat] == 0 ) {
        if ( rasterization_result[idx_flat] < 0.5f ) return;
        bool edge_point = false;
        bool ex_point = false;
        if ( x > 0 ) {
            if ( grid_status[idx_flat - 1] == 2 ) {
                edge_point = true;
            }
            else {
                ex_point |= (rasterization_result[idx_flat - 1] < 0.5f);
            }
        }
        if ( x < p.grid_size - 1 ) {
            if ( grid_status[idx_flat + 1] == 2 ) {
                edge_point = true;
            }
            else {
                ex_point |= (rasterization_result[idx_flat + 1] < 0.5f);
            }
        }
        if ( y > 0 ) {
            if ( grid_status[idx_flat - p.max_size] == 2 ) {
                edge_point = true;
            }
            else {
                ex_point |= (rasterization_result[idx_flat - p.max_size] < 0.5f);
            }
        }
        if ( y < p.grid_size - 1 ) {
            if ( grid_status[idx_flat + p.max_size] == 2 ) {
                edge_point = true;
            }
            else {
                ex_point |= (rasterization_result[idx_flat + p.max_size] < 0.5f);
            }
        }
        if ( edge_point && ex_point ) return;
        grid_status[idx_flat] = 4;
    }
}

static __global__ void k_generate_interior(
    unsigned char* grid_status,
    float2* final_pts,
    int* d_nb_points,
    curandState* states,
    Params p
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ( x >= p.grid_size || y >= p.grid_size ) return;

    int idx = y * p.max_size + x;

    if ( grid_status[idx] == 4 ) {
        int tid = y * p.grid_size + x;
        curandState localState = states[tid];
        float r1 = curand_uniform(&localState) - 0.5f;
        float r2 = curand_uniform(&localState) - 0.5f;
        states[tid] = localState;

        float2 point = make_float2(r1, r2);
        float2 local_pos = (point + make_float2((float)x, (float)y)) * p.one_grid_length;

        int j = atomicAdd(d_nb_points, 1);
        final_pts[j] = local_pos;
    }
}

static __global__ void k_reset_grid_multi(unsigned char* grid_multi_point_size, Params p) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ( x < p.max_size && y < p.max_size ) {
        grid_multi_point_size[y * p.max_size + x] = 0;
    }
}
constexpr int max_grid_particles_size = 4;

static __global__ void k_build_grid(
    float2* final_pts,
    unsigned char* grid_multi_point_size,
    int* grid_multi_point,
    int nb_points,
    Params p
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= nb_points ) return;

    float2 point = final_pts[i];
    int2 index = grid_index(point, p.one_grid_length);
    if ( index.x < 0 || index.x >= p.max_size || index.y < 0 || index.y >= p.max_size ) return;

    int idx_flat = index.y * p.max_size + index.x;

    unsigned char id = atomicAdd(&grid_multi_point_size[idx_flat], 1);
    if ( id < max_grid_particles_size ) {
        int offset = idx_flat * 4 + id;
        grid_multi_point[offset] = i;
    }
}

static __global__ void k_compute_repulsion(
    float2* final_pts,
    unsigned char* grid_multi_point_size,
    int* grid_multi_point,
    float2* force,
    int nb_boundary_points,
    int nb_points,
    Params p
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int actual_i = i + nb_boundary_points;
    if ( actual_i >= nb_points ) return;

    float2 point = final_pts[actual_i];
    int2 index = grid_index(point, p.one_grid_length);
    float r2 = p.radius * p.radius;
    float2 d = make_float2(0.0f, 0.0f);

    int x = index.x;
    int y = index.y;

    for ( int nx = max(0, x - 1); nx < min(p.max_size, x + 2); ++nx ) {
        for ( int ny = max(0, y - 1); ny < min(p.max_size, y + 2); ++ny ) {
            int n_idx = ny * p.max_size + nx;
            int count = min((int)grid_multi_point_size[n_idx], 4);

            for ( int k = 0; k < count; ++k ) {
                int p2_i = grid_multi_point[n_idx * 4 + k];
                if ( p2_i == actual_i ) continue;

                float2 p2 = final_pts[p2_i];
                float2 l = point - p2;
                float l2 = dot(l, l);

                if ( l2 < (r2 * 0.5f) + 1e-6f ) {
                    if ( l2 > 1e-10f ) {
                        // Taichi: d += l.normalized() / l2
                        d = d + (l * (1.0f / (sqrtf(l2) * l2)));
                    }
                }
            }
        }
    }
    force[actual_i] = d;
}

static __global__ void k_apply_force(
    float2* final_pts,
    float2* force,
    int nb_boundary_points,
    int nb_points,
    float factor,
    Params p
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int actual_i = i + nb_boundary_points;
    if ( actual_i >= nb_points ) return;

    float2 d = force[actual_i];
    float maxd = p.radius * factor;
    float d_len_sq = dot(d, d);

    if ( d_len_sq > maxd * maxd ) {
        d = normalize_safe(d) * maxd;
    }
    final_pts[actual_i] = final_pts[actual_i] + d;
}

static __global__ void k_validate(
    float2* final_pts,
    unsigned char* grid_status,
    unsigned char* valid_status,
    int nb_boundary_points,
    int nb_points,
    Params p
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int actual_i = i + nb_boundary_points;
    if ( actual_i >= nb_points ) return;

    int2 index = grid_index(final_pts[actual_i], p.one_grid_length);
    if ( index.x >= 0 && index.x < p.max_size && index.y >= 0 && index.y < p.max_size ) {
        if ( grid_status[index.y * p.max_size + index.x] == 4 ) {
            valid_status[actual_i] = 1;
        }
        else {
            valid_status[actual_i] = 0;
        }
    }
    else {
        valid_status[actual_i] = 0;
    }
}

static __global__ void k_validate_triangles(
    unsigned char* valid_status,
    const int3* triangles,
    const unsigned char* grid_status,
    const float2* pts,
    const int* next_point,
    float r,
    int nb_boundary_points,
    int nb_tris,
    Params p
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= nb_tris ) return;
    int vs[3] = { triangles[i].x, triangles[i].y, triangles[i].z };
    // check area
    auto p0 = pts[vs[0]], p1 = pts[vs[1]], p2 = pts[vs[2]];
    float cross_value = cross(p1 - p0, p2 - p0);
    if ( abs(cross_value) < 1e-8f ) {
        valid_status[i] = 0;
        return;
    }
    if ( vs[0] >= nb_boundary_points || vs[1] >= nb_boundary_points
        || vs[2] >= nb_boundary_points ) {
        valid_status[i] = 1;
        return;
    }
    // printf("%f\n",cross_value);
    // printf("%d\n",v0 >= nb_boundary_points || v1 >= nb_boundary_points
    // || v2 >= nb_boundary_points);
    // printf("v0: %d, v1: %d, v2: %d\n", v0, v1, v2);
    // printf("check center\n");
    // check center
    auto center = (p0 + p1 + p2) / 3.0f;
    int2 index = grid_index(center, p.one_grid_length);
    int idx_flat = index.y * p.max_size + index.x;
    unsigned char status = grid_status[idx_flat];
    if ( status == 4 ) {
        valid_status[i] = 1;
        return;
    }
    else if ( status == 0 ) {
        valid_status[i] = 0;
        return;
    }
    float2 e1 = p1 - p0, e2 = p2 - p0, e3 = p2 - p1;
    const float max_l_sq = r * r * 9;
    if ( len_sq(e1) > max_l_sq || len_sq(e2) > max_l_sq || len_sq(e3) > max_l_sq ) {
        valid_status[i] = 0;
        return;
    }
    int np[3];
    #pragma unroll
    for ( int k = 0; k < 3; ++k )
        np[k] = next_point[vs[k]];

    int start = -1, end = -1;
    for ( int j = 0; j < 3; ++j )
        for ( int k = 0; k < 3; ++k ) {
            if ( np[k] == vs[j] ) {
                start = vs[k];
                end = vs[j];
                break;
            }
        }
    if ( start != -1 ) {
        p0 = pts[start];
        p1 = pts[end];
        if ( cross(center - p0, p1 - p0) > 0 ) {
            valid_status[i] = 0;
            return;
        }
    }

    valid_status[i] = 1;
}


// ==========================================
// Host Controller Class
// ==========================================



Sampler::Sampler() {
    cudaMalloc(&d_nb_points, sizeof(int));
}


Sampler::~Sampler() {
    if ( cudaFree(nullptr) == cudaErrorCudartUnloading ) return; // TODO check it.
    cudaFree(d_grid_status);
    cudaFree(d_grid_point);
    cudaFree(d_final);
    cudaFree(d_grid_multi_point);
    cudaFree(d_grid_multi_point_size);
    cudaFree(d_force);
    cudaFree(d_valid_status);
    cudaFree(d_nb_points);
    cudaFree(d_rng_states);
    if ( d_input_points ) cudaFree(d_input_points);
    if ( d_input_next ) cudaFree(d_input_next);
}

void Sampler::set_radius(float _radius) {
    params.radius = _radius;
    params.one_grid_length = _radius / sqrtf(2.0f);
    params.grid_size = (int)ceil(1.0f / params.one_grid_length);
    params.max_size = ((params.grid_size + 63) / 64) * 64;
    params.n = params.max_size * params.max_size; // Note: Taichi n = max_size^2, usually plenty buffer

    // Re-allocate if size changed (simplified: always re-allocate for this example)
    if ( d_grid_status ) cudaFree(d_grid_status);
    if ( d_grid_point ) cudaFree(d_grid_point);
    if ( d_final ) cudaFree(d_final);
    if ( d_grid_multi_point ) cudaFree(d_grid_multi_point);
    if ( d_grid_multi_point_size ) cudaFree(d_grid_multi_point_size);
    if ( d_force ) cudaFree(d_force);
    if ( d_valid_status ) cudaFree(d_valid_status);
    if ( d_rng_states ) cudaFree(d_rng_states);

    cudaMalloc(&d_grid_status, params.max_size * params.max_size * sizeof(unsigned char));
    cudaMalloc(&d_grid_point, params.max_size * params.max_size * sizeof(int));
    cudaMalloc(&d_final, params.n * sizeof(float2));
    cudaMalloc(&d_grid_multi_point, params.max_size * params.max_size * 4 * sizeof(int));
    cudaMalloc(&d_grid_multi_point_size, params.max_size * params.max_size * sizeof(unsigned char));
    cudaMalloc(&d_force, params.n * sizeof(float2));
    cudaMalloc(&d_valid_status, params.n * sizeof(unsigned char));

    // Init RNG
    cudaMalloc(&d_rng_states, params.n * sizeof(curandState));
    int blockSize = 256;
    int numBlocks = (params.n + blockSize - 1) / blockSize;
    k_init_rng <<<numBlocks, blockSize>>>(d_rng_states, 1234, params.n);
    current_radius_scaled = _radius;
}

void Sampler::sample(
    std::vector<float2>& output_points,
    std::vector<int3>& output_tris,
    const std::vector<float2>& boundary_points,
    const std::vector<int>& next_point,
    float raw_radius,
    float f1, int t1, float f2, int t2
) {
    if ( boundary_points.empty() ) return;

    float x_min = FLT_MAX, y_min = FLT_MAX;
    float x_max = -FLT_MAX, y_max = -FLT_MAX;

    for ( const auto& p : boundary_points ) {
        if ( p.x < x_min ) x_min = p.x;
        if ( p.y < y_min ) y_min = p.y;
        if ( p.x > x_max ) x_max = p.x;
        if ( p.y > y_max ) y_max = p.y;
    }

    float width = x_max - x_min;
    float height = y_max - y_min;

    // Python: scale = 1.0 / (max(width, height) + radius)
    float scale = 1.0f / (fmaxf(width, height) + raw_radius);
    float2 offset = make_float2(x_min, y_min);

    // Python: radius_scaled = radius * scale
    float radius_scaled = raw_radius * scale;


    set_radius(radius_scaled);

    // Python: points_normalized = (boundary_points - offset + radius * 0.5) * scale
    std::vector<float2> points_normalized(boundary_points.size());
    for ( size_t i = 0; i < boundary_points.size(); ++i ) {
        float2 p = boundary_points[i];
        points_normalized[i].x = (p.x - offset.x + raw_radius * 0.5f) * scale;
        points_normalized[i].y = (p.y - offset.y + raw_radius * 0.5f) * scale;
    }
    std::vector<graphics::Edge> edges(points_normalized.size());
    float factor = params.grid_size;
    for ( int i = 0; i < edges.size(); ++i ) {
        int j = next_point[i];
        edges[i] = { { points_normalized[i].x * factor, points_normalized[i].y * factor },
            { points_normalized[j].x * factor, points_normalized[j].y * factor } };
    }

    auto& rasterizer = graphics::VulkanCudaRasterizer::Instance();
    float* d_rasterization_result = rasterizer.render(edges, params.max_size, params.max_size, 0);

    // ---------------------------------------------------------
    // 2. GPU Processing
    // ---------------------------------------------------------

    int num_pts = points_normalized.size();
    // 1. Setup Input
    if ( d_input_points ) cudaFree(d_input_points);
    if ( d_input_next ) cudaFree(d_input_next);
    cudaMalloc(&d_input_points, num_pts * sizeof(float2));
    cudaMalloc(&d_input_next, num_pts * sizeof(int));

    cudaMemcpy(d_input_points, points_normalized.data(), num_pts * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_next, next_point.data(), num_pts * sizeof(int), cudaMemcpyHostToDevice);

    // 2. Clear Grid Status & Counters
    cudaMemset(d_grid_status, 0, params.max_size * params.max_size * sizeof(unsigned char));
    cudaMemset(d_nb_points, 0, sizeof(int));

    // 3. Fill Boundary (Kernel 1)
    int blockSize = 256;
    int numBlocks = (num_pts + blockSize - 1) / blockSize;
    k_fill_boundary<<<numBlocks, blockSize>>>(
        d_input_points, d_input_next, num_pts,
        d_final, d_grid_status, d_grid_point, d_nb_points, params
        );
    // cudaDeviceSynchronize();

    // Get nb_points
    int h_nb_points = 0;
    cudaMemcpy(&h_nb_points, d_nb_points, sizeof(int), cudaMemcpyDeviceToHost);

    // 4. Scan Line Fill (Kernel 2)
    // numBlocks = (h_nb_points + blockSize - 1) / blockSize;
    // k_scan_line_fill<<<numBlocks, blockSize>>>(
    //     d_final, d_input_points, d_input_next,
    //     d_grid_point, d_grid_status, h_nb_points, params
    //     );
    // cudaDeviceSynchronize();

    int nb_boundary_points = h_nb_points; //

    // 5. Generate Interior (Kernel 3)
    dim3 dimBlock(16, 16);
    dim3 dimGrid((params.grid_size + dimBlock.x - 1) / dimBlock.x,
        (params.grid_size + dimBlock.y - 1) / dimBlock.y);
    k_copy_fill<<<dimGrid, dimBlock >>>(d_grid_status, d_rasterization_result, params);

    k_generate_interior<<<dimGrid, dimBlock>>>(
        d_grid_status, d_final, d_nb_points, d_rng_states, params
        );
    // cudaDeviceSynchronize();

    // Get updated nb_points
    cudaMemcpy(&h_nb_points, d_nb_points, sizeof(int), cudaMemcpyDeviceToHost);

    // 6. Repulsion Loops
    auto run_repulsion = [&](float factor, int times) {
        for ( int t = 0; t < times; ++t ) {
            // Reset Grid Multi
            dim3 gridDimMax((params.max_size + 15) / 16, (params.max_size + 15) / 16);
            k_reset_grid_multi<<<gridDimMax, dimBlock >>>(d_grid_multi_point_size, params);

            // Build Grid
            int blocks = (h_nb_points + 255) / 256;
            k_build_grid<<<blocks, 256>>>(
                d_final, d_grid_multi_point_size, d_grid_multi_point, h_nb_points, params
                );

            // Compute Force (Only for non-boundary points)
            int inner_count = h_nb_points - nb_boundary_points;
            if ( inner_count > 0 ) {
                blocks = (inner_count + 255) / 256;
                k_compute_repulsion<<<blocks, 256 >>>(
                    d_final, d_grid_multi_point_size, d_grid_multi_point, d_force,
                    nb_boundary_points, h_nb_points, params
                    );

                // Apply Force
                k_apply_force<<<blocks, 256 >>>(
                    d_final, d_force, nb_boundary_points, h_nb_points, factor, params
                    );
            }
            // cudaDeviceSynchronize();
        }
    };

    run_repulsion(f1, t1);
    run_repulsion(f2, t2);

    // 7. Validation
    int inner_count = h_nb_points - nb_boundary_points;
    if ( inner_count > 0 ) {
        int blocks = (inner_count + 255) / 256;
        k_validate<<<blocks, 256>>>(
            d_final, d_grid_status, d_valid_status, nb_boundary_points, h_nb_points, params
            );
    }
    cudaDeviceSynchronize();

    // 8. Retrieve Data
    std::vector<float2> result_final(h_nb_points);
    std::vector<unsigned char> result_valid(h_nb_points);
    cudaMemcpy(result_final.data(), d_final, h_nb_points * sizeof(float2), cudaMemcpyDeviceToHost);
    cudaMemcpy(result_valid.data(), d_valid_status, h_nb_points * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Filter valid points (Taichi logic: only return valid points generated inside, excluding boundary?)
    // The python code: return final[nb_boundary:][valid > 0]

    // 9. triangulate
    output_points.reserve(points_normalized.size() + inner_count);
    output_points.resize(points_normalized.size());
    memcpy(output_points.data(), points_normalized.data(), points_normalized.size() * sizeof(float2));

    for ( int i = nb_boundary_points; i < h_nb_points; ++i ) {
        if ( result_valid[i] > 0 ) {
            output_points.push_back(result_final[i]);
        }
    }
    std::vector<int2> constraints(next_point.size());
    for ( int i = 0; i < next_point.size(); ++i ) {
        constraints[i] = { i, next_point[i] };
    }
    // return output;
    auto tris = delaunay_2d_cuda_type_impl(output_points, constraints);
    thrust::device_vector<int3> d_tris(tris.begin(), tris.end());
    thrust::device_vector<unsigned char> d_valid_status_tris(tris.size());
    thrust::device_vector<float2> d_pts(output_points.begin(), output_points.end());
    int block = 256;
    k_validate_triangles<<<(tris.size() + block - 1) / block,block>>>
        (d_valid_status_tris.data().get(), d_tris.data().get(),
        d_grid_status, d_pts.data().get(), d_input_next, radius_scaled,
        points_normalized.size(), tris.size(), params
        );
    for ( auto& output_point : output_points ) {
        output_point.x = output_point.x / scale - raw_radius * 0.5f + offset.x;
        output_point.y = output_point.y / scale - raw_radius * 0.5f + offset.y;
    }
    thrust::host_vector<unsigned char> h_valid_status_tris(d_valid_status_tris);
    output_tris.reserve(tris.size());
    for ( int i = 0; i < tris.size(); ++i ) {
        if ( h_valid_status_tris[i] == 1 ) {
            output_tris.push_back(tris[i]);
        }
    }

}


// ==========================================
// Main / Interface
// ==========================================

void sample_points_impl(std::vector<float>& boundary, std::vector<int>& next_pt, float radius, std::vector<float>& output_points,
    std::vector<int>& output_tris) {
    //float radius = 0.05f;
    init_device();
    Sampler sampler{};

    //std::vector<float2> boundary = {
    //    {0.1f, 0.1f}, {0.9f, 0.1f}, {0.9f, 0.9f}, {0.1f, 0.9f}
    //};
    //std::vector<int> next_pt = { 1, 2, 3, 0 };
    std::vector<float2> boundary_(boundary.size() / 2);
    std::memcpy(boundary_.data(), boundary.data(), sizeof(float) * boundary.size());

    std::vector<float2> points;
    std::vector<int3> tris;
    sampler.sample(points, tris, boundary_, next_pt, radius, 0.02f, 15, 0.01f, 35);
    //std::cout << "Generated " << points.size() << " points." << std::endl;
    //for (size_t i = 0; i < std::min(points.size(), (size_t)10); i++) {
    //    std::cout << points[i].x << ", " << points[i].y << std::endl;
    //}
    int size = sampler.params.max_size * sampler.params.max_size;
    // std::vector<unsigned char> res(size);
    // cudaMemcpy(res.data(), sampler.d_grid_status, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    output_points.resize(points.size() * 2);
    std::memcpy(output_points.data(), points.data(), sizeof(float) * output_points.size());
    output_tris.resize(tris.size() * 3);
    std::memcpy(output_tris.data(), tris.data(), sizeof(int) * output_tris.size());
}

std::vector<int> sample_points_debug2(std::vector<float>& boundary, std::vector<float>& points) {
    START_TIMER;

    std::vector<float2> boundary_(boundary.size() / 2);
    std::memcpy(boundary_.data(), boundary.data(), sizeof(float) * boundary.size());
    thrust::device_vector<float2> vertices(boundary_.size());
    thrust::copy(boundary_.begin(), boundary_.end(), vertices.begin());

    std::vector<float2> points_(points.size() / 2);
    std::memcpy(points_.data(), points.data(), sizeof(float) * points.size());
    thrust::device_vector<float2> query_pts(points.size());
    thrust::copy(points_.begin(), points_.end(), query_pts.begin());
    RECORD_TIME("copy data");

    BVH2D bvh;
    lbvh2d::initialize(vertices.size());
    lbvh2d::build_point_bvh(vertices, bvh);
    CUDA_CHECK(cudaDeviceSynchronize());
    RECORD_TIME("build bvh");

    thrust::device_vector<lbvh2d::NearestResult> results(query_pts.size());
    int block = 256;
    int n = query_pts.size();
    lbvh2d::query_nearest_kernel<<<(n + block - 1) / block, block>>>(
        query_pts.data().get(), n,
        bvh.nodes.data().get(), bvh.aabbs.data().get(),
        bvh.root_idx, results.data().get(),
        vertices.data().get());
    CUDA_CHECK(cudaDeviceSynchronize());
    RECORD_TIME("query_nearest_kernel");

    std::vector<lbvh2d::NearestResult> results_h;
    results_h.resize(results.size());
    thrust::copy(results.begin(), results.end(), results_h.begin());
    std::vector<int> res(n);
    for ( int i = 0; i < results_h.size(); ++i ) {
        res[i] = results_h[i].prim_idx;
    }
    return res;
}
std::vector<unsigned char> sample_points_debug(std::vector<float>& boundary, std::vector<int>& next_pt, float radius) {
    //float radius = 0.05f;
    START_TIMER;
    Sampler sampler{};

    //std::vector<float2> boundary = {
    //    {0.1f, 0.1f}, {0.9f, 0.1f}, {0.9f, 0.9f}, {0.1f, 0.9f}
    //};
    //std::vector<int> next_pt = { 1, 2, 3, 0 };
    std::vector<float2> boundary_(boundary.size() / 2);
    std::memcpy(boundary_.data(), boundary.data(), sizeof(float) * boundary.size());

    std::vector<float2> points;
    std::vector<int3> tris;
    sampler.sample(points, tris, boundary_, next_pt, radius, 0.02f, 15, 0.01f, 35);

    int size = sampler.params.max_size * sampler.params.max_size;
    RECORD_TIME("sampler.sample");

    std::vector<unsigned char> res(size);
    cudaMemcpy(res.data(), sampler.d_grid_status, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    /*thrust::device_vector<float2> vertices(boundary_.size());
    thrust::copy(boundary_.begin(), boundary_.end(), vertices.begin());
    thrust::device_vector<float2> query_pts(points.size());
    thrust::copy(points.begin(), points.end(), query_pts.begin());
    BVH2D bvh;
    lbvh2d::initialize(vertices.size());
    lbvh2d::build_point_bvh(vertices, bvh);
    CUDA_CHECK(cudaDeviceSynchronize());
    RECORD_TIME("build bvh");

    thrust::device_vector<NearestResult> results(query_pts.size());
    int block = 256;
    int n = query_pts.size();
    lbvh2d::query_nearest_kernel<<<(n + block - 1) / block, block>>>(
        query_pts.data().get(), n,
        bvh.nodes.data().get(), bvh.aabbs.data().get(),
        bvh.root_idx, results.data().get(),
        vertices.data().get());
    CUDA_CHECK(cudaDeviceSynchronize());
    RECORD_TIME("build bvh");

    std::vector<NearestResult> results_h;
    results_h.resize(results.size());
    thrust::copy(results.begin(), results.end(), results_h.begin());
    std::vector<int> res(n);
    for ( int i = 0; i < results_h.size(); ++i ) {
        res[i] = results_h[i].prim_idx;
    }*/
    return res;
}
std::vector<int> sample_points_debug3(std::vector<float>& boundary, std::vector<int>& next_pt, float radius) {
    START_TIMER;
    init_device();
    // auto res = triangulator::earcut(boundary);

    // 2. 创建光栅化器实例
    auto& rasterizer = graphics::VulkanCudaRasterizer::Instance();

    std::vector<graphics::Edge> edges(boundary.size() / 2);

    float factor = 1;
    for ( int i = 0; i < edges.size(); ++i ) {
        int j = next_pt[i];
        edges[i] = { { boundary[i * 2] * factor, boundary[i * 2 + 1] * factor },
            { boundary[j * 2] * factor, boundary[j * 2 + 1] * factor } };
    }
    // 准备数据
    RECORD_TIME("copy");

    // 3. 渲染 (动态分辨率，如 1024x1024)
    float* d_result = rasterizer.render(edges, 1024, 1024, 0);
    RECORD_TIME("render");
    std::vector<float> res_float(1024 * 1024);
    CUDA_CHECK(cudaMemcpy(res_float.data(), d_result, sizeof(float) * 1024 * 1024, cudaMemcpyDeviceToHost));
    std::vector<int> res(res_float.size());
    for ( int i = 0; i < res_float.size(); ++i ) {
        res[i] = (int)res_float[i];
    }
    RECORD_TIME("cpy");

    return res;
}


static void find_points_locations(
    const thrust::device_vector<float2>& query_pts,
    const thrust::device_vector<float2>& vertices,
    const thrust::device_vector<int3>& faces,
    thrust::device_vector<lbvh2d::LocationResult>& results) {
    unsigned int n_faces = faces.size();
    unsigned int n_queries = query_pts.size();

    if ( n_faces == 0 || n_queries == 0 ) return;

    lbvh2d::initialize(faces.size());
    BVH2D bvh;
    lbvh2d::build_face_bvh(vertices, faces, bvh);
    results.resize(n_queries);
    query_location_kernel<<<(n_queries + 255) / 256, 256>>>(
        thrust::raw_pointer_cast(query_pts.data()),
        n_queries,
        thrust::raw_pointer_cast(bvh.nodes.data()),
        thrust::raw_pointer_cast(bvh.aabbs.data()),
        bvh.root_idx,
        thrust::raw_pointer_cast(vertices.data()),
        thrust::raw_pointer_cast(faces.data()),
        thrust::raw_pointer_cast(results.data())
        );
    CUDA_CHECK(cudaDeviceSynchronize());
}

void find_map_weight_impl(
    const std::vector<float>& map_points,
    const std::vector<int>& map_tris,
    const std::vector<float>& query_points,
    std::vector<int>& res_index, std::vector<float>& res_weight
) {
    thrust::device_vector<float2> d_points(query_points.size() / 2);
    cudaMemcpy(d_points.data().get(), query_points.data(), d_points.size() * sizeof(float2), cudaMemcpyHostToDevice);
    thrust::device_vector<float2> d_vertices(map_points.size() / 2);
    cudaMemcpy(d_vertices.data().get(), map_points.data(), d_vertices.size() * sizeof(float2), cudaMemcpyHostToDevice);
    thrust::device_vector<int3> d_faces(map_tris.size() / 3);
    cudaMemcpy(d_faces.data().get(), map_tris.data(), d_faces.size() * sizeof(int3), cudaMemcpyHostToDevice);
    thrust::device_vector<lbvh2d::LocationResult> d_results;
    find_points_locations(d_points, d_vertices, d_faces, d_results);
    thrust::host_vector<lbvh2d::LocationResult> h_results = d_results;
    res_index.resize(h_results.size());
    res_weight.resize(h_results.size() * 3);
    for ( int i = 0; i < h_results.size(); ++i ) {
        auto res = h_results[i];
        res_index[i] = res.prim_idx;
        res_weight[i * 3] = res.u;
        res_weight[i * 3 + 1] = res.v;
        res_weight[i * 3 + 2] = res.w;
    }
}
