// adapted from Newton sdf
#include "sdf.cuh"

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/system/detail/generic/remove.inl>

#include "solid_angle.cuh"
#include "common/geometric_algorithms.h"
#include "common/vec_math.h"

struct MeshDevice {
    unsigned int root_idx;
    const int2* nodes;
    const lbvh3d::AABB3D* aabbs;
    const float3* vertices;
    const int3* faces;
    const lbvh3d::SolidAngleProps* solid_angle_props;
};
__device__ inline int max_dim(float3 a) {
    float x = abs(a.x), y = abs(a.y), z = abs(a.z);
    return (x >= y && x >= z) ? 0 :
               (y >= z) ? 1 : 2;
}
__device__ inline float diff_product(float a, float b, float c, float d) {
    float cd = c * d;
    float diff = fmaf(a, b, -cd);
    float error = fmaf(-c, d, cd);

    return diff + error;
}
__device__ inline float xorf(float x, int y) { return __int_as_float(__float_as_int(x) ^ y); }

__device__ inline int sign_mask(float x) { return __float_as_int(x) & 0x80000000; }

// http://jcgt.org/published/0002/01/05/
__device__ inline bool intersect_ray_tri_woop(
    const float3& p,
    const float3& dir_,
    const float3& a,
    const float3& b,
    const float3& c,
    float& t
) {
    float dir[] = { dir_.x, dir_.y, dir_.z };

    int kz = max_dim(dir_);
    int kx = kz + 1;
    if ( kx == 3 )
        kx = 0;
    int ky = kx + 1;
    if ( ky == 3 )
        ky = 0;

    if ( dir[kz] < 0.0f ) {
        int tmp = kx;
        kx = ky;
        ky = tmp;
    }

    float Sx = dir[kx] / dir[kz];
    float Sy = dir[ky] / dir[kz];
    float Sz = 1.0f / dir[kz];

    float A[3] = { a.x - p.x, a.y - p.y, a.z - p.z };
    float B[3] = { b.x - p.x, b.y - p.y, b.z - p.z };
    float C[3] = { c.x - p.x, c.y - p.y, c.z - p.z };

    const float Ax = A[kx] - Sx * A[kz];
    const float Ay = A[ky] - Sy * A[kz];
    const float Bx = B[kx] - Sx * B[kz];
    const float By = B[ky] - Sy * B[kz];
    const float Cx = C[kx] - Sx * C[kz];
    const float Cy = C[ky] - Sy * C[kz];

    float U = diff_product(Cx, By, Cy, Bx);
    float V = diff_product(Ax, Cy, Ay, Cx);
    float W = diff_product(Bx, Ay, By, Ax);

    if ( U == 0.0f || V == 0.0f || W == 0.0f ) {
        double CxBy = (double)Cx * (double)By;
        double CyBx = (double)Cy * (double)Bx;
        U = (float)(CxBy - CyBx);
        double AxCy = (double)Ax * (double)Cy;
        double AyCx = (double)Ay * (double)Cx;
        V = (float)(AxCy - AyCx);
        double BxAy = (double)Bx * (double)Ay;
        double ByAx = (double)By * (double)Ax;
        W = (float)(BxAy - ByAx);
    }
    if ( (U < 0.0f || V < 0.0f || W < 0.0f) && (U > 0.0f || V > 0.0f || W > 0.0f) ) {
        return false;
    }

    float det = U + V + W;

    if ( det == 0.0f ) {
        return false;
    }

    const float Az = Sz * A[kz];
    const float Bz = Sz * B[kz];
    const float Cz = Sz * C[kz];
    const float T = U * Az + V * Bz + W * Cz;

    int det_sign = sign_mask(det);
    if ( xorf(T, det_sign)
        < 0.0f )  // || xorf(T,det_sign) > hit.t * xorf(det, det_sign)) // early out if hit.t is specified
    {
        return false;
    }

    const float rcpDet = 1.0f / det;
    t = T * rcpDet;
    // printf("(%f, %f, %f), (%f, %f, %f), (%f, %f, %f)\n",a.x,a.y,a.z,b.x,b.y,b.z,c.x,c.y,c.z);
    // printf("U:V:W:(%f, %f, %f)\n",U,V,W);
    // int3 signs = make_int3(sign_mask(W), sign_mask(V), sign_mask(Bx));
    return true;
}
__device__ inline bool
intersect_ray_aabb(const float3& pos, const float3& rcp_dir,
    const float3& lower, const float3& upper, float& t) {
    float l1, l2, lmin, lmax;

    l1 = (lower.x - pos.x) * rcp_dir.x;
    l2 = (upper.x - pos.x) * rcp_dir.x;
    lmin = min(l1, l2);
    lmax = max(l1, l2);

    l1 = (lower.y - pos.y) * rcp_dir.y;
    l2 = (upper.y - pos.y) * rcp_dir.y;
    lmin = max(min(l1, l2), lmin);
    lmax = min(max(l1, l2), lmax);

    l1 = (lower.z - pos.z) * rcp_dir.z;
    l2 = (upper.z - pos.z) * rcp_dir.z;
    lmin = max(min(l1, l2), lmin);
    lmax = min(max(l1, l2), lmax);

    bool hit = ((lmax >= 0.f) & (lmax >= lmin));
    if ( hit )
        t = lmin;

    return hit;
}

__device__ int mesh_query_ray_count_intersections(const MeshDevice* mesh, const float3& start, const float3& dir) {

    float3 rcp_dir = make_float3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);

    const float eps = 1.e-3f;
    int num_hit = 0;
    float temp_t;

    int stack[32];
    stack[0] = mesh->root_idx;
    int count = 1;
    while ( count ) {
        unsigned int node_idx = stack[--count];

        const auto& aabb = mesh->aabbs[node_idx];
        // todo: switch to robust ray-aabb, or expand bounds in build stage
        bool hit = intersect_ray_aabb(
            start, rcp_dir, aabb.min - eps, aabb.max + eps, temp_t
            );
        if ( hit ) {
            int2 node = mesh->nodes[node_idx];
            if ( node.y == 0 ) {
                unsigned int prim_idx = node.x - 1;
                int3 f = mesh->faces[prim_idx];

                float3 p = mesh->vertices[f.x];
                float3 q = mesh->vertices[f.y];
                float3 r = mesh->vertices[f.z];

                if ( intersect_ray_tri_woop(start, dir, p, q, r, temp_t) ) {
                    if ( temp_t >= 0.0f ) {
                        num_hit++;
                    }
                }
            }
            else {
                stack[count++] = node.x - 1;
                stack[count++] = node.y - 1;
            }
        }
    }

    return num_hit;
}
__device__ __host__ inline uint32_t lcg_next(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

//  return a float between [0, 1)
__device__ __host__ inline float randf_lcg(uint32_t& state) {
    return lcg_next(state) * (1.0f / 4294967296.0f);
}

__device__ __host__ inline float randf_range_lcg(uint32_t& state, float lo, float hi) {
    return lo + (hi - lo) * randf_lcg(state);
}
__device__ inline float
mesh_query_inside_parity(const MeshDevice* mesh, const float3& p, const float3 base_dir, int n_sample,
    float perturbation_scale) {
    int vote = 0;

    // deterministic
    // uint32_t rand_state = rand_init(42);
    uint32_t rand_state = 42 + blockIdx.x * blockDim.x + threadIdx.x;
    for ( int i = 0; i < n_sample; ++i ) {

        float3 dir;
        // do {
        //     dir = base_dir
        //         + make_float3(
        //             randf(rand_state, -perturbation_scale, perturbation_scale),
        //             randf(rand_state, -perturbation_scale, perturbation_scale),
        //             randf(rand_state, -perturbation_scale, perturbation_scale)
        //             );
        // } while ( length_sq(dir) < 1e-8f );
        dir = base_dir + make_float3(
            randf_range_lcg(rand_state, -perturbation_scale, perturbation_scale),
            randf_range_lcg(rand_state, -perturbation_scale, perturbation_scale),
            randf_range_lcg(rand_state, -perturbation_scale, perturbation_scale)
            );

        int hit_times = mesh_query_ray_count_intersections(mesh, p, dir);
        if ( hit_times % 2 ) {
            vote++;
        }
        // printf("p:(%f,%f,%f),dir:(%f,%f,%f),hit_times:%d\n", p.x, p.y, p.z, dir.x, dir.y, dir.z, hit_times);
    }

    if ( vote * 2 >= n_sample )
        return -1.0f;
    else
        return 1.0f;
}

__device__ float solid_angle_iterative(
    const MeshDevice* mesh,
    const float3& p,
    float accuracy_sq
) {
    // Define a small stack for iterative traversal.
    // Increase STACK_SIZE if deeper trees are expected.
    constexpr int STACK_SIZE = 64;
    int stack[STACK_SIZE];
    int at_child[STACK_SIZE];  // 0 = unvisited, 1 = left done, 2 = right done
    float angle[STACK_SIZE];

    stack[0] = mesh->root_idx;
    at_child[0] = 0;
    angle[0] = 0.0f;
    int count = 1;

    while ( count ) {
        const int node_idx = stack[count - 1];
        const int2 node = mesh->nodes[node_idx];

        if ( node.y == 0 ) {
            // Leaf node: compute exact solid angle of its triangle.
            unsigned int prim_idx = node.x - 1;
            int3 f = mesh->faces[prim_idx];
            float3 v0 = mesh->vertices[f.x];
            float3 v1 = mesh->vertices[f.y];
            float3 v2 = mesh->vertices[f.z];
            angle[count - 1] = lbvh3d::robust_solid_angle(v0, v1, v2, p);
            count--;
        }
        else {
            // Internal node.
            if ( at_child[count - 1] == 0 ) {
                // First visit: evaluate whether approximation is acceptable.
                float approx;
                bool need_descend = lbvh3d::evaluate_node_solid_angle(
                    p, mesh->solid_angle_props[node_idx], approx, accuracy_sq);
                if ( need_descend ) {
                    // Push left child.
                    unsigned int left_idx = node.x - 1;
                    stack[count] = left_idx;
                    at_child[count - 1] = 1;
                    angle[count] = 0.0f;
                    at_child[count] = 0;
                    count++;
                }
                else {
                    // Approximation is accurate enough – do not descend.
                    angle[count - 1] = approx;
                    count--;
                }
            }
            else if ( at_child[count - 1] == 1 ) {
                // Returned from left child – accumulate its angle.
                angle[count - 1] += angle[count];
                // Push right child.
                unsigned int right_idx = node.y - 1;
                stack[count] = right_idx;
                at_child[count - 1] = 2;
                angle[count] = 0.0f;
                at_child[count] = 0;
                count++;
            }
            else {
                // Both children processed – accumulate right child's angle.
                angle[count - 1] += angle[count];
                count--;
            }
        }
    }
    return angle[0];
}

__device__ float query_mesh_sdf(const MeshDevice* mesh, float3 pos, float max_dist, float winding_threshold, bool use_parity
) {
    const float3& qp = pos;
    float best_dist = max_dist * max_dist;
    int best_prim = INT_MAX;

    unsigned int stack[64];
    auto* aabbs = mesh->aabbs;
    int sp = 0;
    stack[sp++] = mesh->root_idx;

    while ( sp > 0 ) {
        unsigned int node_idx = stack[--sp];
        if ( lbvh3d::dist_sq_point_aabb_3d(qp, aabbs[node_idx]) >= best_dist ) continue;

        int2 node = mesh->nodes[node_idx];

        if ( node.y == 0 ) {
            unsigned int prim_idx = node.x - 1;
            if ( lbvh3d::dist_sq_point_aabb_3d(qp, aabbs[prim_idx]) > best_dist ) continue;
            int3 f = mesh->faces[prim_idx];

            float3 v0 = mesh->vertices[f.x];
            float3 v1 = mesh->vertices[f.y];
            float3 v2 = mesh->vertices[f.z];

            float dist_sq = dist_sq_point_triangle_3d(qp, v0, v1, v2);

            if ( dist_sq < best_dist ) {
                best_dist = dist_sq;
                best_prim = prim_idx;

                if ( best_dist < 1e-16f ) break;
            }
        }
        else if ( sp < 64 - 2 ) {
            unsigned int left_child = node.x - 1;
            unsigned int right_child = node.y - 1;

            float dist_left = lbvh3d::dist_sq_point_aabb_3d(qp, aabbs[left_child]);
            float dist_right = lbvh3d::dist_sq_point_aabb_3d(qp, aabbs[right_child]);

            if ( dist_left > dist_right ) {
                if ( dist_left < best_dist ) stack[sp++] = left_child;
                if ( dist_right < best_dist ) stack[sp++] = right_child;
            }
            else {
                if ( dist_right < best_dist ) stack[sp++] = right_child;
                if ( dist_left < best_dist ) stack[sp++] = left_child;
            }
        }
    }

    float dist = 0, inside = 0;
    if ( best_prim != INT_MAX ) {
        dist = sqrtf(best_dist);
        if ( use_parity ) {
            inside = mesh_query_inside_parity(mesh, qp, make_float3(1.1f, 1.2f, 1.f), 3, 0.2f);
        }
        else if ( mesh->solid_angle_props ) {
            constexpr float accuracy = 2.f;
            float angle = solid_angle_iterative(mesh, qp, accuracy * accuracy);
            inside = (angle * 0.07957747154f > winding_threshold) ? -1.0f : 1.0f;
        }
    }
    return dist * inside;
}
__global__ void build_coarse_sdf_kernel(
    const MeshDevice*__restrict__ mesh,
    float* __restrict__ bg_sdf,
    float3 min_corner,
    float3 cell_size,
    int cells_per_subgrid,
    int3 bg,
    float winding_threshold,
    bool use_parity
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bg.x * bg.y * bg.z;
    if ( tid >= total ) return;

    int3 coords = id_to_xyz(tid, bg.x, bg.y);

    float3 pos = min_corner + make_float3(
        coords.x * cells_per_subgrid * cell_size.x,
        coords.y * cells_per_subgrid * cell_size.y,
        coords.z * cells_per_subgrid * cell_size.z
        );

    bg_sdf[tid] = query_mesh_sdf(mesh, pos, 10000.0f, winding_threshold, use_parity);
}
__global__ void check_subgrid_occupied_kernel(
    MeshDevice* mesh,
    float2 threshold,
    float winding_threshold,
    bool use_parity,
    int* subgrid_required,
    int cells_per_subgrid,
    int nx, int ny,
    float3 min_corner,
    float3 cell_size,
    int total
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid >= total ) return;

    int3 coords = id_to_xyz(tid, nx, ny);
    float3 center = min_corner + make_float3(
        (coords.x * cells_per_subgrid + cells_per_subgrid * 0.5f) * cell_size.x,
        (coords.y * cells_per_subgrid + cells_per_subgrid * 0.5f) * cell_size.y,
        (coords.z * cells_per_subgrid + cells_per_subgrid * 0.5f) * cell_size.z
        );
    float sdf = query_mesh_sdf(mesh, center, 10000.0f, winding_threshold, use_parity);
    if ( (sdf > 0.f && sdf < threshold.y) || (sdf <= 0.f && sdf > threshold.x) )
        subgrid_required[tid] = 1;
    else
        subgrid_required[tid] = 0;
}
__device__ inline int3 write_subgrid_slot(
    uint32_t* start_slots,
    int address,
    int tex_blocks_per_dim,
    int bx, int by, int bz,
    int w, int h,
    int local_sample) {
    int3 ac;
    ac.z = address / (tex_blocks_per_dim * tex_blocks_per_dim);
    int r = address - ac.z * tex_blocks_per_dim * tex_blocks_per_dim;
    ac.y = r / tex_blocks_per_dim;
    ac.x = r - ac.y * tex_blocks_per_dim;

    if ( local_sample == 0 ) {
        uint32_t slot = (uint32_t)ac.x |
            ((uint32_t)ac.y << 10) |
            ((uint32_t)ac.z << 20);
        int slot_idx = idx3d(bx, by, bz, w, h);
        start_slots[slot_idx] = slot;
    }
    return ac;
}
__global__ void populate_subgrid_texture_kernel(
    MeshDevice* mesh,
    const int* subgrid_addresses_inv,  // length = num_required
    uint32_t* subgrid_start_slots,
    float* subgrid_texture,
    int cells_per_subgrid,
    float3 min_corner,
    float3 cell_size,
    float winding_threshold,
    bool use_parity,
    int nx, int ny, int nz,
    int tex_blocks_per_dim,
    int tex_size,
    int num_required,
    float sdf_min,
    float sdf_range_inv,
    sdf::QuantizationMode mode
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int samples_per_dim = cells_per_subgrid + 1;
    int samples_per_subgrid = samples_per_dim * samples_per_dim * samples_per_dim;
    int total_work = num_required * samples_per_subgrid;

    if ( tid >= total_work ) return;

    int valid_idx = tid / samples_per_subgrid;
    int local_sample = tid - valid_idx * samples_per_subgrid;

    int subgrid_idx = subgrid_addresses_inv[valid_idx];

    int3 sg = id_to_xyz(subgrid_idx, nx, ny);
    int3 ls = id_to_xyz(local_sample, samples_per_dim, samples_per_dim);
    int gx = sg.x * cells_per_subgrid + ls.x;
    int gy = sg.y * cells_per_subgrid + ls.y;
    int gz = sg.z * cells_per_subgrid + ls.z;

    float3 pos = make_float3(
        min_corner.x + gx * cell_size.x,
        min_corner.y + gy * cell_size.y,
        min_corner.z + gz * cell_size.z
        );
    float sdf_val = query_mesh_sdf(mesh, pos, 10000.0f, winding_threshold, use_parity);

    int3 ac = write_subgrid_slot(subgrid_start_slots, valid_idx, tex_blocks_per_dim,
        sg.x, sg.y, sg.z, nx, ny, local_sample);

    int tex_idx = idx3d(ac.x * samples_per_dim + ls.x,
        ac.y * samples_per_dim + ls.y,
        ac.z * samples_per_dim + ls.z,
        tex_size, tex_size);
    if ( mode == sdf::QuantizationMode::FLOAT32 ) {
        subgrid_texture[tex_idx] = sdf_val;
    }
    else if ( mode == sdf::QuantizationMode::UINT16 ) {
        float normalized = fminf(fmaxf((sdf_val - sdf_min) * sdf_range_inv, 0.0f), 1.0f);
        ((uint16_t*)subgrid_texture)[tex_idx] = (uint16_t)(normalized * 65535.0f);
    }
    else if ( mode == sdf::QuantizationMode::UINT8 ) {
        float normalized = fminf(fmaxf((sdf_val - sdf_min) * sdf_range_inv, 0.0f), 1.0f);
        ((uint8_t*)subgrid_texture)[tex_idx] = (uint8_t)(normalized * 255.0f);
    }
}


sdf::SDF::SDF() {
    params.coarse_texture = 0;
    params.subgrid_texture = 0;
    params.coarse_array = nullptr;
    params.subgrid_array = nullptr;
}
struct IsActiveCell {
    __host__ __device__ bool operator()(int required) const {
        return required == 1;
    }
};
void sdf::SDF::build_from_mesh(const int3* faces, const float3* pos, const lbvh3d::BVH3D& bvh, int num_vertices,
    int max_resolution, float target_voxel_size, int subgrid_size, bool use_parity, const lbvh3d::SolidAngleProps* solid_angle_props) {
    release_textures();

    float3 mesh_min, mesh_max;
    lbvh3d::compute_bounds(pos, num_vertices, mesh_min, mesh_max);
    float margin = 0.05f; // cm
    const float3 min_ext = mesh_min - margin;
    const float3 max_ext = mesh_max + margin;
    float3 ext = max_ext - min_ext;
    float max_ext_scalar = fmaxf(fmaxf(ext.x, ext.y), ext.z);
    if ( max_ext_scalar < 1e-10f ) {
        throw std::runtime_error("SDF::build_from_mesh: max_ext_scalar < 1e-10f");
    }
    if ( target_voxel_size > 0.0f ) {
        int derived_res = (int)ceilf(max_ext_scalar / target_voxel_size);
        derived_res = max(8, ((derived_res + 7) / 8) * 8);
        max_resolution = derived_res;
    }
    else if ( max_resolution <= 0 ) {
        max_resolution = 64;
    }

    if ( max_resolution <= 0 || max_resolution >= (1 << 16) ) {
        throw std::runtime_error("SDF::build_from_mesh: Invalid max_resolution");
    }

    float cell_size_scalar = max_ext_scalar / (float)max_resolution;
    int3 dims = make_int3(
        (int)ceilf(ext.x / cell_size_scalar) + 1,
        (int)ceilf(ext.y / cell_size_scalar) + 1,
        (int)ceilf(ext.z / cell_size_scalar) + 1
        );
    float3 cell_size = make_float3(
        ext.x / (dims.x - 1),
        ext.y / (dims.y - 1),
        ext.z / (dims.z - 1)
        );
    float2 narrow_band_range = { -0.01f, 0.01f };
    float narrow_band_thickness = fmaxf(fabsf(narrow_band_range.x), fabsf(narrow_band_range.y));

    // build_sparse_sdf_from_mesh
    int3 num_cells = dims - 1;
    int w = (num_cells.x + subgrid_size - 1) / subgrid_size;
    int h = (num_cells.y + subgrid_size - 1) / subgrid_size;
    int d = (num_cells.z + subgrid_size - 1) / subgrid_size;
    int total_subgrids = w * h * d;

    int3 whd{ w, h, d };
    int3 bg_cells = whd + 1;
    int total_bg = bg_cells.x * bg_cells.y * bg_cells.z;

    thrust::device_vector<float>& d_bg_sdf = params.d_bg_sdf;
    d_bg_sdf.resize(total_bg);

    MeshDevice mesh{
        .root_idx = bvh.root_idx,
        .nodes = thrust::raw_pointer_cast(bvh.nodes.data()),
        .aabbs = thrust::raw_pointer_cast(bvh.aabbs.data()),
        .vertices = pos,
        .faces = faces,
        .solid_angle_props = solid_angle_props
    };
    thrust::device_vector<MeshDevice> d_mesh(1, mesh);

    MeshDevice* mesh_ptr = d_mesh.data().get();

    // 3. 构建粗网格 SDF
    int block(256);
    // dim3 grid((total_bg + block - 1) / block);
    float winding_threshold = 0.5f;
    build_coarse_sdf_kernel<<<(total_bg + block - 1) / block, block>>>(
        mesh_ptr, d_bg_sdf.data().get(), min_ext, cell_size,
        subgrid_size, bg_cells, winding_threshold, use_parity);

    // 4. 标记占用子网格
    thrust::device_vector<int> d_subgrid_required(total_subgrids);
    float3 half_subgrid = make_float3(subgrid_size * 0.5f * cell_size.x,
        subgrid_size * 0.5f * cell_size.y,
        subgrid_size * 0.5f * cell_size.z);
    float subgrid_radius = norm(half_subgrid);
    float2 threshold = make_float2(-narrow_band_thickness - subgrid_radius,
        narrow_band_thickness + subgrid_radius);
    // 使用3D grid 覆盖所有子网格
    check_subgrid_occupied_kernel<<<(total_subgrids + block - 1) / block, block>>>(mesh_ptr, threshold, winding_threshold,
        use_parity,
        d_subgrid_required.data().get(), subgrid_size, w, h,
        min_ext, cell_size, total_subgrids);

    // 5. 线性度检测（可选）
    // float* d_linearity_errors = nullptr;
    // int* d_subgrid_is_linear = nullptr;
    // float linearization_error_threshold = 0.005f;
    // if ( linearization_error_threshold < 0.0f ) {
    //     linearization_error_threshold = 1e-6f * sqrtf(ext.x * ext.x + ext.y * ext.y + ext.z * ext.z);
    // }
    // if ( linearization_error_threshold > 0.0f ) {
    //     cudaMalloc(&d_linearity_errors, total_subgrids * sizeof(float));
    //     cudaMemset(d_linearity_errors, 0, total_subgrids * sizeof(float));
    //     cudaMalloc(&d_subgrid_is_linear, total_subgrids * sizeof(int));
    //     cudaMemset(d_subgrid_is_linear, 0, total_subgrids * sizeof(int));
    //
    //     int samples_per_dim = subgrid_size + 1;
    //     int samples_per_subgrid = samples_per_dim * samples_per_dim * samples_per_dim;
    //     int total_work = total_subgrids * samples_per_subgrid;
    //     dim3 grid_lin((total_work + block.x - 1) / block.x);
    //     accumulate_linearity_error_kernel<<<grid_lin, block>>>(
    //         mesh_id, d_bg_sdf, d_subgrid_required, d_linearity_errors,
    //         subgrid_size, min_corner, cell_size, winding_threshold, use_parity,
    //         w, h, d, bg_x, bg_y, bg_z);
    //
    //     apply_linearity_kernel<<<(total_subgrids + block.x - 1) / block.x, block>>>(
    //         d_subgrid_required, d_linearity_errors, d_subgrid_is_linear, linearization_error_threshold);
    // }

    // 6. 前缀和计算地址
    thrust::device_vector<int> d_subgrid_addresses(total_subgrids);
    thrust::exclusive_scan(thrust::device, d_subgrid_required.begin(), d_subgrid_required.end(), d_subgrid_addresses.begin());

    // 7. 回读所需子网格数
    int num_required = d_subgrid_addresses.back() + d_subgrid_required.back();
    if ( num_required == 0 ) {
        throw std::runtime_error("SDF::build_from_mesh: No subgrid available");
    }
    thrust::device_vector<int> d_subgrid_addresses_inv(num_required);
    // thrust::lower_bound(thrust::device,
    //     d_subgrid_addresses.begin(), d_subgrid_addresses.end(),
    //     thrust::make_counting_iterator(0), thrust::make_counting_iterator(num_required),
    //     d_subgrid_addresses_inv.begin());
    thrust::copy_if(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator((int)d_subgrid_required.size()),
        d_subgrid_required.begin(),
        d_subgrid_addresses_inv.begin(),
        IsActiveCell()
        );

    // 8. 准备纹理数据
    float global_sdf_min = threshold.x;
    float global_sdf_max = threshold.y;
    float sdf_range = global_sdf_max - global_sdf_min;
    if ( sdf_range < 1e-10f ) sdf_range = 1.0f;
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::device_vector<uint32_t>& d_start_slots = params.subgrid_start_slots;
    d_start_slots.assign(w * h * d, sdf::SLOT_EMPTY);

    // 计算纹理大小
    int tex_size = 1;
    thrust::device_vector<float>& d_subgrid_texture = params.d_subgrid_texture;
    int cubic_root = ceil(cbrt(num_required));
    int tex_blocks_per_dim = max(1, cubic_root);
    while ( tex_blocks_per_dim * tex_blocks_per_dim * tex_blocks_per_dim < num_required )
        tex_blocks_per_dim++;

    int samples_per_dim = subgrid_size + 1;
    tex_size = tex_blocks_per_dim * samples_per_dim;
    int total_tex = tex_size * tex_size * tex_size;
    QuantizationMode quantization_mode = QuantizationMode::FLOAT32;
    // 根据量化模式分配不同类型纹理
    int samples_per_subgrid = samples_per_dim * samples_per_dim * samples_per_dim;
    int total_work = num_required * samples_per_subgrid;
    if ( quantization_mode == QuantizationMode::FLOAT32 ) { // FLOAT32
        d_subgrid_texture.resize(total_tex);
        global_sdf_min = 0.f;
        sdf_range = 1.f;
    }
    else if ( quantization_mode == QuantizationMode::UINT16 ) { // UINT16
        d_subgrid_texture.resize(total_tex / 2);
    }
    else {
        d_subgrid_texture.resize(total_tex / 4);
    }
    thrust::fill(d_subgrid_texture.begin(), d_subgrid_texture.end(), 0.0f);

    populate_subgrid_texture_kernel<<<(total_work + block - 1) / block, block>>>(
        mesh_ptr, d_subgrid_addresses_inv.data().get(), d_start_slots.data().get(),
        d_subgrid_texture.data().get(), subgrid_size, min_ext, cell_size,
        winding_threshold, use_parity, w, h, d, tex_blocks_per_dim, tex_size, num_required,
        global_sdf_min, 1.0f / sdf_range, quantization_mode);

    // 9. 处理线性子网格标记
    // if ( d_subgrid_is_linear ) {
    //     int* h_is_linear = new int[total_subgrids];
    //     cudaMemcpy(h_is_linear, d_subgrid_is_linear, total_subgrids * sizeof(int), cudaMemcpyDeviceToHost);
    //     for ( int i = 0; i < total_subgrids; i++ ) {
    //         if ( h_is_linear[i] ) {
    //             int3 coord = id_to_xyz(i, w, h);
    //             h_start_slots[idx3d(coord.x, coord.y, coord.z, w, h)] = SLOT_LINEAR;
    //         }
    //     }
    //     delete[] h_is_linear;
    // }


    params.coarse_dims = whd;
    params.subgrid_tex_size = tex_size;
    // params.num_subgrids = num_required;
    params.min_extents = min_ext;
    params.max_extents = max_ext;
    params.cell_size = cell_size;
    params.subgrid_size = subgrid_size;
    params.quantization_mode = quantization_mode;
    params.sdf_min_value = global_sdf_min;
    params.sdf_range = sdf_range;

    cudaArray_t coarse_array = nullptr;
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
    cudaExtent extent = make_cudaExtent(bg_cells.x, bg_cells.y, bg_cells.z);
    cudaMalloc3DArray(&coarse_array, &channel_desc, extent);

    cudaMemcpy3DParms copy_params = { 0 };
    copy_params.srcPtr = make_cudaPitchedPtr(
        d_bg_sdf.data().get(),
        bg_cells.x * sizeof(float),
        bg_cells.x,
        bg_cells.y
        );
    copy_params.dstArray = coarse_array;
    copy_params.extent = extent;
    copy_params.kind = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&copy_params);

    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = coarse_array;

    cudaTextureDesc tex_desc = {};
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.addressMode[2] = cudaAddressModeClamp;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = false;

    cudaTextureObject_t coarse_tex = 0;
    cudaCreateTextureObject(&coarse_tex, &res_desc, &tex_desc, nullptr);
    cudaArray_t subgrid_array = nullptr;
    cudaChannelFormatDesc sub_channel_desc;
    if ( quantization_mode == QuantizationMode::FLOAT32 ) {
        sub_channel_desc = cudaCreateChannelDesc<float>();
    }
    else if ( quantization_mode == QuantizationMode::UINT16 ) {
        sub_channel_desc = cudaCreateChannelDesc<uint16_t>();
        tex_desc.readMode = cudaReadModeNormalizedFloat;
    }
    else {
        sub_channel_desc = cudaCreateChannelDesc<uint8_t>();
        tex_desc.readMode = cudaReadModeNormalizedFloat;
    }

    cudaExtent sub_extent = make_cudaExtent(tex_size, tex_size, tex_size);
    cudaMalloc3DArray(&subgrid_array, &sub_channel_desc, sub_extent);

    // 注意：d_subgrid_texture 中的数据需要按 pitched 方式拷贝
    // 由于 d_subgrid_texture 是连续的一维数组，需要构造 pitched pointer
    cudaMemcpy3DParms sub_copy_params = { 0 };
    sub_copy_params.srcPtr = make_cudaPitchedPtr(
        d_subgrid_texture.data().get(),
        tex_size * (quantization_mode == QuantizationMode::FLOAT32 ? sizeof(float) :
                        quantization_mode == QuantizationMode::UINT16 ? sizeof(uint16_t) : sizeof(uint8_t)),
        tex_size,
        tex_size
        );
    sub_copy_params.dstArray = subgrid_array;
    sub_copy_params.extent = sub_extent;
    sub_copy_params.kind = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&sub_copy_params);
    cudaResourceDesc sub_res_desc = {};
    sub_res_desc.resType = cudaResourceTypeArray;
    sub_res_desc.res.array.array = subgrid_array;

    cudaTextureObject_t subgrid_tex = 0;
    cudaCreateTextureObject(&subgrid_tex, &sub_res_desc, &tex_desc, nullptr);
    params.coarse_texture = coarse_tex;
    params.subgrid_texture = subgrid_tex;
    params.coarse_array = coarse_array;
    params.subgrid_array = subgrid_array;
}

__global__ void query_mesh_sdf_kernel(
    const MeshDevice* mesh,
    const float3* points,
    float* bg_sdf,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid >= n ) return;
    float3 pos = points[tid];

    bg_sdf[tid] = query_mesh_sdf(mesh, pos, 10000.0f, 0.1f, true);
}
void sdf::SDF::release_textures() {
    if ( params.coarse_texture ) {
        CUDA_CHECK(cudaDestroyTextureObject(params.coarse_texture));
        params.coarse_texture = 0;
    }
    if ( params.subgrid_texture ) {
        CUDA_CHECK(cudaDestroyTextureObject(params.subgrid_texture));
        params.subgrid_texture = 0;
    }
    if ( params.coarse_array ) {
        CUDA_CHECK(cudaFreeArray(params.coarse_array));
        params.coarse_array = nullptr;
    }
    if ( params.subgrid_array ) {
        CUDA_CHECK(cudaFreeArray(params.subgrid_array));
        params.subgrid_array = nullptr;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}
void sdf::check_inside(const int3* faces, const float3* pos,
    const lbvh3d::BVH3D& bvh, const float3* query_points,
    const int query_points_size, float* res) {

    MeshDevice mesh{
        .root_idx = bvh.root_idx,
        .nodes = thrust::raw_pointer_cast(bvh.nodes.data()),
        .aabbs = thrust::raw_pointer_cast(bvh.aabbs.data()),
        .vertices = pos,
        .faces = faces
    };
    thrust::device_vector<MeshDevice> d_mesh(1, mesh);

    MeshDevice* mesh_ptr = d_mesh.data().get();
    int block = 256;
    query_mesh_sdf_kernel<<<(query_points_size + block - 1) / block, block>>>(
        mesh_ptr, query_points, res, query_points_size);
}
