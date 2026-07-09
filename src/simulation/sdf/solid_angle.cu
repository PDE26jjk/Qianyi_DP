#include "solid_angle.cuh"

__device__ inline void my_swap(int& a, int& b) {
    int c = a;
    a = b;
    b = c;
}
__device__ inline void compute_integrals(
    const float3& a,
    const float3& b,
    const float3& c,
    const float3& P,
    float* integral_ii,
    float* integral_ij,
    float* integral_ik,
    int i)
{
    // a, b, c must be ordered along axis i.
    const float3 oab = b - a;
    const float3 oac = c - a;
    const float3 ocb = b - c;
    const float t = ((const float*)&oab)[i] / ((const float*)&oac)[i];

    const int j = (i == 2) ? 0 : (i + 1);
    const int k = (j == 2) ? 0 : (j + 1);
    const float jdiff = t * ((const float*)&oac)[j] - ((const float*)&oab)[j];
    const float kdiff = t * ((const float*)&oac)[k] - ((const float*)&oab)[k];

    float3 cross_a;
    cross_a.x = (jdiff * ((const float*)&oab)[k] - kdiff * ((const float*)&oab)[j]);
    cross_a.y = kdiff * ((const float*)&oab)[i];
    cross_a.z = jdiff * ((const float*)&oab)[i];

    float3 cross_c;
    cross_c.x = (jdiff * ((const float*)&ocb)[k] - kdiff * ((const float*)&ocb)[j]);
    cross_c.y = kdiff * ((const float*)&ocb)[i];
    cross_c.z = jdiff * ((const float*)&ocb)[i];

    const float area_scale_a = norm(cross_a);
    const float area_scale_c = norm(cross_c);
    const float Pai = ((const float*)&a)[i] - ((const float*)&P)[i];
    const float Pci = ((const float*)&c)[i] - ((const float*)&P)[i];

    // Integral of pi^2 over triangle area
    const float int_ii_a = area_scale_a * (0.5f * Pai * Pai + (2.0f / 3.0f) * Pai * ((const float*)&oab)[i] + 0.25f * ((const float*)&oab)[i] * ((const float*)&oab)[i]);
    const float int_ii_c = area_scale_c * (0.5f * Pci * Pci + (2.0f / 3.0f) * Pci * ((const float*)&ocb)[i] + 0.25f * ((const float*)&ocb)[i] * ((const float*)&ocb)[i]);
    *integral_ii = int_ii_a + int_ii_c;

    int jk = j;
    float* integral = integral_ij;
    float diff = jdiff;
    while (true) {
        if (integral) {
            float obmidj = ((const float*)&b)[jk] + 0.5f * diff;
            float oabmidj = obmidj - ((const float*)&a)[jk];
            float ocbmidj = obmidj - ((const float*)&c)[jk];
            float Paj = ((const float*)&a)[jk] - ((const float*)&P)[jk];
            float Pcj = ((const float*)&c)[jk] - ((const float*)&P)[jk];
            const float int_ij_a = area_scale_a
                * (0.5f * Pai * Paj + (1.0f / 3.0f) * Pai * oabmidj + (1.0f / 3.0f) * Paj * ((const float*)&oab)[i]
                   + 0.25f * ((const float*)&oab)[i] * oabmidj);
            const float int_ij_c = area_scale_c
                * (0.5f * Pci * Pcj + (1.0f / 3.0f) * Pci * ocbmidj + (1.0f / 3.0f) * Pcj * ((const float*)&ocb)[i]
                   + 0.25f * ((const float*)&ocb)[i] * ocbmidj);
            *integral = int_ij_a + int_ij_c;
        }
        if (jk == k) break;
        jk = k;
        integral = integral_ik;
        diff = kdiff;
    }
}


__device__ inline void precompute_triangle_solid_angle_props(
    const float3& a, const float3& b, const float3& c, lbvh3d::SolidAngleProps& my_data)
{
    const float3 ab = b - a;
    const float3 ac = c - a;

    // Weighted area normal
    const float3 N = 0.5f * cross(ab, ac);
    const float area2 = len_sq(N);
    const float area = sqrtf(area2);
    const float3 P = (a + b + c) / 3.0f;

    my_data.box.min = fmin3(fmin3( a,b),c);
    my_data.box.max = fmax3(fmax3( a,b),c);

    my_data.average_p = P;
    my_data.area_P = P * area;
    my_data.normal = N;
    my_data.area = area;

    // Moments at centroid (zero Nij)
    my_data.n_ij_diag = make_float3(0.0f, 0.0f, 0.0f);
    my_data.n_xy = my_data.n_yx = 0.0f;
    my_data.n_yz = my_data.n_zy = 0.0f;
    my_data.n_zx = my_data.n_xz = 0.0f;

    if (area == 0.0f) {
        my_data.n_ijk_diag = make_float3(0.0f, 0.0f, 0.0f);
        my_data.sum_permute_n_xyz = 0.0f;
        my_data.two_n_xxy_n_yxx = my_data.two_n_xxz_n_zxx = 0.0f;
        my_data.two_n_yyz_n_zyy = my_data.two_n_yyx_n_xyy = 0.0f;
        my_data.two_n_zzx_n_xzz = my_data.two_n_zzy_n_yzz = 0.0f;
        return;
    }

    float3 n = N / area;

    // Order vertices along each axis
    float3 values[3] = {a, b, c};
    int order_x[3] = {0,1,2};
    if (a.x > b.x) my_swap(order_x[0], order_x[1]);
    if (values[order_x[0]].x > c.x) my_swap(order_x[0], order_x[2]);
    if (values[order_x[1]].x > values[order_x[2]].x) my_swap(order_x[1], order_x[2]);
    float dx = values[order_x[2]].x - values[order_x[0]].x;

    int order_y[3] = {0,1,2};
    if (a.y > b.y) my_swap(order_y[0], order_y[1]);
    if (values[order_y[0]].y > c.y) my_swap(order_y[0], order_y[2]);
    if (values[order_y[1]].y > values[order_y[2]].y) my_swap(order_y[1], order_y[2]);
    float dy = values[order_y[2]].y - values[order_y[0]].y;

    int order_z[3] = {0,1,2};
    if (a.z > b.z) my_swap(order_z[0], order_z[1]);
    if (values[order_z[0]].z > c.z) my_swap(order_z[0], order_z[2]);
    if (values[order_z[1]].z > values[order_z[2]].z) my_swap(order_z[1], order_z[2]);
    float dz = values[order_z[2]].z - values[order_z[0]].z;

    float integral_xx = 0.0f, integral_xy = 0.0f, integral_yy = 0.0f;
    float integral_yz = 0.0f, integral_zz = 0.0f, integral_zx = 0.0f;

    if (dx > 0.0f) {
        compute_integrals(values[order_x[0]], values[order_x[1]], values[order_x[2]], P,
                          &integral_xx,
                          (dx >= dy && dy > 0.0f) ? &integral_xy : nullptr,
                          (dx >= dz && dz > 0.0f) ? &integral_zx : nullptr,
                          0);
    }
    if (dy > 0.0f) {
        compute_integrals(values[order_y[0]], values[order_y[1]], values[order_y[2]], P,
                          &integral_yy,
                          (dy >= dz && dz > 0.0f) ? &integral_yz : nullptr,
                          (dx < dy && dx > 0.0f) ? &integral_xy : nullptr,
                          1);
    }
    if (dz > 0.0f) {
        compute_integrals(values[order_z[0]], values[order_z[1]], values[order_z[2]], P,
                          &integral_zz,
                          (dx < dz && dx > 0.0f) ? &integral_zx : nullptr,
                          (dy < dz && dy > 0.0f) ? &integral_yz : nullptr,
                          2);
    }

    float3 Niii = make_float3(integral_xx, integral_yy, integral_zz);
    Niii = cw_mul(Niii, n);
    my_data.n_ijk_diag = Niii;
    my_data.sum_permute_n_xyz = 2.0f * (n.x * integral_yz + n.y * integral_zx + n.z * integral_xy);

    float Nxxy = n.x * integral_xy;
    float Nxxz = n.x * integral_zx;
    float Nyyz = n.y * integral_yz;
    float Nyyx = n.y * integral_xy;
    float Nzzx = n.z * integral_zx;
    float Nzzy = n.z * integral_yz;

    my_data.two_n_xxy_n_yxx = 2.0f * Nxxy + n.y * integral_xx;
    my_data.two_n_xxz_n_zxx = 2.0f * Nxxz + n.z * integral_xx;
    my_data.two_n_yyz_n_zyy = 2.0f * Nyyz + n.z * integral_yy;
    my_data.two_n_yyx_n_xyy = 2.0f * Nyyx + n.x * integral_yy;
    my_data.two_n_zzx_n_xzz = 2.0f * Nzzx + n.x * integral_zz;
    my_data.two_n_zzy_n_yzz = 2.0f * Nzzy + n.y * integral_zz;
}
typedef lbvh3d::SolidAngleProps SolidAngleProps;
__device__ inline void combine_precomputed_solid_angle_props(
    SolidAngleProps& my_data, const SolidAngleProps* left_child_data, const SolidAngleProps* right_child_data)
{
    float3 N = left_child_data->normal;
    float3 areaP = left_child_data->area_P;
    float area = left_child_data->area;
    if (right_child_data) {
        N = N + right_child_data->normal;
        areaP = areaP + right_child_data->area_P;
        area = area + right_child_data->area;
    }
    my_data.normal = N;
    my_data.area_P = areaP;
    my_data.area = area;

    lbvh3d::AABB3D box = left_child_data->box;
    if (right_child_data) {
        box.min = fmin3(box.min, right_child_data->box.min);
        box.max = fmax3(box.max, right_child_data->box.max);
    }
    my_data.box = box;

    float3 averageP;
    if (area > 0.0f) {
        averageP = areaP / area;
    } else {
        averageP = 0.5f * (box.min + box.max);
    }
    my_data.average_p = averageP;

    // Copy left child moments
    my_data.n_ij_diag = left_child_data->n_ij_diag;
    my_data.n_xy = my_data.n_yx = 0.0f;
    my_data.n_yz = my_data.n_zy = 0.0f;
    my_data.n_zx = my_data.n_xz = 0.0f;

    my_data.n_ijk_diag = left_child_data->n_ijk_diag;
    my_data.sum_permute_n_xyz = left_child_data->sum_permute_n_xyz;
    my_data.two_n_xxy_n_yxx = left_child_data->two_n_xxy_n_yxx;
    my_data.two_n_xxz_n_zxx = left_child_data->two_n_xxz_n_zxx;
    my_data.two_n_yyz_n_zyy = left_child_data->two_n_yyz_n_zyy;
    my_data.two_n_yyx_n_xyy = left_child_data->two_n_yyx_n_xyy;
    my_data.two_n_zzx_n_xzz = left_child_data->two_n_zzx_n_xzz;
    my_data.two_n_zzy_n_yzz = left_child_data->two_n_zzy_n_yzz;

    if (right_child_data) {
        my_data.n_ij_diag = my_data.n_ij_diag + right_child_data->n_ij_diag;
        my_data.n_ijk_diag = my_data.n_ijk_diag + right_child_data->n_ijk_diag;
        my_data.sum_permute_n_xyz += right_child_data->sum_permute_n_xyz;
        my_data.two_n_xxy_n_yxx += right_child_data->two_n_xxy_n_yxx;
        my_data.two_n_xxz_n_zxx += right_child_data->two_n_xxz_n_zxx;
        my_data.two_n_yyz_n_zyy += right_child_data->two_n_yyz_n_zyy;
        my_data.two_n_yyx_n_xyy += right_child_data->two_n_yyx_n_xyy;
        my_data.two_n_zzx_n_xzz += right_child_data->two_n_zzx_n_xzz;
        my_data.two_n_zzy_n_yzz += right_child_data->two_n_zzy_n_yzz;
    }

    for (int i = 0; i < (right_child_data ? 2 : 1); ++i) {
        const SolidAngleProps& child_data = (i == 0) ? *left_child_data : *right_child_data;
        float3 displacement = child_data.average_p - averageP;
        float3 child_N = child_data.normal;

        // Adjust Nij for displacement
        my_data.n_ij_diag = my_data.n_ij_diag + cw_mul(child_N, displacement);
        float Nxy = child_data.n_xy + child_N.x * displacement.y;
        float Nyx = child_data.n_yx + child_N.y * displacement.x;
        float Nyz = child_data.n_yz + child_N.y * displacement.z;
        float Nzy = child_data.n_zy + child_N.z * displacement.y;
        float Nzx = child_data.n_zx + child_N.z * displacement.x;
        float Nxz = child_data.n_xz + child_N.x * displacement.z;

        my_data.n_xy += Nxy; my_data.n_yx += Nyx;
        my_data.n_yz += Nyz; my_data.n_zy += Nzy;
        my_data.n_zx += Nzx; my_data.n_xz += Nxz;

        // Adjust Nijk for displacement
        my_data.n_ijk_diag = my_data.n_ijk_diag
            + 2.0f * cw_mul(displacement, child_data.n_ij_diag)
            + cw_mul(displacement, cw_mul(displacement, child_data.normal));
        my_data.sum_permute_n_xyz
            += (displacement.x * (Nyz + Nzy) + displacement.y * (Nzx + Nxz) + displacement.z * (Nxy + Nyx));
        my_data.two_n_xxy_n_yxx += 2.0f
                * (displacement.y * child_data.n_ij_diag.x + displacement.x * child_data.n_xy
                   + child_N.x * displacement.x * displacement.y)
            + 2.0f * child_data.n_yx * displacement.x + child_N.y * displacement.x * displacement.x;
        my_data.two_n_xxz_n_zxx += 2.0f
                * (displacement.z * child_data.n_ij_diag.x + displacement.x * child_data.n_xz
                   + child_N.x * displacement.x * displacement.z)
            + 2.0f * child_data.n_zx * displacement.x + child_N.z * displacement.x * displacement.x;
        my_data.two_n_yyz_n_zyy += 2.0f
                * (displacement.z * child_data.n_ij_diag.y + displacement.y * child_data.n_yz
                   + child_N.y * displacement.y * displacement.z)
            + 2.0f * child_data.n_zy * displacement.y + child_N.z * displacement.y * displacement.y;
        my_data.two_n_yyx_n_xyy += 2.0f
                * (displacement.x * child_data.n_ij_diag.y + displacement.y * child_data.n_yx
                   + child_N.y * displacement.y * displacement.x)
            + 2.0f * child_data.n_xy * displacement.y + child_N.x * displacement.y * displacement.y;
        my_data.two_n_zzx_n_xzz += 2.0f
                * (displacement.x * child_data.n_ij_diag.z + displacement.z * child_data.n_zx
                   + child_N.z * displacement.z * displacement.x)
            + 2.0f * child_data.n_xz * displacement.z + child_N.x * displacement.z * displacement.z;
        my_data.two_n_zzy_n_yzz += 2.0f
                * (displacement.y * child_data.n_ij_diag.z + displacement.z * child_data.n_zy
                   + child_N.z * displacement.z * displacement.y)
            + 2.0f * child_data.n_yz * displacement.z + child_N.y * displacement.z * displacement.z;
    }

    my_data.max_p_dist_sq = len_sq(fmax3(my_data.average_p - my_data.box.min,
                                            my_data.box.max - my_data.average_p));
}

namespace lbvh3d {

 
__global__ void refit_face_bvh_with_solid_angle_kernel(
    const float3* vertices,
    // const float3* additional_offset,
    const int3* faces,
    unsigned int n,
    const int2* __restrict__ nodes,
    const unsigned int* __restrict__ parent,
    unsigned int* __restrict__ child_count,
    // AABB3D* __restrict__ aabbs,
    SolidAngleProps* __restrict__ solid_angle_props)   // new parameter
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;

    // Leaf: compute triangle AABB
    unsigned int prim_idx = nodes[i].x - 1;
    int3 f = faces[prim_idx];
    float3 v0 = vertices[f.x], v1 = vertices[f.y], v2 = vertices[f.z];
    // aabbs[i].min = fmin3(v0, fmin3(v1, v2));
    // aabbs[i].max = fmax3(v0, fmax3(v1, v2));

    // if ( additional_offset ) {
    //     float3 v0_off = v0 + additional_offset[f.x];
    //     float3 v1_off = v1 + additional_offset[f.y];
    //     float3 v2_off = v2 + additional_offset[f.z];
    //     aabbs[i].min = fmin3(aabbs[i].min, fmin3(v0_off, fmin3(v1_off, v2_off)));
    //     aabbs[i].max = fmax3(aabbs[i].max, fmax3(v0_off, fmax3(v1_off, v2_off)));
    //     v0 = v0_off; v1 = v1_off; v2 = v2_off; // use offset vertices for solid angle
    // }

    // Compute solid angle props for the leaf triangle
    precompute_triangle_solid_angle_props(v0, v1, v2, solid_angle_props[i]);

    // Bottom-up refit
    unsigned int index = i;
    for ( ;; ) {
        unsigned int p = parent[index];
        if ( p == index || p == UINT_MAX ) return;

        __threadfence();
        unsigned int finished = atomicAdd(&child_count[p], 1);

        if ( finished == 1 ) {
            // Merge children AABB
            int2 node = nodes[p];
            unsigned int lc = node.x - 1;
            unsigned int rc = node.y - 1;
            // AABB3D a = aabbs[lc];
            // AABB3D b = aabbs[rc];
            // aabbs[p].min = make_float3(fminf(a.min.x, b.min.x),
            //                             fminf(a.min.y, b.min.y),
            //                             fminf(a.min.z, b.min.z));
            // aabbs[p].max = make_float3(fmaxf(a.max.x, b.max.x),
            //                             fmaxf(a.max.y, b.max.y),
            //                             fmaxf(a.max.z, b.max.z));

            // Merge solid angle props
            combine_precomputed_solid_angle_props(solid_angle_props[p],
                                                   &solid_angle_props[lc],
                                                   &solid_angle_props[rc]);

            index = p;
        }
        else {
            break;
        }
    }
}
void build_face_bvh_with_solid_angle(const thrust::device_vector<float3>& vertices,
                    const thrust::device_vector<int3>& faces, BVH3D& bvh,
                    thrust::device_vector<SolidAngleProps>& solid_angle_props)
{
    unsigned int n = faces.size();
    if ( n == 0 ) return;
    build_face_bvh(vertices,faces,bvh);
    const float3* d_verts = thrust::raw_pointer_cast(vertices.data());
    const int3* d_faces = thrust::raw_pointer_cast(faces.data());

    solid_angle_props.resize(bvh.nodes.size());   // <-- allocate solid angle array

    unsigned int num_nodes = 2 * n - 1;
    thrust::device_vector<unsigned int> child_count;
    child_count.assign(num_nodes, 0);

    refit_face_bvh_with_solid_angle_kernel<<<(n + 255) / 256, 256>>>(
        d_verts, d_faces, n,
        thrust::raw_pointer_cast(bvh.nodes.data()),
        thrust::raw_pointer_cast(bvh.parent.data()),
        thrust::raw_pointer_cast(child_count.data()),
        thrust::raw_pointer_cast(solid_angle_props.data()));   // <-- pass solid angle array
}
}
