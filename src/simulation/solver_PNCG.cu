#include "solver_PNCG.cuh"

#include <thrust/transform_reduce.h>

#include "constraint.cuh"
#include "contact/collision.cuh"
#include "dynamics/bending.cuh"
#include "dynamics/planar.cuh"

void SolverPNCG::init() {
    SolverBase::init();
    Jx_diag.resize(params.nb_all_cloth_vertices);
    forces.resize(params.nb_all_cloth_vertices);
    Jx.resize(params.nb_all_cloth_edges);
    Jx_bend_cross.resize(params.nb_all_cloth_vertices);
    p.resize(params.nb_all_vertices);
}

static __global__ void update_begin(
    float3* __restrict__ vertices_world,
    float3* __restrict__ x_tilde,
    const float3* __restrict__ vertices,
    const float3* __restrict__ velocities,
    const int* __restrict__ vertices_obj,
    const Mat4* __restrict__ world_matrices,
    const char* __restrict__ mask,
    // const float* __restrict__ masses,
    const int nb_cloth_vertices,
    const float h,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        int obj = vertices_obj[i];
        auto x0 = mul_homo(world_matrices[obj], vertices[i]);
        vertices_world[i] = x0;
        if ( mask[i] || i >= nb_cloth_vertices ) {
            x_tilde[i] = x0;
        }
        else {
            x_tilde[i] = x0 + h * velocities[i] + make_float3(0.f, 0.f, h * h * -9.8f);
        }
    }
}

static __global__ void update_end(
    float3* __restrict__ vertices_local,
    float3* __restrict__ velocities,
    const float3* __restrict__ vertices_prev,
    const float3* __restrict__ vertices_world,
    const char* __restrict__ vertices_mask,
    const int* __restrict__ vertices_proxy,
    const int* __restrict__ vertices_obj,
    const Mat4* __restrict__ world_matrices_inv,
    const float h,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        int idx = vertices_proxy[i];
        auto x = vertices_world[idx];
        auto v = (x - vertices_prev[idx]) / h;
        if ( !vertices_mask[idx] ) {
            v = v * expf(-h * 0.5f);
            constexpr float max_velocity = 20.f;
            if ( norm(v) > max_velocity )
                v = normalized(v) * max_velocity;
            velocities[i] = v;
            x = vertices_prev[idx] + v * h;
        }
        else {
            x = vertices_prev[idx];
            velocities[i] = make_float3(0.f, 0.f, 0.f);
        }

        int obj = vertices_obj[i];
        vertices_local[i] = mul_homo(world_matrices_inv[obj], x);
    }
}

static __global__ void update_elastic_constraint_end1(
    Mat3* __restrict__ Jx_diag,
    float3* __restrict__ forces,
    const float3* __restrict__ x,
    const float3* __restrict__ x_tilde,
    const float* __restrict__ masses,
    const char* __restrict__ mask,
    const float h_sqr,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        float mass = mask[i] ? 1e6f : masses[i];
        Jx_diag[i] *= h_sqr;
        Jx_diag[i] -= Mat3::identity(mass);
        forces[i] = forces[i] * h_sqr + (x_tilde[i] - x[i]) * mass;
    }
}
static __global__ void update_elastic_constraint_end2(
    Mat3* __restrict__ Jx,
    Mat3* __restrict__ Jx_bend_cross,
    const float h_sqr,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        Jx[i] *= h_sqr;
        Jx_bend_cross[i] *= h_sqr;
    }
}
void SolverPNCG::compute_elastic_constraint(float3* x, float3* x_tilde, float3* forces) {
    int block = 256;
    // const auto gravity = make_float3(0.f, 0.f, -9.8f);
    int nb_cloth_vertices = params.nb_all_cloth_vertices;
    int nb_cloth_edges = params.nb_all_cloth_edges;

    Mat3* Jx_diag = this->Jx_diag.data().get();
    Mat3* Jx = this->Jx.data().get();
    Mat3* Jx_bend_cross = this->Jx_bend_cross.data().get();
    cudaMemsetAsync(forces, 0, nb_cloth_vertices * sizeof(float3));
    cudaMemsetAsync(Jx_diag, 0, nb_cloth_vertices * sizeof(Mat3));
    cudaMemsetAsync(Jx, 0, nb_cloth_edges * sizeof(Mat3));
    cudaMemsetAsync(Jx_bend_cross, 0, nb_cloth_edges * sizeof(Mat3));


    int n = nb_cloth_edges;
    int blocksPerGrid = (n + block - 1) / block;
    compute_spring_forces<<<blocksPerGrid, block>>>(
        Jx, Jx_diag,
        forces, nullptr,
        vertices_world.data().get(),
        edges.data().get(),
        edge_lengths.data().get(),
        n);
    n = params.nb_all_cloth_triangles;

    // compute_BW_FEM<<<(n + block - 1) / block, block>>>(
    //     // compute_ARAP_FEM<<<(n + block - 1) / block, block>>>(
    //     Jx, Jx_diag,
    //     forces, nullptr,
    //     x,
    //     triangles.data().get(),
    //     edges.data().get(),
    //     vertices_obj.data().get(),
    //     nullptr,
    //     Dms.data().get(),
    //     n);
    n = params.nb_all_cloth_edges;

    // compute_quadratic_Bending_IBM<<<(n + block - 1) / block, block>>>(
    //     Jx, Jx_diag, Jx_bend_cross,
    //     forces, nullptr,
    //     IBM_q.data().get(),
    //     x,
    //     edges.data().get(),
    //     e2t.data().get(),
    //     triangles.data().get(),
    //     edge_opposite_points.data().get(),
    //     n, 0.2f);
    if ( !sewing_done ) {
        float min_dist = 2e-3f;
        n = params.nb_all_stitches;
        compute_stitch_constraint<<<(n + block - 1) / block, block>>>(
            nullptr, Jx_diag, forces, nullptr, x, vertices_obj.data().get(), obj_data.data().get(),
            vertices_mask.data().get(), stitches.data().get(), min_dist,1e6f, n);
    }
    float h_sqr = dt * dt;
    update_elastic_constraint_end1<<<(nb_cloth_vertices + block - 1) / block, block>>>(
        Jx_diag, forces, x, x_tilde, masses.data().get(),
        vertices_mask.data().get(), h_sqr, nb_cloth_vertices);
    update_elastic_constraint_end2<<<(nb_cloth_edges + block - 1) / block, block>>>(
        Jx, Jx_bend_cross, h_sqr, nb_cloth_edges);
}


static __global__ void compute_collision_Hp_neg(
    float3* __restrict__ Hp_neg,
    const float3* __restrict__ p,
    const UnifiedNormalConstraint* __restrict__ constraints,
    const float* __restrict__ invMass,
    float d_hat,
    int num_constraints
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid >= num_constraints ) return;

    auto& c = constraints[tid];

    // float3 t = make_float3(0.f, 0.f, 0.f);
    float3 p_sum = make_float3(0.f, 0.f, 0.f); // 类似求雅可比乘以 p
    #pragma unroll
    for ( int i = 0; i < 4; ++i ) {
        // t += c.w[i] * x[c.v[i]];
        if ( invMass[c.v[i]] > 0.f )
            p_sum += c.w[i] * p[c.v[i]];
    }

    float d = c.lambda;
    if ( d >= d_hat || d <= 1e-6f ) return;

    // float3 u = t / d;
    // float d_ratio = d / d_hat;
    // float log_term = logf(d_ratio);
    //
    // float E_prime = -kappa * (d - d_hat) * (2.0f * log_term + 1.0f - 1.0f / d_ratio);
    // float E_double_prime = -kappa * (2.0f * log_term + 3.0f - 2.0f / d_ratio - 1.0f / (d_ratio * d_ratio));

    // Mat3 I = Mat3::identity(1.0f);
    // Mat3 uuT = Mat3::outer_product(u, u);
    // Mat3 H_local = uuT * E_double_prime + (I - uuT) * (E_prime / d);
    auto& H_local = c.H;

    // float3 Hp_sum = H_local * p_sum;

    for ( int i = 0; i < 4; ++i ) {
        int vi = c.v[i];
        if ( invMass[vi] > 0.f ) {
            // Hp_neg 存储的是 -Hp，所以乘上 (-w_i)
            // float3 delta_Hp_neg = Hp_sum * (-c.w[i]);
            // atomicAddFloat3(&Hp_neg[vi], delta_Hp_neg);
            float3 off_diag_p_sum = p_sum - c.w[i] * p[c.v[i]];
            float3 delta_Hp_off_diag = H_local * off_diag_p_sum * (-c.w[i]);
            atomicAddFloat3(&Hp_neg[vi], delta_Hp_off_diag);
        }
    }
}

void SolverPNCG::compute_collision_constraint(float3* x, float3* forces, bool change_hash) {
    START_TIMER;
    int block = 256;
    int cloth_vertex_size = params.nb_all_cloth_vertices;
    int n = cloth_vertex_size;
    float max_dist = params.cloth_edge_mean_length * 1.5f;
    if ( change_hash ) {
        tp_result_size_h = ee_result_size_h = 0;
        collision_collect_near_pairs(x, max_dist, true, false,
            true, true);
    }
    RECORD_TIME("collision_collect_near_pairs");


    debug_colors.assign(cloth_vertex_size, make_float3(0.5f, 0.5f, 0.5f));
    int num_constraints = tp_result_size_h + ee_result_size_h;
    if ( num_constraints > 0 ) {
        int all_vertex_size = params.nb_all_vertices;
        // collision_tp_to_normal_constraints<<<(tp_result_size_h + block - 1) / block, block>>>(
        //     normal_constraints.data().get(),
        //     debug_colors.data().get(),
        //     tp_collision_result.data().get(),
        //     constraint_colors.data().get(),
        //     triangle_indices.data().get(),
        //     mass_inv.data().get(),
        //     tp_result_size_h);
        // collision_ee_to_normal_constraints<<<(ee_result_size_h + block - 1) / block, block>>>(
        //     normal_constraints.data().get(),
        //     debug_colors.data().get(),
        //     ee_collision_result.data().get(),
        //     constraint_colors.data().get(),
        //     edges.data().get(),
        //     mass_inv.data().get(),
        //     tp_result_size_h,
        //     ee_result_size_h);
        compute_normal_constraint_IPC_energy<<<(num_constraints + block - 1) / block, block>>>(
            Jx_diag.data().get(),
            forces,
            normal_constraints.data().get(),
            x, mass_inv.data().get(), obj_data.data().get(),
            vertices_obj.data().get(),1.f,
            num_constraints);
    }
    // RECORD_TIME("collision_to_normal_constraints");
}
static __global__ void compute_p_next(
    float3* __restrict__ p,
    const float3* __restrict__ Pg_neg,
    float beta,
    int n // vertices size
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        p[i] = Pg_neg[i] + p[i] * beta;
    }
}
static __global__ void compute_y(
    float3* __restrict__ y,
    const float3* __restrict__ g_neg,
    const float3* __restrict__ g_neg_prev,
    int n // vertices size
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        y[i] = g_neg_prev[i] - g_neg[i];
    }
}
static __global__ void update_x(
    float3* __restrict__ x,
    const float3* __restrict__ p,
    float alpha,
    int n // vertices size
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        x[i] = x[i] + p[i] * alpha;
    }
}

static __global__ void Jx_mult_x_diag_kernel(
    float3* __restrict__ res,
    const Mat3* __restrict__ Jx_diag,
    const float3* __restrict__ x,
    int n // vertices size
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        res[i] = Jx_diag[i] * x[i];
    }
}
static __global__ void Jx_diag_inv_mult_x_neg_kernel(
    float3* __restrict__ res,
    const Mat3* __restrict__ Jx_diag,
    const float3* __restrict__ x,
    int n // vertices size
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        res[i] = Jx_diag[i].inverse() * -x[i];
        // printf("%f,%f,%f\n",res[i].x,res[i].y,res[i].z);
    }
}

static __global__ void Jx_mul_x_offdiag_kernel(
    float3* __restrict__ res,
    const Mat3* __restrict__ Jx,
    const Mat3* __restrict__ Jx_bend_cross,
    const float3* __restrict__ x,
    const int2* __restrict__ edges,
    const int2* __restrict__ edge_opposite_points,
    int n // edge size
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        auto [v0_i,v1_i] = edges[i];

        atomicAddFloat3(&res[v0_i], Jx[i] * x[v1_i]);
        atomicAddFloat3(&res[v1_i], Jx[i].transpose() * x[v0_i]);

        auto p_op = edge_opposite_points[i];
        if ( p_op.x != -1 && p_op.y != -1 ) {
            atomicAddFloat3(&res[p_op.x], Jx_bend_cross[i] * x[p_op.y]);
            atomicAddFloat3(&res[p_op.y], Jx_bend_cross[i].transpose() * x[p_op.x]);
        }
    }
}
void SolverPNCG::P_mult_x(
    float3* __restrict__ dst,
    const float3* __restrict__ src
) {
    int block = 256;
    int n = params.nb_all_cloth_vertices;
    Jx_diag_inv_mult_x_neg_kernel<<<(n + block - 1) / block, block>>>(
        dst, Jx_diag.data().get(), src, n);
}
void SolverPNCG::H_mult_x_neg(
    float3* __restrict__ dst,
    const float3* __restrict__ src
) {
    int block = 256;
    int n = params.nb_all_cloth_vertices;

    Jx_mult_x_diag_kernel<<<(n + block - 1) / block, block>>>(
        dst, Jx_diag.data().get(), src, n);

    n = params.nb_all_cloth_edges;
    Jx_mul_x_offdiag_kernel<<<(n + block - 1) / block, block>>>(
        dst, Jx.data().get(),
        Jx_bend_cross.data().get(),
        src, edges.data().get(),
        edge_opposite_points.data().get(),
        n);
    n = ee_result_size_h + tp_result_size_h;
    if ( n > 0 ) {
        compute_collision_Hp_neg<<<(n + block - 1) / block, block>>>(
            dst, src, normal_constraints.data().get(),
            mass_inv.data().get(), d_hat, n);
    }
}
struct length_functor {
    __host__ __device__
    float operator()(const float3& vec) const {
        return sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
    }
};
static float compute_max_length(float3* x, int n) {
    auto begin = thrust::device_pointer_cast(x);
    auto end = thrust::device_pointer_cast(x + n);
    return thrust::transform_reduce(
        begin, end,
        length_functor(),
        0.0f,
        thrust::maximum<float>()
        );
}
// X. Shen, R. Cai, M. Bi, and T. Lv, "Preconditioned Nonlinear Conjugate Gradient Method for Real-time Interior-point Hyperelasticity," 
void SolverPNCG::update(float dt) {
    this->dt = dt;
    int block = 256;
    int nb_cloth_vertices = params.nb_all_cloth_vertices;
    int n = params.nb_all_vertices;
    int blocksPerGrid = (n + block - 1) / block;

    float3* x_tilde = temp_vertices_f3.data().get();
    float3* x = vertices_world.data().get();
    update_begin<<<blocksPerGrid, block>>>(
        x, x_tilde,
        vertices_local.data().get(),
        velocities.data().get(),
        vertices_obj.data().get(),
        world_matrices.data().get(),
        vertices_mask.data().get(),
        // masses.data().get(),
        nb_cloth_vertices, dt, n);
    check_sewing(frame > 20);
    check_update_pick();
    cudaMemcpyAsync(vertices_old.data().get(), x,
        params.nb_all_vertices * sizeof(float3), cudaMemcpyDeviceToDevice);
    fill_inv_mass<<<(n + block - 1) / block, block>>>(
        mass_inv.data().get(),
        vertices_obj.data().get(),
        object_types.data().get(),
        masses.data().get(),
        vertices_mask.data().get(), n);
    float3* p = this->p.data().get();
    // float3* p_prev = this->p_prev.data().get();
    float3* g_neg = this->forces.data().get();
    float3* g_neg_prev = this->temp_vertices2_f3.data().get();
    float3* y = this->temp_vertices3_f3.data().get();
    float3* Py = this->velocities.data().get();
    float3* Pg_neg = Py;
    float3* Hp_neg = Py;

    n = nb_cloth_vertices;
    int max_iter = 40;
    int iter = 0;
    float beta = 0.f;
    float dE0{}, dE{};
    float eps = 0.001f;
    d_hat = params.cloth_edge_mean_length * 0.8f;
    // d_hat = 2e-3f;
    cudaMemsetAsync(p, 0, nb_cloth_vertices * sizeof(float3));
    for ( ; iter < max_iter; ++iter ) {
        compute_elastic_constraint(x, x_tilde, g_neg);
        compute_collision_constraint(x, g_neg, iter % 20 == 0);
        if ( iter > 0 ) {
            compute_y<<<(n + block - 1) / block, block>>>(y, g_neg, g_neg_prev, n);
            cudaMemcpyAsync(g_neg_prev, g_neg, nb_cloth_vertices * sizeof(float3),
                cudaMemcpyDeviceToDevice);
            P_mult_x(Py, y);
            float yPy = vector_field_dot(y, Py);
            float yp = vector_field_dot(y, p);
            float pg_prev = -vector_field_dot(p, g_neg);
            float gPy = -vector_field_dot(g_neg, Py);
            beta = (gPy - yPy * pg_prev / yp) / yp;
            // std::cout << "yp = " << yp << std::endl;
            // std::cout << "beta = " << beta << std::endl;
        }
        P_mult_x(Pg_neg, g_neg);
        compute_p_next<<<(n + block - 1) / block, block>>>(p, Pg_neg, beta, n);
        float gp = -vector_field_dot(p, g_neg);
        H_mult_x_neg(Hp_neg, p);
        float pHp_neg = vector_field_dot(p, Hp_neg);

        // std::cout << "pHp_neg = " << pHp_neg << std::endl;
        pHp_neg = min(pHp_neg, -1e-8f);
        float max_p_norm = compute_max_length(p, n);
        float alpha = min(d_hat/(2.f*(max_p_norm + 1e-8f)), gp/pHp_neg);
        // std::cout << "alpha = " << alpha << std::endl;
        alpha = max(1e-8f, alpha);
        update_x<<<(n + block - 1) / block, block>>>(x, p, alpha, n);
        dE = -alpha * gp + 0.5f * alpha * alpha * pHp_neg;

        if ( iter == 0 ) dE0 = dE;
        else if ( dE < dE0 * eps ) break;

        std::swap(g_neg, g_neg_prev);
    }
    std::cout << "iter = " << iter << ", dE=" << dE << std::endl;

    update_end<<<n + block - 1, block>>>(
        vertices_local.data().get(),
        velocities.data().get(),
        vertices_old.data().get(),
        x,
        vertices_mask.data().get(),
        vertex_proxy.data().get(),
        vertices_obj.data().get(),
        world_matrices_inv.data().get(),
        dt, n);
    reset_pick_mask();
    CUDA_CHECK(cudaDeviceSynchronize());
    frame++;
}
