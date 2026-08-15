#include "solver_linear.cuh"

#include "common/atomic_utils.cuh"

#include <cub/cub.cuh>
#include <thrust/transform_reduce.h>
#include <iostream>

#include "simulation/geometry.cuh"


void SolverPCG::init(int diag_size, int edge_size,bool Jx_nondiag_identity_only, bool use_preconditioner_) {
    use_preconditioner = use_preconditioner_;
    LinearSolver::init(diag_size, edge_size, Jx_nondiag_identity_only);
    Ad.resize(diag_size);
    d.resize(diag_size);
    r.resize(diag_size);
    z.resize(diag_size);

    temp1.resize(6);

}
template<bool UsePreprocessingDiag>
static __global__ void before_ite_kernel(
    float3* __restrict__ r,
    float3* __restrict__ d,
    float3* __restrict__ z,
    const float3* __restrict__ b,
    const Mat3* __restrict__ M_inv,
    const float3* __restrict__ Ax,
    int n // size of r
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        r[i] = b[i] - Ax[i];
        if constexpr ( UsePreprocessingDiag ) {
            z[i] = M_inv[i] * r[i];
            d[i] = z[i];
        }
        else {
            d[i] = r[i];
        }
    }
}

static __global__ void compute_alpha_kernel(
    float* alpha,
    const float* delta_old,
    const float* d_dot_Ad) {
    if ( blockIdx.x == 0 && threadIdx.x == 0 ) {
        if ( *d_dot_Ad < 0.f ) {
            // printf("PCG indefiniteness: d_dot_Ad = %f!!!\n", *d_dot_Ad);
            *alpha = 0.f;
        }
        else
            *alpha = *delta_old / *d_dot_Ad;
    }
}

static __global__ void compute_beta_kernel(
    float* beta,
    const float* delta_new,
    const float* delta_old) {
    if ( blockIdx.x == 0 && threadIdx.x == 0 ) {
        *beta = *delta_new / *delta_old;
    }
}
template<bool UsePreprocessingDiag>
static __global__ void ite_kernel1(
    float3* __restrict__ r,
    float3* __restrict__ x,
    float3* __restrict__ z,
    const float3* __restrict__ d,
    const float3* __restrict__ Ad,
    const Mat3* __restrict__ M_inv,
    const float* alpha,
    int n) {
    float a = *alpha;
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        x[i] = x[i] + a * d[i];
        r[i] = r[i] - a * Ad[i];
        if constexpr ( UsePreprocessingDiag ) {
            z[i] = M_inv[i] * r[i];
        }
    }
}

static __global__ void ite_kernel2(
    float3* __restrict__ d,
    const float3* __restrict__ r,
    const float* beta,
    int n) {
    float b = *beta;
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        d[i] = r[i] + b * d[i];
    }
}
static __global__ void A_projection_kernel(
    Mat3* __restrict__ Jx,
    const int2* __restrict__ edges,
    const float3* __restrict__ r,
    const float* d_projection_limit,
    int n
) {
    float projection_limit = *d_projection_limit;
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        int2 edge = edges[i];
        if ( norm(r[edge.x]) > projection_limit || norm(r[edge.y]) > projection_limit ) {
            Jx[i] = Mat3::identity(1e-8f);
        }
    }
}
static __global__ void add_stiff_to_A_diag_kernel(
    Mat3* __restrict__ Jx,
    Mat3* __restrict__ M_inv,
    const float3* __restrict__ r,
    const float3* __restrict__ Ad,
    const float3* __restrict__ d,
    const float projection_limit,
    float extra_stiff,
    const char* mask,
    int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        // if ( norm(r[i]) > projection_limit ) {
        float dAd = dot(Ad[i], d[i]);
        float dAd_nearby = 0.f;
        if ( i + 1 < n ) {
            dAd_nearby += dot(Ad[i + 1], d[i + 1]);
        }
        if ( i - 1 >= 0 ) {
            dAd_nearby += dot(Ad[i - 1], d[i - 1]);
        }
        float factor = fabsf(dAd_nearby - dAd) / fabsf(dAd);
        if ( factor > 5 || factor < 0.2 || dAd < projection_limit ) {
            Jx[i].add_diag(extra_stiff);
            M_inv[i] = Jx[i].inverse();
        }

        // if (mask[i] || ((i+1 <n) && mask[i+1])|| ( (i-1 >=0) && mask[i-1])) {
        //     printf("mask: %d,r %f,dAd %f\n", (int)mask[i],norm(r[i]),dot(Ad[i],d[i]));
        // }
    }
}
// Attempt to make A positive determinism
void SolverPCG::try_make_A_PD() {
    auto geo = simulator->get_geo();
    int threadsPerBlock = 256;
    float* temp = this->temp1.data().get();
    float* d_dot_Ad_ptr = temp + 4;
    int n = m_edge_size;
    // A_projection_kernel<<<(n + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
    //     Jx_nondiag.data().get(),
    //     geo->edges.data().get(),
    //     r.data().get(),
    //     d_projection_limit,
    //     n);
    // TODO try something like 
    // Fernández-Fernández J A, Löschner F, Bender J. Progressively Projected Newton’s Method[M]. 2026-05-23. arXiv, 2026.

    n = m_diag_size;
    float3* d = this->d.data().get();
    float3* Ad = this->Ad.data().get();
    add_stiff_to_A_diag_kernel<<<(n + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
        Jx_diag.data().get(),
        M_inv.data().get(),
        r.data().get(),
        Ad, d,
        projection_limit,
        extra_stiff,
        geo->vertices_mask.data().get(),
        n);
}
void SolverPCG::print_debug(const float3* rhs) {
    thrust::host_vector<Mat3> Jx_diag_host = Jx_diag;
    thrust::host_vector<Mat3> Jx_nondiag_host = Jx_nondiag;
    auto geo = simulator->get_geo();
    thrust::host_vector<int2> edges_host = geo->edges;
    std::cout << "edges = [ ";
    for ( int i = 0; i < m_edge_size; ++i ) {
        std::cout << "(" << edges_host[i].x << ", " << edges_host[i].y << ")";
        if ( i < m_edge_size - 1 ) std::cout << ",";
        std::cout << " ";
    }
    std::cout << "]\n";
    std::cout << "Jx_diag = [ ";
    for ( int i = 0; i < m_diag_size; ++i ) {
        const Mat3& m = Jx_diag_host[i];
        std::cout << "[[" << m.r[0].x << ", " << m.r[0].y << ", " << m.r[0].z << "], "
            << "[" << m.r[1].x << ", " << m.r[1].y << ", " << m.r[1].z << "], "
            << "[" << m.r[2].x << ", " << m.r[2].y << ", " << m.r[2].z << "]]";
        if ( i < m_diag_size - 1 ) std::cout << ",";
        std::cout << " ";
    }
    std::cout << "]\n";
    std::cout << "Jx_nondiag = [ ";
    for ( int i = 0; i < m_edge_size; ++i ) {
        const Mat3& m = Jx_nondiag_host[i];
        std::cout << "[[" << m.r[0].x << ", " << m.r[0].y << ", " << m.r[0].z << "], "
            << "[" << m.r[1].x << ", " << m.r[1].y << ", " << m.r[1].z << "], "
            << "[" << m.r[2].x << ", " << m.r[2].y << ", " << m.r[2].z << "]]";
        if ( i < m_edge_size - 1 ) std::cout << ",";
        std::cout << " ";
    }
    std::cout << "]\n";
    std::cout << "rhs = [ ";
    std::vector<float3> h_rhs(m_diag_size); 
    cudaMemcpy(h_rhs.data(),rhs,m_diag_size * sizeof(float3),cudaMemcpyDeviceToHost);
    for ( int i = 0; i < m_diag_size; ++i ) {
        const float3& m = h_rhs[i];
        std::cout << "[" << m.x << ", " << m.y << ", " << m.z << "]" ;
        if ( i < m_diag_size - 1 ) std::cout << ",";
        std::cout << " ";
    }
    std::cout << "]" << std::endl;
}

// rhs should not be all zero
template<bool UsePreprocessingDiag>
void SolverPCG::solve_impl(float3* dx, const float3* rhs, int max_iters) {
    int block = 256;
    int n = m_diag_size;
    float3* x = dx;
    const float3* b = rhs;
    float3* r = this->r.data().get();
    float3* z = this->z.data().get();
    float3* d = this->d.data().get();
    float3* Ad = this->Ad.data().get();
    float3* Ax = Ad;

    Mat3* M_inv = this->M_inv.data().get();

    float* temp = this->temp1.data().get();
    float* d_alpha_ptr = temp;
    float* d_beta_ptr = temp + 1;
    float* d_delta_new_ptr = temp + 2;
    float* d_delta_old_ptr = temp + 3;
    float* d_dot_Ad_ptr = temp + 4;

    int blocksPerGrid = (n + block - 1) / block;

    A_mult_x(Ax, x);
    // r = b - A @ x; d = r || r = b - A @ x; z = M^{-1} @ r; d = z;
    before_ite_kernel<UsePreprocessingDiag> <<<blocksPerGrid, block >>>(
        r, d, z, b, M_inv, Ax, n);

    float delta_new_host;
    vector_field_dot(r, UsePreprocessingDiag ? z : r, d_delta_new_ptr);

    int iter = 0;
    const int check_interval = 10;
    int max_try = 10;
    for ( ; iter < max_iters; ++iter ) {
        cudaMemcpyAsync(d_delta_old_ptr, d_delta_new_ptr, sizeof(float),
            cudaMemcpyDeviceToDevice);
        // Ad = A @ d
        A_mult_x(Ad, d);
        if ( iter > 5 && iter % check_interval == 0 ) {
            cudaMemcpy(&delta_new_host, d_delta_new_ptr, sizeof(float), cudaMemcpyDeviceToHost);
            if ( delta_new_host < 1e-5f || isnan(delta_new_host) ) {
                break;
            }

            float d_dot_Ad_host;
            cudaMemcpy(&d_dot_Ad_host, d_dot_Ad_ptr, sizeof(float), cudaMemcpyDeviceToHost);
            if ( d_dot_Ad_host < 0.f || delta_new_host > 1e4 ) {
                throw std::exception("PCG indefiniteness");
                #if 0
                if ( max_try <= 0 ) {
                    // print_debug(rhs);
                    throw std::exception("PCG indefiniteness");
                }
                extra_stiff = 1.f;
                projection_limit = -1e2f;
                max_try -= 1;
                max_iters += 10;
                std::cout << "PCG add max_iters to: " << max_iters << " d_dot_Ad:" << d_dot_Ad_host << std::endl;
                for ( int j = 0; j < 100; j++ ) {
                    try_make_A_PD();
                    extra_stiff *= 4.f;
                    projection_limit *= 0.9f;
                    cudaMemset(x, 0, n * sizeof(float));
                    cudaMemset(Ad, 0, n * sizeof(float));
                    compute_alpha_kernel<<<1, 1>>>(d_alpha_ptr, d_delta_old_ptr, d_dot_Ad_ptr);
                    ite_kernel1<UsePreprocessingDiag> <<<blocksPerGrid, block >>>(
                        r, x, z, d, Ad, M_inv, d_alpha_ptr, n);
                    vector_field_dot(r, UsePreprocessingDiag ? z : r, d_delta_new_ptr);
                    compute_beta_kernel<<<1, 1>>>(d_beta_ptr, d_delta_new_ptr, d_delta_old_ptr);
                    ite_kernel2<<<blocksPerGrid, block>>>(
                        d, UsePreprocessingDiag ? z : r, d_beta_ptr, n);
                    A_mult_x(Ad, d);
                    float dAd = vector_field_dot_sync(d, Ad);
                    std::cout << "dAd: " << dAd << std::endl;
                    if ( dAd > 0.f ) break;
                
                }
                
                cudaMemset(x, 0, n * sizeof(float));
                cudaMemset(Ax, 0, n * sizeof(float));
                before_ite_kernel<UsePreprocessingDiag> <<<blocksPerGrid, block >>>(
                    r, d, z, b, M_inv, Ax, n);
                vector_field_dot(r, UsePreprocessingDiag ? z : r, d_delta_new_ptr);
                cudaMemcpyAsync(d_delta_old_ptr, d_delta_new_ptr, sizeof(float),
                    cudaMemcpyDeviceToDevice);
                A_mult_x(Ad, d);
                #endif
            }
        }

        vector_field_dot(d, Ad, d_dot_Ad_ptr);

        // alpha = (r^T * r || z) / dot(d, Ad)
        compute_alpha_kernel<<<1, 1>>>(d_alpha_ptr, d_delta_old_ptr, d_dot_Ad_ptr);

        // x^{i+1} = x^{i} + alpha * d
        // r^{i+1} = r^{i} + alpha * Ad
        // || z^{i+1} = M^{-1} @ r^{i+1}
        ite_kernel1<UsePreprocessingDiag> <<<blocksPerGrid, block >>>(
            r, x, z, d, Ad, M_inv, d_alpha_ptr, n);

        vector_field_dot(r, UsePreprocessingDiag ? z : r, d_delta_new_ptr);

        // beta = delta_new / delta_old
        compute_beta_kernel<<<1, 1>>>(d_beta_ptr, d_delta_new_ptr, d_delta_old_ptr);

        // d^{i+1} = r^{i+1} + beta * d^{i} || d^{i+1} = z^{i+1} + beta * d^{i}
        ite_kernel2<<<blocksPerGrid, block>>>(
            d, UsePreprocessingDiag ? z : r, d_beta_ptr, n);

    }

    cudaMemcpy(&delta_new_host, d_delta_new_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "PCG iter = " << iter << ", delta=" << delta_new_host << std::endl;

    if ( isnan(delta_new_host) ) {
        print_debug(rhs);
        std::cout << "PCG ended with NaN residual." << std::endl;
        throw std::exception("PCG nan");
    }
}
template void SolverPCG::solve_impl<true>(float3* dx, const float3* rhs, int max_iters);
template void SolverPCG::solve_impl<false>(float3* dx, const float3* rhs, int max_iters);
