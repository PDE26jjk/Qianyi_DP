#include "solver_linear.cuh"

#include "common/atomic_utils.cuh"

#include <cub/cub.cuh>
#include <thrust/transform_reduce.h>
#include <iostream>

#include "simulation/geometry.cuh"

// ---- 自定义迭代器：融合点积计算与归约 ----
struct DotProductIterator {
    // 满足 CUB 要求的迭代器类型定义
    using difference_type = std::ptrdiff_t;
    using value_type = float;
    using pointer = const float*;   // 不真正使用
    using reference = float;          // 按值返回
    using iterator_category = std::random_access_iterator_tag;

    const float3* ptr_a = nullptr;
    const float3* ptr_b = nullptr;
    int stride_a = 1;   // 步长（元素数），当前场景连续存放，均为1
    int stride_b = 1;

    __host__ __device__ __forceinline__
    reference operator*() const {
        return dot(*ptr_a, *ptr_b);
    }

    __host__ __device__ __forceinline__
    reference operator[](difference_type n) const {
        return dot(ptr_a[n * stride_a], ptr_b[n * stride_b]);
    }

    __host__ __device__ __forceinline__ DotProductIterator& operator++() {
        ptr_a += stride_a;
        ptr_b += stride_b;
        return *this;
    }
    __host__ __device__ __forceinline__ DotProductIterator operator++(int) {
        DotProductIterator tmp = *this;
        ++(*this);
        return tmp;
    }
    __host__ __device__ __forceinline__ DotProductIterator& operator--() {
        ptr_a -= stride_a;
        ptr_b -= stride_b;
        return *this;
    }
    __host__ __device__ __forceinline__ DotProductIterator operator--(int) {
        DotProductIterator tmp = *this;
        --(*this);
        return tmp;
    }
    __host__ __device__ __forceinline__ DotProductIterator& operator+=(difference_type n) {
        ptr_a += n * stride_a;
        ptr_b += n * stride_b;
        return *this;
    }
    __host__ __device__ __forceinline__ DotProductIterator& operator-=(difference_type n) {
        ptr_a -= n * stride_a;
        ptr_b -= n * stride_b;
        return *this;
    }
    __host__ __device__ __forceinline__ DotProductIterator operator+(difference_type n) const {
        return DotProductIterator{ ptr_a + n * stride_a, ptr_b + n * stride_b, stride_a, stride_b };
    }
    __host__ __device__ __forceinline__ DotProductIterator operator-(difference_type n) const {
        return DotProductIterator{ ptr_a - n * stride_a, ptr_b - n * stride_b, stride_a, stride_b };
    }
    __host__ __device__ __forceinline__ difference_type operator-(const DotProductIterator& other) const {
        return (ptr_a - other.ptr_a) / stride_a;
    }
    __host__ __device__ __forceinline__ bool operator==(const DotProductIterator& other) const {
        return ptr_a == other.ptr_a;
    }
    __host__ __device__ __forceinline__ bool operator!=(const DotProductIterator& other) const {
        return ptr_a != other.ptr_a;
    }
};
void LinearSolver::init(int diag_size,int edge_size) {
    m_edge_size = edge_size;
    m_diag_size = diag_size;
    Jx_nondiag.resize(edge_size);
    Jx_diag.resize(diag_size);
    M_inv.resize(diag_size);
    Jx_bend_cross.resize(edge_size);

    d_sum_result.resize(1);

    DotProductIterator iter{ nullptr, nullptr, 1, 1 };
    size_t temp_bytes = 0;
    cub::DeviceReduce::Sum(nullptr, temp_bytes, iter, d_sum_result.data().get(), diag_size);
    d_cub_temp.resize(temp_bytes);
}
// void SolverPCG::init(int diag_size, int edge_size,bool use_preconditioner_) {
//     use_preconditioner = use_preconditioner_;
//     LinearSolver::init(diag_size, edge_size);
//     Ad.resize(diag_size);
//     d.resize(diag_size);
//     r.resize(diag_size);
//     z.resize(diag_size);
// }

void LinearSolver::vector_field_dot(const float3* a, const float3* b, float* result) {
    int n = m_diag_size;

    DotProductIterator iter{ a, b, 1, 1 };

    size_t temp_bytes = d_cub_temp.size();
    cub::DeviceReduce::Sum(
        d_cub_temp.data().get(),
        temp_bytes,
        iter,
        result,
        n);

}

// struct DotProductFunctor {
//     __device__ __forceinline__
//     float operator()(const thrust::tuple<const float3&, const float3&>& t) const {
//         const float3& va = thrust::get<0>(t);
//         const float3& vb = thrust::get<1>(t);
//         return va.x * vb.x + va.y * vb.y + va.z * vb.z;
//     }
// };
//
float LinearSolver::vector_field_dot_sync(const float3* a, const float3* b) {
    // int n = m_diag_size;
    //
    // auto a_begin = thrust::device_pointer_cast(a);
    // auto a_end = thrust::device_pointer_cast(a + n);
    // auto b_begin = thrust::device_pointer_cast(b);
    //
    // return thrust::transform_reduce(
    //     thrust::make_zip_iterator(thrust::make_tuple(a_begin, b_begin)),
    //     thrust::make_zip_iterator(thrust::make_tuple(a_end, b_begin)),
    //     DotProductFunctor{},
    //     0.0f,
    //     thrust::plus<float>());

    vector_field_dot(a,b,d_sum_result.data().get());
    float result;
    cudaMemcpy(&result, d_sum_result.data().get(), sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}
__global__ void Jx_mult_x_diag_kernel(
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


__global__ void A_mul_x_offdiag_kernel(
    float3* __restrict__ res,
    const Mat3* __restrict__ Jx_nondiag,
    const Mat3* __restrict__ Jx_bend_cross,
    const float3* __restrict__ x,
    const int2* __restrict__ edges,
    const int2* __restrict__ edge_opposite_points,
    int n // edge size
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        auto [v0_i,v1_i] = edges[i];

        atomicAddFloat3(&res[v0_i], Jx_nondiag[i] * x[v1_i]);
        atomicAddFloat3(&res[v1_i], Jx_nondiag[i].transpose() * x[v0_i]);

        auto p_op = edge_opposite_points[i];
        if ( p_op.x != -1 && p_op.y != -1 ) {
            atomicAddFloat3(&res[p_op.x], Jx_bend_cross[i] * x[p_op.y]);
            atomicAddFloat3(&res[p_op.y], Jx_bend_cross[i].transpose() * x[p_op.x]);
        }
    }
}

void LinearSolver::A_mult_x(
    float3* __restrict__ dst,
    const float3* __restrict__ src
) {
    auto geo = simulator->get_geo();
    int threadsPerBlock = 256;
    int n = m_diag_size;

    Jx_mult_x_diag_kernel<<<(n + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
        dst,
        Jx_diag.data().get(),
        src,
        n);

    n = m_edge_size;
    A_mul_x_offdiag_kernel<<<(n + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
        dst,
        Jx_nondiag.data().get(),
        Jx_bend_cross.data().get(),
        src,
        geo->edges.data().get(),
        geo->edge_opposite_points.data().get(),
        n);

}



// __global__ void vector_field_dot_kernel(
//     float* __restrict__ res,
//     const float3* a,
//     const float3* b,
//     int n // size of res
// ) {
//     for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
//           i += blockDim.x * gridDim.x ) {
//         res[i] = dot(a[i], b[i]);
//     }
// }
//

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

template<bool UsePreprocessingDiag>
static __global__ void ite_kernel1(
    float3* __restrict__ r,
    float3* __restrict__ x,
    float3* __restrict__ z,
    const float3* __restrict__ d,
    const float3* __restrict__ Ad,
    const Mat3* __restrict__ M_inv,
    float alpha,
    int n // size of r
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        x[i] = x[i] + alpha * d[i];
        r[i] = r[i] - alpha * Ad[i];
        if constexpr ( UsePreprocessingDiag ) {
            z[i] = M_inv[i] * r[i];
        }
    }
}
static __global__ void ite_kernel2(
    float3* __restrict__ d,
    const float3* __restrict__ r,
    float beta,
    int n // size of r
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        d[i] = r[i] + beta * d[i];
    }
}

// Linear conjugate gradient
// template<bool UsePreprocessingDiag>
// void SolverPCG::solve_impl(float3* dx, const float3* rhs) {
//     int block = 256;
//
//     int n = m_diag_size;
//     float3* x = dx;
//     const float3* b = rhs;
//     float3* r = this->r.data().get();
//     // cudaMemcpyAsync(r, b, n * sizeof(float3), cudaMemcpyDeviceToDevice);
//     float3* z = this->z.data().get();
//     float3* d = this->d.data().get();
//     float3* Ad = this->Ad.data().get();
//     float3* Ax = Ad;
//     Mat3* M_inv = this->M_inv.data().get();
//
//     int blocksPerGrid = (n + block - 1) / block;
//
//     A_mult_x(Ax, x);
//     // r = b - A @ x; d = r || r = b - A @ x; z = M^{-1} @ r; d = z;
//     before_ite_kernel<UsePreprocessingDiag><<<blocksPerGrid, block>>>(
//         r, d, z, b, M_inv, Ax, n);
//
//     float delta_new = vector_field_dot_sync(r, UsePreprocessingDiag ? z : r);
//     int max_iter = 1000;
//     int iter = 0;
//     float alpha = -1.f;
//     float d_dot_Ad = -1.f;
//     for ( ; iter < max_iter; ++iter ) {
//         if ( (iter > 5 && delta_new < 1e-5f) || isnan(delta_new) )break;
//         float delta_old = delta_new;
//         // Ad = A @ d
//         A_mult_x(Ad, d);
//         d_dot_Ad = vector_field_dot_sync(d, Ad);
//         // if (d_dot_Ad < 1e-16) break;
//         // alpha = (r^T * r || z) / dot(d, Ad)
//         alpha = delta_old / d_dot_Ad;
//         if ( isinf(alpha) || isinf(d_dot_Ad) || isnan(alpha) ) {
//             std::cout << "error 1: a: " << alpha << ", d_dot_Ad=" << d_dot_Ad << " delta=" << delta_old << std::endl;
//             break;
//         }
//         // x^{i+1} = x^{i} + alpha * d
//         // r^{i+1} = r^{i} + alpha * Ad
//         // || z^{i+1} = M^{-1} @ r^{i+1}
//         ite_kernel1<UsePreprocessingDiag><<<blocksPerGrid, block>>>(
//             r, x, z, d, Ad, M_inv, alpha, n);
//
//         delta_new = vector_field_dot_sync(r, UsePreprocessingDiag ? z : r);
//         if ( isinf(delta_new) ) {
//             std::cout << "error 3: a: " << alpha << ", d_dot_Ad=" << d_dot_Ad << " delta=" << delta_new << std::endl;
//             break;
//         }
//         float beta = delta_new / delta_old;
//         // d^{i+1} = r^{i+1} + beta * d^{i} || d^{i+1} = z^{i+1} + beta * d^{i}
//         ite_kernel2<<<blocksPerGrid, block>>>(
//             d, UsePreprocessingDiag ? z : r, beta, n);
//     }
//     // if ( isnan(delta_new) ) {
//     //     std::cout << "error 2: a: " << alpha << ", d_dot_Ad=" << d_dot_Ad << " delta=" << delta_new << std::endl;
//     // }
//     std::cout << "PCG iter = " << iter << ", delta=" << delta_new << std::endl;
//
// }
//
// template void SolverPCG::solve_impl<true>(float3* dx, const float3* rhs);
// template void SolverPCG::solve_impl<false>(float3* dx, const float3* rhs);

void SolverJacobi::init(int diag_size, int edge_size) {
    LinearSolver::init(diag_size, edge_size);
    r.resize(diag_size);
    Ax.resize(diag_size);
}
__global__ void jacobi_update_kernel(
    float3* __restrict__ x,
    const float3* __restrict__ r,
    const Mat3* __restrict__ M_inv,
    int n) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        x[i] = x[i] + M_inv[i] * r[i];
    }
}
__global__ void residual_kernel(
    float3* __restrict__ r,
    const float3* __restrict__ b,
    const float3* __restrict__ Ax,
    int n) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        r[i] = b[i] - Ax[i];
    }
}

void SolverJacobi::solve(float3* dx, const float3* rhs,int max_iters) {
    int n = m_diag_size;
    int block = 256;
    int grid = (n + block - 1) / block;

    float3* x = dx;
    const float3* b = rhs;
    float3* r_ptr = r.data().get();
    float3* Ax_ptr = Ax.data().get();
    Mat3* M_inv_ptr = M_inv.data().get();

    const float tol = 1e-5f;
    int iter = 0;
    float r_dot_r = -1.f;
    for ( ; iter < max_iters; ++iter ) {
        // Ax = A * x
        A_mult_x(Ax_ptr, x);
        // r = b - Ax
        residual_kernel<<<grid, block>>>(r_ptr, b, Ax_ptr, n);
        
        if ( iter % 10 == 0 && iter > 0 ) {
            r_dot_r = vector_field_dot_sync(r_ptr, r_ptr);
            if ( r_dot_r < tol ) {
                 break;
            }
        }

        // x = x + M^{-1} * r
        jacobi_update_kernel<<<grid, block>>>(x, r_ptr, M_inv_ptr, n);
    }
    std::cout << "Jacobi iter = " << iter << ", delta=" << r_dot_r << std::endl;
}
