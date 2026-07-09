#pragma once
#include "simulation/contact/lbvh.cuh"
#include <cuda_runtime.h>
#include <vector_types.h>

#include "solid_angle.cuh"


namespace sdf {

enum class QuantizationMode {
    FLOAT32,
    UINT16,
    UINT8
};
struct TextureSDFParams {
    float3 min_extents;
    float3 max_extents;
    float3 cell_size;       // 子网格中单个体素的物理尺寸
    int3 coarse_dims;       // 粗网格的维度 (W, H, D)
    int subgrid_size;       // 通常为 8
    int subgrid_tex_size;   // 3D纹理图集的整体尺寸
    
    float sdf_min_value;    // 用于量化反算
    float sdf_range;        
    QuantizationMode quantization_mode;

    thrust::device_vector<float> d_bg_sdf;
    thrust::device_vector<float> d_subgrid_texture;
    cudaArray_t coarse_array = nullptr;
    cudaArray_t subgrid_array = nullptr;
    
    // CUDA 硬件纹理对象
    cudaTextureObject_t coarse_texture = 0;  // 存储粗网格SDF值或Slot Index
    cudaTextureObject_t subgrid_texture = 0; // 存储高精度SDF数据的3D纹理图集
    thrust::device_vector<uint32_t> subgrid_start_slots;
};
struct MeshResult {
    thrust::device_vector<float3> positions; 
};
constexpr uint32_t SLOT_EMPTY = 0xFFFFFFFFu;
constexpr uint32_t SLOT_LINEAR = 0xFFFFFFFEu;
class SDF {
public:
    SDF();
    // ~SDF(); 

    void build_from_mesh(const int3* faces, const float3* pos, const lbvh3d::BVH3D& bvh, int num_vertices, int max_resolution, float target_voxel_size = 0.f, int
        subgrid_size = 8,bool use_parity=true,
        const lbvh3d::SolidAngleProps* solid_angle_props=nullptr);
    MeshResult compute_isomesh_from_texture_sdf(float isovalue);
    TextureSDFParams params;
    void release_textures();
};
void check_inside(const int3* faces, const float3* pos,
    const lbvh3d::BVH3D& bvh, const float3* query_points,
    const int query_points_size, float* res);
}
__host__ __device__ inline int idx3d(int x, int y, int z, int sx, int sy) {
    return (z * sy + y) * sx + x;
}
__device__ inline int3 id_to_xyz(int tid, int nx, int ny) {
    int z = tid / (nx * ny);
    int r = tid - z * nx * ny;
    int y = r / nx;
    int x = r - y * nx;
    return make_int3(x, y, z);
}

