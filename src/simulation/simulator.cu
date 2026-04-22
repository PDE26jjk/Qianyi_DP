// #include "solver_base.cuh"

#include <thrust/device_vector.h>
// #include <thrust/host_vector.h>
#include <map>
#include <thrust/sort.h>

#include "common/cuda_utils.h"

#include <thrust/execution_policy.h>

#include "solver_base.cuh"
#include "solver_Chebyshev.cuh"
#include "solver_PCG.cuh"
#include "solver_PNCG.cuh"


Simulator::Simulator() {}
extern bool device_initialized;
extern int active_device_id;
void Simulator::reset() {
    try {
        if ( !device_initialized ) return;
        cudaError_t err = cudaSetDevice(active_device_id);
        if ( err == cudaSuccess ) {
            delete m_solver;
            m_solver = nullptr;
            cudaDeviceSynchronize();
        }
        else {
            printf("[Qianyi] The CUDA context has been destroyed; skip releasing the solver. \n");
        }
    }
    catch ( ... ) {
        throw;
    }
    device_initialized = false;
}
Simulator::~Simulator() {
    reset();
}
static void printLastCudaError(const char* context = nullptr) {
    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess ) {
        if ( context ) {
            std::cerr << "[" << context << "] ";
        }
        std::cerr << "Last CUDA Error: " << cudaGetErrorString(err)
            << " (code: " << err << ")" << std::endl;

        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        std::cerr << "Device: " << device << " (" << prop.name << ")" << std::endl;
    }
    else {
        if ( context ) {
            std::cout << "[" << context << "] No CUDA error" << std::endl;
        }
    }
}

template<typename T, typename otherT=T>
static void copy_to_device(const std::vector<T>& data, thrust::device_vector<otherT>& dst) {
    const size_t size = data.size() * sizeof(T) / sizeof(otherT);
    auto first = reinterpret_cast<const otherT*>(data.data());
    if ( size > 0 ) {
        dst.resize(size);
        CUDA_CHECK(cudaMemcpy(
            thrust::raw_pointer_cast(dst.data()),
            first,
            size * sizeof(otherT),
            cudaMemcpyHostToDevice
        ));
    }
}
// #include "common/vec_math.h"
void Simulator::init(const std::vector<float>& _vertices,
    const std::vector<float>& _vertices_sim,
    const std::vector<int>& _edges,
    const std::vector<int>& _triangles,
    const std::vector<float>& _normals,
    const std::vector<int>& _object_types,
    const std::vector<ObjectDataInput>& obj_data_input,
    const std::vector<Mat4>& _world_matrices,
    const std::vector<int>& _vertex_index_offsets,
    const std::vector<int>& _triangle_index_offsets,
    const std::vector<float>& pin_fixed,
    const std::vector<float>& pin_attached,
    const std::vector<SewingData>& _sewings,
    const std::vector<int2>& _stitches,
    int nb_all_cloth_v, int nb_all_cloth_e, int nb_all_cloth_f
) {
    if ( m_last_solver_name != m_solver_name || m_solver == nullptr ) {
        m_last_solver_name = m_solver_name;
        delete m_solver;
        create_solver();
    }
    auto& params = m_solver->params;
    params.nb_all_objects = (int)obj_data_input.size();
    params.nb_all_vertices = (int)_vertices.size() / 3;
    params.nb_all_edges = (int)_edges.size() / 2;
    params.nb_all_triangles = (int)_triangles.size() / 3;
    params.nb_all_cloth_triangles = nb_all_cloth_f / 3;
    params.nb_all_cloth_edges = nb_all_cloth_e / 2;
    params.nb_all_cloth_vertices = nb_all_cloth_v / 3;
    params.nb_all_stitches = (int)_stitches.size();

    cudaDeviceSynchronize();
    copy_to_device<float, float3>(_vertices, m_solver->vertices_2D);
    copy_to_device<float, float3>(_normals, m_solver->normals_input);
    copy_to_device<int, int2>(_edges, m_solver->edges);
    copy_to_device<int, int3>(_triangles, m_solver->triangles);
    copy_to_device(_object_types, m_solver->object_types);
    copy_to_device(obj_data_input, m_solver->obj_data);
    copy_to_device(_world_matrices, m_solver->world_matrices);
    copy_to_device(_sewings, m_solver->sewing_lines);
    copy_to_device(_stitches, m_solver->stitches);
    copy_to_device(_vertex_index_offsets, m_solver->vertex_index_offsets);
    copy_to_device(_triangle_index_offsets, m_solver->triangle_index_offsets);
    copy_to_device(pin_fixed, m_solver->pin_fixed);
    copy_to_device(pin_attached, m_solver->pin_attached);

    // vertices data for simulation,
    copy_to_device<float, float3>(_vertices_sim, m_solver->vertices_local);
    m_solver->init();

    // thrust::host_vector<int3> triangles = m_gpu->triangles;
    // std::vector triangles_(triangles.begin(), triangles.end());
    // thrust::host_vector<int3> indices = m_gpu->indices;
    // std::vector indices_(indices.begin(), indices.end());
    // thrust::host_vector<int2> e2t = m_gpu->e2t;
    // std::vector e2t_(e2t.begin(), e2t.end());
    // thrust::host_vector<Mat4> world_matrices = m_gpu->world_matrices;
    // std::vector world_matrices_(world_matrices.begin(), world_matrices.end());
    //
    // thrust::host_vector<Mat4> world_matrices_inv = m_gpu->world_matrices_inv;
    // std::vector world_matrices_inv_(world_matrices_inv.begin(), world_matrices_inv.end());
    CUDA_CHECK(cudaDeviceSynchronize());
    // printLastCudaError("init");
    return;
}

void Simulator::update(float h) {
    m_solver->update(h);
}

const SimulatorParams* Simulator::get_params() const {
    return &m_solver->params;
}

void Simulator::copy_vertices(float* ptr, bool world_space) {
    return m_solver->copy_vertices(ptr, world_space);
}
void Simulator::copy_debug_colors(float* ptr) {
    return m_solver->copy_debug_colors(ptr);
}

int Simulator::add_pick_triangle(int mesh_index, int tri_index, float3 position) {
    return m_solver->add_pick_triangle(mesh_index, tri_index, position);
}
void Simulator::update_pick_triangle(int index, float3 position) {
    m_solver->update_pick_triangle(index, position);
}
void Simulator::remove_pick_triangle(int index) {
    m_solver->remove_pick_triangle(index);
}
void Simulator::clear_pick_triangle() {
    m_solver->clear_pick_triangle();
}

int Simulator::add_picker(float3 position) {
    return m_solver->add_picker(position);
}
void Simulator::update_picker(int index, float3 position) {
    m_solver->update_picker(index, position);
}
void Simulator::remove_picker(int index) {
    m_solver->remove_picker(index);
}
void Simulator::clear_picker() {
    m_solver->clear_picker();
}
void Simulator::set_parameter(const std::string& key, float value) {
    m_parameters[key] = value;
}
float Simulator::get_parameter(const std::string& key, float default_value) const {
    auto it = m_parameters.find(key);
    if ( it != m_parameters.end() ) {
        return it->second;
    }
    return default_value;
}
void Simulator::update_world_matrix(int obj_index, const std::vector<float>& matrix) {
    m_solver->update_world_matrix(obj_index, matrix);
}
void Simulator::update_local_vertices(int obj_index, const std::vector<float>& vertices) {
    m_solver->update_local_vertices(obj_index, vertices);
}

std::vector<std::string> Simulator::get_all_solver() {
    return { "Explicit", "PCG", "Chebyshev", "PNCG" };
}

void Simulator::set_solver(const std::string& string) {
    m_solver_name = string;
}
void Simulator::create_solver() {
    if ( m_solver_name == "Explicit" ) {
        m_solver = new SolverBase(this);
    }
    else if ( m_solver_name == "PCG" ) {
        m_solver = new SolverPCG(this);
    }
    else if ( m_solver_name == "Chebyshev" ) {
        m_solver = new SolverChebyshev(this);
    }
    else if ( m_solver_name == "PNCG" ) {
        m_solver = new SolverPNCG(this);
    }
    else {
        throw std::runtime_error("Unknown solver type: " + m_solver_name);
    }
}
