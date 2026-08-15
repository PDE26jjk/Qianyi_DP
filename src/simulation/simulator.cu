// #include "solver_base.cuh"

#include <thrust/device_vector.h>
// #include <thrust/host_vector.h>
#include <map>
#include <thrust/sort.h>

#include "common/cuda_utils.h"

#include <thrust/execution_policy.h>

#include "solver_base.cuh"
#include "solver_explicit.cuh"
// #include "solver_PNCG.cuh"
#include "geometric_operator.cuh"
#include "geometry.cuh"
#include "solver_PDNewton.cuh"
#include "solver_VBD.cuh"
#include "solver_XPBD.cuh"
#include "contact/contact.cuh"

Simulator::Simulator() {}
void Simulator::reset() {
    try {
        if ( !cuda_device_initialized() ) return;
        if ( cuda_device_valid() ) {
            delete m_solver;
            delete m_geo;
            m_solver = nullptr;
            m_geo = nullptr;
            cudaDeviceSynchronize();
        }
        else {
            printf("[Qianyi] The CUDA context has been destroyed; skip releasing the solver. \n");
        }
    }
    catch ( ... ) {
        throw;
    }
    g_cuda_device_initialized = false;
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

// #include "common/vec_math.h"
void Simulator::init(const GeoDataInput& geo
) {
    if ( m_last_solver_name != m_solver_name || m_solver == nullptr ) {
        m_last_solver_name = m_solver_name;
        delete m_solver;
        create_solver();
    }
    if ( m_geo == nullptr ) {
        m_geo = new Geometry(this);
    }
    m_geo->init(geo);
    m_solver->init();

    dt = 0.01f;
    frame = -1;
    // Check errors
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Simulator::update(float h) {
    dt = h;
    frame += 1;
    m_geo->update_for_frame();
    m_geo->collision_detect();

    m_solver->begin_frame();
    // auto& edge_lookup = m_geo->edge_lookup;

    float dt_rest = h;
    float step_h = max(1e-20f, get_parameter("step_h", 0.001f));
    if ( step_h > h || step_h <= 0.0f ) step_h = h;
    int iters = (int)ceilf(h / step_h);
    step_h = h / (float)iters;
    for ( int substep = 0; substep < iters; substep++ ) {
        if ( substep > 10000 ) break;
        // step_h = step_h > dt_rest ? dt_rest : step_h;
        dt_rest -= step_h;
        float factor = clamp(1.f - (dt_rest / dt), 0., 1.f);

        m_geo->update_for_step(step_h, factor);
        m_solver->step(step_h);
    }
    m_geo->end_for_frame();
}

const SimulatorParams* Simulator::get_geo_params() const {
    return &m_geo->params;
}
Geometry* Simulator::get_geo() const {
    return m_geo;
}

void Simulator::copy_vertices(float* ptr, bool world_space) {
    return m_geo->copy_vertices(ptr, world_space);
}
void Simulator::copy_debug_colors(float* ptr) {
    return m_geo->copy_debug_colors(ptr);
}

int Simulator::add_pick_triangle(int mesh_index, int tri_index, float3 position) {
    return m_geo->add_pick_triangle(mesh_index, tri_index, position);
}
void Simulator::update_pick_triangle(int index, float3 position) {
    m_geo->update_pick_triangle(index, position);
}
void Simulator::remove_pick_triangle(int index) {
    m_geo->remove_pick_triangle(index);
}
void Simulator::clear_pick_triangle() {
    m_geo->clear_pick_triangle();
}

int Simulator::add_picker(float3 position) {
    return m_geo->add_picker(position);
}
void Simulator::update_picker(int index, float3 position) {
    m_geo->update_picker(index, position);
}
void Simulator::remove_picker(int index) {
    m_geo->remove_picker(index);
}
void Simulator::clear_picker() {
    m_geo->clear_picker();
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
    m_geo->upload_world_matrix(obj_index, matrix);
}
void Simulator::update_local_vertices(int obj_index, const std::vector<float>& vertices) {
    m_geo->upload_local_vertices(obj_index, vertices);
}

std::vector<std::string> Simulator::get_all_solver() {
    return { "Explicit", "PCG", "Chebyshev", "PNCG" };
}

void Simulator::set_solver(const std::string& string) {
    m_solver_name = string;
}
void Simulator::create_solver() {
    if ( m_solver_name == "Explicit" ) {
        m_solver = new SolverExplicit(this);
    }
    else if ( m_solver_name == "PDNewton" ) {
        m_solver = new SolverPDNewton(this);
    }
    else if ( m_solver_name == "XPBD" ) {
        m_solver = new SolverXPBD(this);
    }
    else if ( m_solver_name == "VBD" ) {
        m_solver = new SolverVBD(this);
    }
    // else if ( m_solver_name == "Chebyshev" ) {
    //     m_solver = new SolverChebyshev(this);
    // }
    // else if ( m_solver_name == "PNCG" ) {
    //     m_solver = new SolverPNCG(this);
    // }
    else {
        throw std::runtime_error("Unknown solver type: " + m_solver_name);
    }
}
CheckPointData Simulator::get_check_point_data(int index) const {
    assert(index < m_geo->params.nb_all_vertices);
    CheckPointData res;
    cudaMemcpy(&res.mass, m_geo->masses.data().get() + index, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&res.force, m_geo->forces.data().get() + index, sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(&res.force_elastic, m_geo->elastic_forces.data().get() + index, sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(&res.pos_world, m_geo->pos_world.data().get() + index, sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(&res.pos_prev, m_geo->pos_step_prev.data().get() + index, sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(&res.pos_pred, m_geo->pos_pred.data().get() + index, sizeof(float3), cudaMemcpyDeviceToHost);
    Contact& contact = m_geo->get_contact();
    std::vector<int> nearby_faces(broad_phase_size);
    cudaMemcpy(nearby_faces.data(),
        contact.broad_phase_vf.data().get() + index * broad_phase_size,
        sizeof(int) * broad_phase_size, cudaMemcpyDeviceToHost);
    if ( nearby_faces[0] > 0 ) {
        res.nearby_faces = std::vector<int>(nearby_faces.begin() + 1, nearby_faces.begin() + nearby_faces[0] + 1);
    }
    return res;
}
static __global__ void get_edge_by_points_index(
    int p0, int p1, int* res, const int2* __restrict__ dir_edges,
    const int2* __restrict__ edge_lookup
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx == 0 ) {
        int e = -1;
        find_edge(p0, p1, edge_lookup, dir_edges, e);
        res[0] = e;
    }
}
CheckEdgeData Simulator::get_check_edge_data(int p0, int p1) const {
    assert(p0 < m_geo->params.nb_all_vertices);
    assert(p1 < m_geo->params.nb_all_vertices);

    CheckEdgeData res;
    int* e;
    cudaMalloc(&e, sizeof(int));
    get_edge_by_points_index<<<1,1>>>(p0, p1, e, m_geo->dir_edges.data().get(), m_geo->edge_lookup.data().get());
    CUDA_CHECK(cudaDeviceSynchronize());
    int eid;
    cudaMemcpy(&eid, e, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(e);
    if ( eid < 0 ) throw std::runtime_error("Failed to get edge data");

    Contact& contact = m_geo->get_contact();
    std::vector<int> nearby_edges(broad_phase_size);
    cudaMemcpy(nearby_edges.data(),
        contact.broad_phase_ee.data().get() + eid * broad_phase_size,
        sizeof(int) * broad_phase_size, cudaMemcpyDeviceToHost);
    if ( nearby_edges[0] > 0 ) {
        res.nearby_edges = std::vector<int>(nearby_edges.begin() + 1, nearby_edges.begin() + nearby_edges[0] + 1);
    }
    std::vector<int> nearby_faces(broad_phase_size);
    cudaMemcpy(nearby_faces.data(),
        contact.broad_phase_ef.data().get() + eid * broad_phase_size,
        sizeof(int) * broad_phase_size, cudaMemcpyDeviceToHost);
    if ( nearby_faces[0] > 0 ) {
        res.nearby_faces = std::vector<int>(nearby_faces.begin() + 1, nearby_faces.begin() + nearby_faces[0] + 1);
    }
    cudaMemcpy(&res.normal, m_geo->edge_normals.data().get() + eid, sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(&res.tris, m_geo->e2t.data().get() + eid, sizeof(int2), cudaMemcpyDeviceToHost);
    return res;
}


CheckEdgeCollisionData Simulator::get_check_edge_collision_data(int p0, int p1) const {
    assert(p0 < m_geo->params.nb_all_vertices);
    assert(p1 < m_geo->params.nb_all_vertices);

    CheckEdgeCollisionData res;
    int* e;
    cudaMalloc(&e, sizeof(int));
    get_edge_by_points_index<<<1,1>>>(p0, p1, e, m_geo->dir_edges.data().get(), m_geo->edge_lookup.data().get());
    CUDA_CHECK(cudaDeviceSynchronize());
    int eid;
    cudaMemcpy(&eid, e, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(e);
    if ( eid < 0 ) throw std::runtime_error("Failed to get edge data");

    Contact& contact = m_geo->get_contact();
    contact.get_check_edge_collision_data(eid,res);

    return res;
}