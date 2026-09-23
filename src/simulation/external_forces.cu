#include "geometry.cuh"

#include "common/atomic_utils.cuh"
#include "common/cuda_utils.h"
#include "common/noise.cuh"

// ---------------------------------------------------------------------------
// External forces: constant normal pressure and wind.
//
// See the `external-forces` capability. Both loads are assembled into
// `external_forces` and consumed by the PDNewton inertia kernel as a
// right-hand-side term: neither contributes a Hessian block or a diagonal
// entry to the linear system.
//
// Pressure needs no volume state. For a uniform pressure the force is
// `-p * dV/dx`, and that gradient is the local area-weighted vertex normal, so
// the per-face force is exactly `p * A * n`. Only the second derivative of the
// volume would need the enclosed-volume machinery, and that term is not
// evaluated by design (see design.md for the criterion that would revisit it).
// ---------------------------------------------------------------------------

static __global__ void update_wind_field_kernel(
    const float3* __restrict__ pos_world,
    float3* __restrict__ wind_velocity,
    float3 base_wind,
    float gust_amount,
    float gust_frequency,
    float turbulence,
    float turbulence_scale,
    float turbulence_speed,
    float time,
    int num_vertices
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_vertices ) return;
    float3 w = base_wind;
    if ( gust_amount != 0.f ) {
        // Centred 1D noise so the wind breathes around its mean speed instead
        // of switching on and off.
        const float g = value_noise1(time * gust_frequency) * 2.f - 1.f;
        w = w * (1.f + gust_amount * g);
    }
    if ( turbulence != 0.f ) {
        // Advect the noise pattern downstream so the eddies travel with the
        // mean flow instead of standing still in world space.
        const float3 sample = (pos_world[i] - base_wind * (time * turbulence_speed))
            * turbulence_scale;
        w += curl_noise(sample, 1e-2f) * turbulence;
    }
    wind_velocity[i] = w;
}

static __global__ void accumulate_pressure_force_kernel(
    const float3* __restrict__ pos_world,
    const int3* __restrict__ tri_indices,
    const int* __restrict__ vertices_obj,
    const ObjectDataInput* __restrict__ obj_data,
    float3* __restrict__ external_forces,
    int num_tris
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_tris ) return;
    const int3 t = tri_indices[i];
    const float pressure = obj_data[vertices_obj[t.x]].pressure;
    if ( pressure == 0.f ) return;
    const float3 p0 = pos_world[t.x];
    const float3 p1 = pos_world[t.y];
    const float3 p2 = pos_world[t.z];
    // The load is p times the CURRENT face area, not the rest area: that is
    // what a pressure acting on a surface means, and it is the same
    // configuration the force gradient `dF = -p * d(dV/dx)` refers to. It also
    // keeps the closed-surface property that a uniform pressure has no net
    // force, at any deformed shape, which is exactly what inflates a shell.
    //
    // |cross(e1, e2)| is twice the area, so 0.5 * p * cross(e1, e2) is exactly
    // `p * area * n_hat` without a normalization and without any volume term.
    // Degenerate triangles fall out at zero.
    const float3 load = cross(p1 - p0, p2 - p0) * (0.5f * pressure);
    const float3 share = load * (1.f / 3.f);
    atomicAddFloat3(&external_forces[t.x], share);
    atomicAddFloat3(&external_forces[t.y], share);
    atomicAddFloat3(&external_forces[t.z], share);
}

static __global__ void accumulate_wind_force_kernel(
    const float3* __restrict__ pos_world,
    const float3* __restrict__ velocities,
    const float3* __restrict__ wind_velocity,
    const int3* __restrict__ tri_indices,
    const int* __restrict__ vertices_obj,
    const ObjectDataInput* __restrict__ obj_data,
    float3* __restrict__ external_forces,
    float air_density,
    float drag_coefficient,
    float lift_coefficient,
    int num_tris
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_tris ) return;
    const int3 t = tri_indices[i];
    const float3 e1 = pos_world[t.y] - pos_world[t.x];
    const float3 e2 = pos_world[t.z] - pos_world[t.x];
    const float3 n_raw = cross(e1, e2);
    const float two_area = norm(n_raw);
    if ( two_area <= 0.f ) return;
    const float3 n = n_raw / two_area;
    const float area = 0.5f * two_area;
    // Relative velocity of the air with respect to the surface, so a positive
    // normal projection means the air is pushing the surface downstream.
    const float3 v_face = (velocities[t.x] + velocities[t.y] + velocities[t.z]) * (1.f / 3.f);
    const float3 w_face = (wind_velocity[t.x] + wind_velocity[t.y] + wind_velocity[t.z])
        * (1.f / 3.f);
    const float3 v_rel = w_face - v_face;
    const ObjectDataInput object = obj_data[vertices_obj[t.x]];
    const float c_d = object.wind_drag >= 0.f ? object.wind_drag : drag_coefficient;
    const float c_l = object.wind_lift >= 0.f ? object.wind_lift : lift_coefficient;
    const float v_n = dot(v_rel, n);
    const float v_sq = dot(v_rel, v_rel);
    // Frozen, SIGGRAPH 2014: the separated lift/drag form
    //   F = 0.5 * rho * A * [ (C_D - C_L) (v.n) v + C_L |v|^2 n ].
    const float3 load = (v_rel * ((c_d - c_l) * v_n) + n * (c_l * v_sq))
        * (0.5f * air_density * area);
    const float3 share = load * (1.f / 3.f);
    atomicAddFloat3(&external_forces[t.x], share);
    atomicAddFloat3(&external_forces[t.y], share);
    atomicAddFloat3(&external_forces[t.z], share);
}

bool Geometry::has_wind() const {
    const float wx = get_global_parameter("wind_x", 0.f);
    const float wy = get_global_parameter("wind_y", 0.f);
    const float wz = get_global_parameter("wind_z", 0.f);
    const float turbulence = get_global_parameter("wind_turbulence", 0.f);
    return wx != 0.f || wy != 0.f || wz != 0.f || turbulence != 0.f;
}

void Geometry::update_wind_field() {
    const int n = params.nb_all_vertices;
    if ( n <= 0 ) return;
    float3* field = wind_velocity.data().get();
    if ( !has_wind() ) {
        // Keep the disabled path inert even when wind is turned off mid-run.
        cudaMemsetAsync(field, 0, n * sizeof(float3));
        return;
    }
    const float3 base_wind = make_float3(
        get_global_parameter("wind_x", 0.f),
        get_global_parameter("wind_y", 0.f),
        get_global_parameter("wind_z", 0.f));
    const float time = (float)simulator->frame * simulator->dt;
    const int block = 256;
    update_wind_field_kernel<<<(n + block - 1) / block, block>>>(
        pos_world.data().get(),
        field,
        base_wind,
        max(0.f, get_global_parameter("wind_gust", 0.f)),
        max(1e-6f, get_global_parameter("wind_gust_frequency", 0.5f)),
        max(0.f, get_global_parameter("wind_turbulence", 0.f)),
        max(1e-6f, get_global_parameter("wind_turbulence_scale", 0.5f)),
        max(0.f, get_global_parameter("wind_turbulence_speed", 1.f)),
        time, n);
}

void Geometry::accumulate_external_forces() {
    const int n = params.nb_all_vertices;
    if ( n <= 0 ) return;
    cudaMemsetAsync(external_forces.data().get(), 0, n * sizeof(float3));
    const int num_tris = params.nb_all_cloth_triangles;
    if ( num_tris <= 0 ) return;
    const bool wind_on = has_wind();
    if ( !external_forces_configured && !wind_on ) return;
    const int block = 256;
    const int grid = (num_tris + block - 1) / block;
    const int3* tri_indices = triangle_indices.data().get();
    const int* vertices_object = vertices_obj.data().get();
    const ObjectDataInput* objects = obj_data.data().get();
    if ( external_forces_configured ) {
        accumulate_pressure_force_kernel<<<grid, block>>>(
            pos_world.data().get(), tri_indices, vertices_object, objects,
            external_forces.data().get(), num_tris);
    }
    if ( wind_on ) {
        accumulate_wind_force_kernel<<<grid, block>>>(
            pos_world.data().get(),
            velocities.data().get(),
            wind_velocity.data().get(),
            tri_indices, vertices_object, objects,
            external_forces.data().get(),
            max(0.f, get_global_parameter("air_density", 1.225f)),
            get_global_parameter("wind_drag_coefficient", 1.f),
            get_global_parameter("wind_lift_coefficient", 0.f),
            num_tris);
    }
}
