#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "common/cuda_utils.h"

#include <iostream>
#include <vector>
#include <cmath>
#include "sampler.h"
#include "common/atomic_utils.cuh"

__device__ inline float2 operator+(const float2& a, const float2& b) { return make_float2(a.x + b.x, a.y + b.y); }
__device__ inline float2 operator-(const float2& a, const float2& b) { return make_float2(a.x - b.x, a.y - b.y); }
__device__ inline float2 operator*(const float2& a, float b) { return make_float2(a.x * b, a.y * b); }
__device__ inline float dot(const float2& a, const float2& b) { return a.x * b.x + a.y * b.y; }
__device__ inline float2 normalize_safe(const float2& a) {
	float l = sqrtf(dot(a, a));
	if (l < 1e-8f) return make_float2(0, 0);
	return a * (1.0f / l);
}

// Taichi: grid_index
__device__ int2 grid_index(float2 p, float one_grid_length) {
	return make_int2((int)(p.x / one_grid_length), (int)(p.y / one_grid_length));
}
__device__ bool try_test(int2 index, unsigned char* grid_status, int max_size) {
	int idx = index.y * max_size + index.x;
	unsigned char old = atomicOr(&grid_status[idx], (unsigned char)(1 << 1));
	return old == unsigned char(0);
}

// ==========================================
// Kernels
// ==========================================

__global__ void k_init_rng(curandState* states, long seed, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		curand_init(seed, idx, 0, &states[idx]);
	}
}

__global__ void k_fill_boundary(
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
	if (i >= num_input_points) return;

	float2 point = points[i];
	int2 index = grid_index(point, p.one_grid_length);

	if (index.x < 0 || index.x >= p.max_size || index.y < 0 || index.y >= p.max_size) return;

	int idx_flat = index.y * p.max_size + index.x;

	if (grid_status[idx_flat] == 0) {
		if (try_test(index, grid_status, p.max_size)) {
			int j = atomicAdd(d_nb_points, 1);
			final_pts[j] = point;
			grid_point[idx_flat] = i;

		}
	}
	float2 next_p = points[next_point[i]];
	float2 mid_point = (point + next_p) * 0.5f;
	int2 mid_index = grid_index(mid_point, p.one_grid_length);

	if (mid_index.x >= 0 && mid_index.x < p.max_size && mid_index.y >= 0 && mid_index.y < p.max_size) {
		int mid_idx_flat = mid_index.y * p.max_size + mid_index.x;
		if (grid_status[mid_idx_flat] == 0) {
			if (try_test(mid_index, grid_status, p.max_size)) {
				int k = atomicAdd(d_nb_points, 1);
				final_pts[k] = mid_point;
				grid_point[mid_idx_flat] = i;
			}
		}
	}

}

__global__ void k_scan_line_fill(
	float2* final_pts,
	const float2* points,
	const int* next_point,
	int* grid_point,
	unsigned char* grid_status,
	int nb_current_points,
	Params p
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= nb_current_points) return;

	float2 point = final_pts[i];
	int2 index = grid_index(point, p.one_grid_length);

	if (index.x < 0 || index.x >= p.max_size || index.y < 0 || index.y >= p.max_size) return;

	int idx_flat = index.y * p.max_size + index.x;
	int grid_pt_idx = grid_point[idx_flat];

	float2 next_p = points[next_point[grid_pt_idx]];

	int d = (next_p.x - point.x > 0) ? 1 : -1;

	int x = index.x;
	int y = index.y;

	while (true) {
		y = y + d;
		if (y < 0 || y >= p.max_size) break;

		int curr_idx = y * p.max_size + x;
		if (grid_status[curr_idx] != 0) {
			break;
		}
		else {
			grid_status[curr_idx] = 4;
		}
	}
}

__global__ void k_generate_interior(
	unsigned char* grid_status,
	float2* final_pts,
	int* d_nb_points,
	curandState* states,
	Params p
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= p.grid_size || y >= p.grid_size) return;

	int idx = y * p.max_size + x; 

	if (grid_status[idx] == 4) {
		int tid = y * p.grid_size + x;
		curandState localState = states[tid];
		float r1 = curand_uniform(&localState);
		float r2 = curand_uniform(&localState);
		states[tid] = localState;

		float2 point = make_float2(r1, r2);
		float2 local_pos = (point + make_float2((float)x, (float)y)) * p.one_grid_length;

		int j = atomicAdd(d_nb_points, 1);
		final_pts[j] = local_pos;
	}
}

__global__ void k_reset_grid_multi(unsigned char* grid_multi_point_size, Params p) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < p.max_size && y < p.max_size) {
		grid_multi_point_size[y * p.max_size + x] = 0;
	}
}
constexpr int max_grid_particles_size = 4;

__global__ void k_build_grid(
	float2* final_pts,
	unsigned char* grid_multi_point_size,
	int* grid_multi_point,
	int nb_points,
	Params p
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= nb_points) return;

	float2 point = final_pts[i];
	int2 index = grid_index(point, p.one_grid_length);
	if (index.x < 0 || index.x >= p.max_size || index.y < 0 || index.y >= p.max_size) return;

	int idx_flat = index.y * p.max_size + index.x;

	unsigned char id = atomicAdd(&grid_multi_point_size[idx_flat], 1);
	if (id < max_grid_particles_size) {
		int offset = idx_flat * 4 + id;
		grid_multi_point[offset] = i;
	}
}

__global__ void k_compute_repulsion(
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
	if (actual_i >= nb_points) return;

	float2 point = final_pts[actual_i];
	int2 index = grid_index(point, p.one_grid_length);
	float r2 = p.radius * p.radius;
	float2 d = make_float2(0.0f, 0.0f);

	int x = index.x;
	int y = index.y;

	for (int nx = max(0, x - 1); nx < min(p.max_size, x + 2); ++nx) {
		for (int ny = max(0, y - 1); ny < min(p.max_size, y + 2); ++ny) {
			int n_idx = ny * p.max_size + nx;
			int count = min((int)grid_multi_point_size[n_idx], 4);

			for (int k = 0; k < count; ++k) {
				int p2_i = grid_multi_point[n_idx * 4 + k];
				if (p2_i == actual_i) continue;

				float2 p2 = final_pts[p2_i];
				float2 l = point - p2;
				float l2 = dot(l, l);

				if (l2 < (r2 * 0.5f) + 1e-6f) {
					if (l2 > 1e-10f) {
						// Taichi: d += l.normalized() / l2
						d = d + (l * (1.0f / (sqrtf(l2) * l2)));
					}
				}
			}
		}
	}
	force[actual_i] = d;
}

__global__ void k_apply_force(
	float2* final_pts,
	float2* force,
	int nb_boundary_points,
	int nb_points,
	float factor,
	Params p
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int actual_i = i + nb_boundary_points;
	if (actual_i >= nb_points) return;

	float2 d = force[actual_i];
	float maxd = p.radius * factor;
	float d_len_sq = dot(d, d);

	if (d_len_sq > maxd * maxd) {
		d = normalize_safe(d) * maxd;
	}
	final_pts[actual_i] = final_pts[actual_i] + d;
}

__global__ void k_validate(
	float2* final_pts,
	unsigned char* grid_status,
	unsigned char* valid_status,
	int nb_boundary_points,
	int nb_points,
	Params p
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int actual_i = i + nb_boundary_points;
	if (actual_i >= nb_points) return;

	int2 index = grid_index(final_pts[actual_i], p.one_grid_length);
	if (index.x >= 0 && index.x < p.max_size && index.y >= 0 && index.y < p.max_size) {
		if (grid_status[index.y * p.max_size + index.x] == 4) {
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

// ==========================================
// Host Controller Class
// ==========================================



Sampler::Sampler() {
	cudaMalloc(&d_nb_points, sizeof(int));
}

Sampler::~Sampler() {
	cudaFree(d_grid_status);
	cudaFree(d_grid_point);
	cudaFree(d_final);
	cudaFree(d_grid_multi_point);
	cudaFree(d_grid_multi_point_size);
	cudaFree(d_force);
	cudaFree(d_valid_status);
	cudaFree(d_nb_points);
	cudaFree(d_rng_states);
	if (d_input_points) cudaFree(d_input_points);
	if (d_input_next) cudaFree(d_input_next);
}

void Sampler::set_radius(float _radius) {
	params.radius = _radius;
	params.one_grid_length = _radius / sqrtf(2.0f);
	params.grid_size = (int)ceil(1.0f / params.one_grid_length);
	params.max_size = ((params.grid_size + 63) / 64) * 64;
	params.n = params.max_size * params.max_size; // Note: Taichi n = max_size^2, usually plenty buffer

	// Re-allocate if size changed (simplified: always re-allocate for this example)
	if (d_grid_status) cudaFree(d_grid_status);
	if (d_grid_point) cudaFree(d_grid_point);
	if (d_final) cudaFree(d_final);
	if (d_grid_multi_point) cudaFree(d_grid_multi_point);
	if (d_grid_multi_point_size) cudaFree(d_grid_multi_point_size);
	if (d_force) cudaFree(d_force);
	if (d_valid_status) cudaFree(d_valid_status);
	if (d_rng_states) cudaFree(d_rng_states);

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
	k_init_rng << <numBlocks, blockSize >> >(d_rng_states, 1234, params.n);
	current_radius_scaled = _radius;
}

std::vector<float2> Sampler::sample(
	const std::vector<float2>& boundary_points,
	const std::vector<int>& next_point,
	float raw_radius,
	float f1, int t1, float f2, int t2
) {
	if (boundary_points.empty()) return {};

	float x_min = FLT_MAX,y_min = FLT_MAX;
	float x_max = -FLT_MAX,y_max = -FLT_MAX;

	for (const auto& p : boundary_points) {
		if (p.x < x_min) x_min = p.x;
		if (p.y < y_min) y_min = p.y;
		if (p.x > x_max) x_max = p.x;
		if (p.y > y_max) y_max = p.y;
	}

	float width = x_max - x_min;
	float height = y_max - y_min;

	// Python: scale = 1.0 / (max(width, height) + radius)
	float scale = 1.0f / (fmaxf(width, height) + raw_radius);
	float2 offset = make_float2(x_min, y_min);

	// Python: radius_scaled = radius * scale
	float radius_scaled = raw_radius * scale;

	if (current_radius_scaled > radius_scaled) {
		set_radius(radius_scaled);
	}
	else {
		set_radius(radius_scaled);
	}

	// Python: points_normalized = (boundary_points - offset + radius * 0.5) * scale
	std::vector<float2> points_normalized(boundary_points.size());
	for (size_t i = 0; i < boundary_points.size(); ++i) {
		float2 p = boundary_points[i];
		points_normalized[i].x = (p.x - offset.x + raw_radius * 0.5f) * scale;
		points_normalized[i].y = (p.y - offset.y + raw_radius * 0.5f) * scale;
	}

	// ---------------------------------------------------------
	// 2. GPU Processing
	// ---------------------------------------------------------

	int num_pts = points_normalized.size();
	// 1. Setup Input
	if (d_input_points) cudaFree(d_input_points);
	if (d_input_next) cudaFree(d_input_next);
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
	k_fill_boundary << <numBlocks, blockSize >> >(
		d_input_points, d_input_next, num_pts,
		d_final, d_grid_status, d_grid_point, d_nb_points, params
	);
	cudaDeviceSynchronize();

	// Get nb_points
	int h_nb_points = 0;
	cudaMemcpy(&h_nb_points, d_nb_points, sizeof(int), cudaMemcpyDeviceToHost);

	// 4. Scan Line Fill (Kernel 2)
	numBlocks = (h_nb_points + blockSize - 1) / blockSize;
	k_scan_line_fill << <numBlocks, blockSize >> >(
		d_final, d_input_points, d_input_next,
		d_grid_point, d_grid_status, h_nb_points, params
	);
	cudaDeviceSynchronize();

	int nb_boundary_points = h_nb_points; //

	// 5. Generate Interior (Kernel 3)
	dim3 dimBlock(16, 16);
	dim3 dimGrid((params.grid_size + dimBlock.x - 1) / dimBlock.x,
		(params.grid_size + dimBlock.y - 1) / dimBlock.y);

	k_generate_interior << <dimGrid, dimBlock >> >(
		d_grid_status, d_final, d_nb_points, d_rng_states, params
	);
	cudaDeviceSynchronize();

	// Get updated nb_points
	cudaMemcpy(&h_nb_points, d_nb_points, sizeof(int), cudaMemcpyDeviceToHost);

	// 6. Repulsion Loops
	auto run_repulsion = [&](float factor, int times)
	{
		for (int t = 0; t < times; ++t) {
			// Reset Grid Multi
			dim3 gridDimMax((params.max_size + 15) / 16, (params.max_size + 15) / 16);
			k_reset_grid_multi << <gridDimMax, dimBlock >> >(d_grid_multi_point_size, params);

			// Build Grid
			int blocks = (h_nb_points + 255) / 256;
			k_build_grid << <blocks, 256 >> >(
				d_final, d_grid_multi_point_size, d_grid_multi_point, h_nb_points, params
			);

			// Compute Force (Only for non-boundary points)
			int inner_count = h_nb_points - nb_boundary_points;
			if (inner_count > 0) {
				blocks = (inner_count + 255) / 256;
				k_compute_repulsion << <blocks, 256 >> >(
					d_final, d_grid_multi_point_size, d_grid_multi_point, d_force,
					nb_boundary_points, h_nb_points, params
				);

				// Apply Force
				k_apply_force << <blocks, 256 >> >(
					d_final, d_force, nb_boundary_points, h_nb_points, factor, params
				);
			}
			cudaDeviceSynchronize();
		}
	};

	run_repulsion(f1, t1);
	run_repulsion(f2, t2);

	// 7. Validation
	int inner_count = h_nb_points - nb_boundary_points;
	if (inner_count > 0) {
		int blocks = (inner_count + 255) / 256;
		k_validate << <blocks, 256 >> >(
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

	std::vector<float2> output;
	output.reserve(inner_count);

	// Python: final_points / scale - radius * 0.5 + offset
	for (int i = nb_boundary_points; i < h_nb_points; ++i) {
		if (result_valid[i] > 0) {
			float2 p_norm = result_final[i];
			float2 p_world;
			p_world.x = p_norm.x / scale - raw_radius * 0.5f + offset.x;
			p_world.y = p_norm.y / scale - raw_radius * 0.5f + offset.y;
			output.push_back(p_world);
		}
	}
	return output;
}


// ==========================================
// Main / Interface
// ==========================================

std::vector<float> test(std::vector<float>& boundary, std::vector<int>& next_pt, float radius) {
	//float radius = 0.05f;
	Sampler sampler{};

	//std::vector<float2> boundary = {
	//    {0.1f, 0.1f}, {0.9f, 0.1f}, {0.9f, 0.9f}, {0.1f, 0.9f}
	//};
	//std::vector<int> next_pt = { 1, 2, 3, 0 };
	std::vector<float2> boundary_(boundary.size() / 2);
	std::memcpy(boundary_.data(), boundary.data(), sizeof(float) * boundary.size());

	std::vector<float2> points = sampler.sample(boundary_, next_pt, radius, 0.02f, 15, 0.01f, 35);
	//std::cout << "Generated " << points.size() << " points." << std::endl;
	//for (size_t i = 0; i < std::min(points.size(), (size_t)10); i++) {
	//    std::cout << points[i].x << ", " << points[i].y << std::endl;
	//}
	int size = sampler.params.max_size * sampler.params.max_size;
	std::vector<unsigned char> res(size);
	cudaMemcpy(res.data(), sampler.d_grid_status, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	std::vector<float> points_(points.size() * 2);
	std::memcpy(points_.data(), points.data(), sizeof(float) * points_.size());
	return points_;
}