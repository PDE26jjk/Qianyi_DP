#pragma once

#include <vector>

struct Params {
    float radius;
    float one_grid_length;
    int grid_size;
    int max_size;
    int n;
};

class Sampler {

public:
    Params params;

    // Device pointers
    unsigned char* d_grid_status = nullptr;
    int* d_grid_point = nullptr;
    float2* d_final = nullptr;
    int* d_grid_multi_point = nullptr; // flattened 3D array
    unsigned char* d_grid_multi_point_size = nullptr;
    float2* d_force = nullptr;
    unsigned char* d_valid_status = nullptr;

    int* d_nb_points = nullptr;
    float current_radius_scaled = FLT_MAX;

    // Inside/outside mask: loop edges plus per-fine-row edge buckets.
    float2* d_loop_edges = nullptr;
    int* d_row_counts = nullptr;
    int* d_row_offsets = nullptr;
    int* d_row_edges = nullptr;
    int* d_row_cursor = nullptr;
    int fine_rows = 0;
    float fine_length = 0.0f;
    static constexpr int fine_rows_per_cell = 4;

    // Helper device memory for inputs
    float2* d_input_points = nullptr;
    int2* d_edge_indices = nullptr;

    Sampler();
    ~Sampler();
    void set_radius(float _radius);

    void sample(
        std::vector<float2>& output_points,
        std::vector<int3>& output_tris,
        const std::vector<float2>& all_points,
        const std::vector<int2>& edge_indices, const std::vector<int>& curve_sizes, const std::vector<bool>& is_holes, float
        raw_radius, float f1, int t1, float f2, int t2, int triangulator = 0,
        float boundary_margin_cells = 0.8f
    );
};
