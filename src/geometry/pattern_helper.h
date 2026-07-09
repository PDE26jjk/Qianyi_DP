#pragma once

#include <vector>

#include "lbvh_2d.h"
class PatternHelper {
    PatternHelper();
    ~PatternHelper();

public:
    static PatternHelper& Instance();
    void update_edges(std::vector<float>& edge_points, std::vector<int>& loop_sizes, std::vector<float>& loop_transforms); // (N,2), (M,1), (M,16)
    void find_nearest_edge(float query_point[2], int& res_index, float& res_weight);
    bool check_edge_intersection(std::vector<float>& edge_points, int& res_index, float& res_weight);
    
    std::vector<lbvh2d::FullIntersectionResult> get_all_intersections(std::vector<float>& edge_points,
        std::vector<int>& curve_sizes,
        std::vector<int8_t>& is_loops, std::vector<int>& num_sections_per_curve, std::vector<int>& section_point_sizes);
    void deduplicate_points(std::vector<float>& point_data, float threshold, std::vector<float>& out_point_data,
        std::vector<unsigned int>& out_map);

private:
    struct Impl;
    Impl* impl;
};
