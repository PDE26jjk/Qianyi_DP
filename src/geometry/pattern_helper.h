#pragma once

#include <vector>
class PatternHelper {
    PatternHelper();
    ~PatternHelper();

public:
    static PatternHelper& Instance();
    void update_edges(std::vector<float>& edge_points, std::vector<int>& loop_sizes, std::vector<float>& loop_transforms); // (N,2), (M,1), (M,16)
    void find_nearest_edge(float query_point[2], int& res_index, float& res_weight);
    bool check_edge_intersection(std::vector<float>& edge_points, int& res_index, float& res_weight);

private:
    struct Impl;
    Impl* impl;
};
