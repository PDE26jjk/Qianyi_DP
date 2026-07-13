#pragma once
#include <vector>
#include <vector_types.h>
int graph_coloring_cpu(int num_nodes,const std::vector<int2>& edges, int algorithm,
    std::vector<int>& node_colors,bool balance=false, float target_max_min_ratio = 1.1f);

int graph_coloring_cuda(int num_nodes, int num_edges, int2* edges, int* node_colors,
    bool balance=false, float target_max_min_ratio = 1.1f); 