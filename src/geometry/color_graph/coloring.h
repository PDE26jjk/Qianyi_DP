#pragma once
#include <vector>
#include <vector_types.h>
int graph_coloring(int num_nodes,const std::vector<int2>& edges, int algorithm, std::vector<int>& node_colors,bool balance=false, float target_max_min_ratio = 1.1f);


