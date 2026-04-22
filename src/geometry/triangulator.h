#pragma once
#include <vector>

namespace triangulator {
std::vector<int> earcut(const std::vector<float>& input);
std::vector<int> delaunay(const std::vector<float>& pointVecIn,
                const std::vector<int>& constraintVecIn);
}
