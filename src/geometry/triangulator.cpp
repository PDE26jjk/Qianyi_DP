#include "triangulator.h"
#include "mapbox/earcut.hpp"
#include <array>

std::vector<int> delaunay_2d_impl(const std::vector<float>& pointVecIn,
    const std::vector<int>& constraintVecIn);

namespace triangulator {

std::vector<int> earcut(const std::vector<float>& input) {
    using N = int;
    using Coord = double;
    using Point = std::array<Coord, 2>;
    std::vector<std::vector<Point>> polygon;
    std::vector<Point> points(input.size() / 2);
    for ( int i = 0; i < points.size(); i++ ) {
        points[i][0] = input[2 * i];
        points[i][1] = input[2 * i + 1];
    }
    polygon.push_back(points);
    return mapbox::earcut<N>(polygon);
}
std::vector<int> delaunay(const std::vector<float>& pointVecIn, const std::vector<int>& constraintVecIn) {
    return delaunay_2d_impl(pointVecIn, constraintVecIn);
}

}
