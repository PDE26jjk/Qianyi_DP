#include "sdf.cuh"

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/system/detail/generic/remove.inl>

#include "common/geometric_algorithms.h"
#include "common/vec_math.h"
// Cube corner offsets relative to the cube origin (bottom-left-front corner)
// Order: z=0 face (corners 0-3), z=1 face (corners 4-7)
static const int MC_CUBE_CORNER_OFFSETS[8][3] = {
    { 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 }, // z=0 face (corners 0-3)
    { 0, 0, 1 }, { 1, 0, 1 }, { 1, 1, 1 }, { 0, 1, 1 }  // z=1 face (corners 4-7)
};

// For each of the 12 edges, the pair of corner indices it connects.
// Edges 0-3: bottom face (z=0), edges 4-7: top face (z=1),
// edges 8-11: vertical edges connecting corner i to corner i+4.
static const int MC_EDGE_TO_CORNERS[12][2] = {
    { 0, 1 }, { 1, 2 }, { 3, 2 }, { 0, 3 }, { 4, 5 }, { 5, 6 },
    { 7, 6 }, { 4, 7 }, { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }
};
static const uint32_t MC_CASE_TO_TRI_RANGE[257] = {
    0, 0, 3, 6, 12, 15, 21, 27, 36, 39, 45, 51, 60, 66, 75, 84, 90, 93, 99, 105, 114,
    120, 129, 138, 150, 156, 165, 174, 186, 195, 207, 219, 228, 231, 237, 243, 252,
    258, 267, 276, 288, 294, 303, 312, 324, 333, 345, 357, 366, 372, 381, 390, 396,
    405, 417, 429, 438, 447, 459, 471, 480, 492, 507, 522, 528, 531, 537, 543, 552,
    558, 567, 576, 588, 594, 603, 612, 624, 633, 645, 657, 666, 672, 681, 690, 702,
    711, 723, 735, 750, 759, 771, 783, 798, 810, 825, 840, 852, 858, 867, 876, 888,
    897, 909, 915, 924, 933, 945, 957, 972, 984, 999, 1008, 1014, 1023, 1035, 1047,
    1056, 1068, 1083, 1092, 1098, 1110, 1125, 1140, 1152, 1167, 1173, 1185, 1188, 1191,
    1197, 1203, 1212, 1218, 1227, 1236, 1248, 1254, 1263, 1272, 1284, 1293, 1305, 1317,
    1326, 1332, 1341, 1350, 1362, 1371, 1383, 1395, 1410, 1419, 1425, 1437, 1446, 1458,
    1467, 1482, 1488, 1494, 1503, 1512, 1524, 1533, 1545, 1557, 1572, 1581, 1593, 1605,
    1620, 1632, 1647, 1662, 1674, 1683, 1695, 1707, 1716, 1728, 1743, 1758, 1770, 1782,
    1791, 1806, 1812, 1827, 1839, 1845, 1848, 1854, 1863, 1872, 1884, 1893, 1905, 1917,
    1932, 1941, 1953, 1965, 1980, 1986, 1995, 2004, 2010, 2019, 2031, 2043, 2058, 2070,
    2085, 2100, 2106, 2118, 2127, 2142, 2154, 2163, 2169, 2181, 2184, 2193, 2205, 2217,
    2232, 2244, 2259, 2268, 2280, 2292, 2307, 2322, 2328, 2337, 2349, 2355, 2358, 2364,
    2373, 2382, 2388, 2397, 2409, 2415, 2418, 2427, 2433, 2445, 2448, 2454, 2457, 2460,
    2460,
};

static const int MC_TRI_LOCAL_INDICES[2460] = {
    0, 8, 3, 0, 1, 9, 1, 8, 3, 9, 8, 1, 1, 2, 10, 0, 8, 3, 1, 2, 10, 9, 2, 10, 0, 2, 9, 2, 8, 3, 2,
    10, 8, 10, 9, 8, 3, 11, 2, 0, 11, 2, 8, 11, 0, 1, 9, 0, 2, 3, 11, 1, 11, 2, 1, 9, 11, 9, 8, 11, 3,
    10, 1, 11, 10, 3, 0, 10, 1, 0, 8, 10, 8, 11, 10, 3, 9, 0, 3, 11, 9, 11, 10, 9, 9, 8, 10, 10, 8, 11, 4,
    7, 8, 4, 3, 0, 7, 3, 4, 0, 1, 9, 8, 4, 7, 4, 1, 9, 4, 7, 1, 7, 3, 1, 1, 2, 10, 8, 4, 7, 3,
    4, 7, 3, 0, 4, 1, 2, 10, 9, 2, 10, 9, 0, 2, 8, 4, 7, 2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, 8,
    4, 7, 3, 11, 2, 11, 4, 7, 11, 2, 4, 2, 0, 4, 9, 0, 1, 8, 4, 7, 2, 3, 11, 4, 7, 11, 9, 4, 11, 9,
    11, 2, 9, 2, 1, 3, 10, 1, 3, 11, 10, 7, 8, 4, 1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, 4, 7, 8, 9,
    0, 11, 9, 11, 10, 11, 0, 3, 4, 7, 11, 4, 11, 9, 9, 11, 10, 9, 5, 4, 9, 5, 4, 0, 8, 3, 0, 5, 4, 1,
    5, 0, 8, 5, 4, 8, 3, 5, 3, 1, 5, 1, 2, 10, 9, 5, 4, 3, 0, 8, 1, 2, 10, 4, 9, 5, 5, 2, 10, 5,
    4, 2, 4, 0, 2, 2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, 9, 5, 4, 2, 3, 11, 0, 11, 2, 0, 8, 11, 4,
    9, 5, 0, 5, 4, 0, 1, 5, 2, 3, 11, 2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, 10, 3, 11, 10, 1, 3, 9,
    5, 4, 4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, 5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, 5, 4, 8, 5,
    8, 10, 10, 8, 11, 9, 7, 8, 5, 7, 9, 9, 3, 0, 9, 5, 3, 5, 7, 3, 0, 7, 8, 0, 1, 7, 1, 5, 7, 1,
    5, 3, 3, 5, 7, 9, 7, 8, 9, 5, 7, 10, 1, 2, 10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, 8, 0, 2, 8,
    2, 5, 8, 5, 7, 10, 5, 2, 2, 10, 5, 2, 5, 3, 3, 5, 7, 7, 9, 5, 7, 8, 9, 3, 11, 2, 9, 5, 7, 9,
    7, 2, 9, 2, 0, 2, 7, 11, 2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, 11, 2, 1, 11, 1, 7, 7, 1, 5, 9,
    5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, 5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, 11, 10, 0, 11,
    0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, 11, 10, 5, 7, 11, 5, 10, 6, 5, 0, 8, 3, 5, 10, 6, 9, 0, 1, 5,
    10, 6, 1, 8, 3, 1, 9, 8, 5, 10, 6, 1, 6, 5, 2, 6, 1, 1, 6, 5, 1, 2, 6, 3, 0, 8, 9, 6, 5, 9,
    0, 6, 0, 2, 6, 5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, 2, 3, 11, 10, 6, 5, 11, 0, 8, 11, 2, 0, 10,
    6, 5, 0, 1, 9, 2, 3, 11, 5, 10, 6, 5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, 6, 3, 11, 6, 5, 3, 5,
    1, 3, 0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, 3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, 6, 5, 9, 6,
    9, 11, 11, 9, 8, 5, 10, 6, 4, 7, 8, 4, 3, 0, 4, 7, 3, 6, 5, 10, 1, 9, 0, 5, 10, 6, 8, 4, 7, 10,
    6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, 6, 1, 2, 6, 5, 1, 4, 7, 8, 1, 2, 5, 5, 2, 6, 3, 0, 4, 3,
    4, 7, 8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, 7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, 3,
    11, 2, 7, 8, 4, 10, 6, 5, 5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, 0, 1, 9, 4, 7, 8, 2, 3, 11, 5,
    10, 6, 9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, 8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, 5,
    1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, 0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, 6,
    5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, 10, 4, 9, 6, 4, 10, 4, 10, 6, 4, 9, 10, 0, 8, 3, 10, 0, 1, 10,
    6, 0, 6, 4, 0, 8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, 1, 4, 9, 1, 2, 4, 2, 6, 4, 3, 0, 8, 1,
    2, 9, 2, 4, 9, 2, 6, 4, 0, 2, 4, 4, 2, 6, 8, 3, 2, 8, 2, 4, 4, 2, 6, 10, 4, 9, 10, 6, 4, 11,
    2, 3, 0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, 3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, 6, 4, 1, 6,
    1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, 9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, 8, 11, 1, 8, 1, 0, 11,
    6, 1, 9, 1, 4, 6, 4, 1, 3, 11, 6, 3, 6, 0, 0, 6, 4, 6, 4, 8, 11, 6, 8, 7, 10, 6, 7, 8, 10, 8,
    9, 10, 0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, 10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, 10, 6, 7, 10,
    7, 1, 1, 7, 3, 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7,
    3, 9, 7, 8, 0, 7, 0, 6, 6, 0, 2, 7, 3, 2, 6, 7, 2, 2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, 2,
    0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, 1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, 11,
    2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, 8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, 0, 9, 1, 11,
    6, 7, 7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, 7, 11, 6, 7, 6, 11, 3, 0, 8, 11, 7, 6, 0, 1, 9, 11,
    7, 6, 8, 1, 9, 8, 3, 1, 11, 7, 6, 10, 1, 2, 6, 11, 7, 1, 2, 10, 3, 0, 8, 6, 11, 7, 2, 9, 0, 2,
    10, 9, 6, 11, 7, 6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, 7, 2, 3, 6, 2, 7, 7, 0, 8, 7, 6, 0, 6,
    2, 0, 2, 7, 6, 2, 3, 7, 0, 1, 9, 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, 10, 7, 6, 10, 1, 7, 1,
    3, 7, 10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, 0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, 7, 6, 10, 7,
    10, 8, 8, 10, 9, 6, 8, 4, 11, 8, 6, 3, 6, 11, 3, 0, 6, 0, 4, 6, 8, 6, 11, 8, 4, 6, 9, 0, 1, 9,
    4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, 6, 8, 4, 6, 11, 8, 2, 10, 1, 1, 2, 10, 3, 0, 11, 0, 6, 11, 0,
    4, 6, 4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, 10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, 8,
    2, 3, 8, 4, 2, 4, 6, 2, 0, 4, 2, 4, 6, 2, 1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, 1, 9, 4, 1,
    4, 2, 2, 4, 6, 8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, 10, 1, 0, 10, 0, 6, 6, 0, 4, 4, 6, 3, 4,
    3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, 10, 9, 4, 6, 10, 4, 4, 9, 5, 7, 6, 11, 0, 8, 3, 4, 9, 5, 11,
    7, 6, 5, 0, 1, 5, 4, 0, 7, 6, 11, 11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, 9, 5, 4, 10, 1, 2, 7,
    6, 11, 6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, 7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, 3, 4, 8, 3,
    5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, 7, 2, 3, 7, 6, 2, 5, 4, 9, 9, 5, 4, 0, 8, 6, 0, 6, 2, 6,
    8, 7, 3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, 9,
    5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, 1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, 4, 0, 10, 4,
    10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, 7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, 6, 9, 5, 6, 11, 9, 11,
    8, 9, 3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, 0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, 6, 11, 3, 6,
    3, 5, 5, 3, 1, 1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, 0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1,
    2, 10, 11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, 6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, 5,
    8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, 9, 5, 6, 9, 6, 0, 0, 6, 2, 1, 5, 8, 1, 8, 0, 5, 6, 8, 3,
    8, 2, 6, 2, 8, 1, 5, 6, 2, 1, 6, 1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, 10, 1, 0, 10,
    0, 6, 9, 5, 0, 5, 6, 0, 0, 3, 8, 5, 6, 10, 10, 5, 6, 11, 5, 10, 7, 5, 11, 11, 5, 10, 11, 7, 5, 8,
    3, 0, 5, 11, 7, 5, 10, 11, 1, 9, 0, 10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, 11, 1, 2, 11, 7, 1, 7,
    5, 1, 0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, 9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, 7, 5, 2, 7,
    2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, 2, 5, 10, 2, 3, 5, 3, 7, 5, 8, 2, 0, 8, 5, 2, 8, 7, 5, 10,
    2, 5, 9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, 9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, 1,
    3, 5, 3, 7, 5, 0, 8, 7, 0, 7, 1, 1, 7, 5, 9, 0, 3, 9, 3, 5, 5, 3, 7, 9, 8, 7, 5, 9, 7, 5,
    8, 4, 5, 10, 8, 10, 11, 8, 5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, 0, 1, 9, 8, 4, 10, 8, 10, 11, 10,
    4, 5, 10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, 2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, 0,
    4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, 0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, 9,
    4, 5, 2, 11, 3, 2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, 5, 10, 2, 5, 2, 4, 4, 2, 0, 3, 10, 2, 3,
    5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, 5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, 8, 4, 5, 8, 5, 3, 3,
    5, 1, 0, 4, 5, 1, 0, 5, 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, 9, 4, 5, 4, 11, 7, 4, 9, 11, 9,
    10, 11, 0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, 1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, 3, 1, 4, 3,
    4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, 4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, 9, 7, 4, 9, 11, 7, 9,
    1, 11, 2, 11, 1, 0, 8, 3, 11, 7, 4, 11, 4, 2, 2, 4, 0, 11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, 2,
    9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, 9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, 3, 7, 10, 3,
    10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, 1, 10, 2, 8, 7, 4, 4, 9, 1, 4, 1, 7, 7, 1, 3, 4, 9, 1, 4,
    1, 7, 0, 8, 1, 8, 7, 1, 4, 0, 3, 7, 4, 3, 4, 8, 7, 9, 10, 8, 10, 11, 8, 3, 0, 9, 3, 9, 11, 11,
    9, 10, 0, 1, 10, 0, 10, 8, 8, 10, 11, 3, 1, 10, 11, 3, 10, 1, 2, 11, 1, 11, 9, 9, 11, 8, 3, 0, 9, 3,
    9, 11, 1, 2, 9, 2, 11, 9, 0, 2, 11, 8, 0, 11, 3, 2, 11, 2, 3, 8, 2, 8, 10, 10, 8, 9, 9, 10, 2, 0,
    9, 2, 2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, 1, 10, 2, 1, 3, 8, 9, 1, 8, 0, 9, 1, 0, 3, 8,
};

struct MC_storage {
    thrust::device_vector<int> tri_range_table;
    thrust::device_vector<int> tri_local_inds_table;
    thrust::device_vector<int2> edge_to_verts_table;
    thrust::device_vector<int3> corner_offsets_table;
    thrust::device_vector<int2> flat_edge_verts_table;
};
static MC_storage* mc_storage = nullptr;
MC_storage& mc_storage_instance() {
    if ( mc_storage == nullptr ) {
        mc_storage = new MC_storage();
        mc_storage->tri_range_table.assign(
            MC_CASE_TO_TRI_RANGE,
            MC_CASE_TO_TRI_RANGE + 257
            );
        mc_storage->tri_local_inds_table.assign(
            MC_TRI_LOCAL_INDICES,
            MC_TRI_LOCAL_INDICES + 2460
            );
        thrust::host_vector<int2> h_edge_verts(12);
        for ( int i = 0; i < 12; ++i ) {
            h_edge_verts[i] = make_int2(
                MC_EDGE_TO_CORNERS[i][0],
                MC_EDGE_TO_CORNERS[i][1]
                );
        }
        mc_storage->edge_to_verts_table = h_edge_verts;
        thrust::host_vector<int3> h_corner_offsets(8);
        for ( int i = 0; i < 8; ++i ) {
            h_corner_offsets[i] = make_int3(
                MC_CUBE_CORNER_OFFSETS[i][0],
                MC_CUBE_CORNER_OFFSETS[i][1],
                MC_CUBE_CORNER_OFFSETS[i][2]
                );
        }
        mc_storage->corner_offsets_table = h_corner_offsets;
        constexpr int total_tri_entries = 2460;
        thrust::host_vector<int2> h_flat_edge_verts(total_tri_entries);
        for ( int i = 0; i < total_tri_entries; ++i ) {
            int edge_idx = MC_TRI_LOCAL_INDICES[i];
            h_flat_edge_verts[i] = make_int2(
                MC_EDGE_TO_CORNERS[edge_idx][0],
                MC_EDGE_TO_CORNERS[edge_idx][1]
                );
        }
        mc_storage->flat_edge_verts_table = h_flat_edge_verts;
    }
    return *mc_storage;
}
struct is_not_empty {
    __host__ __device__ bool operator()(uint32_t slot) const {
        return slot != sdf::SLOT_EMPTY;
    }
};
struct TextureSDFData {
    float3 sdf_box_lower, sdf_box_upper, inv_sdf_dx, voxel_size;
    int3 coarse_dims;
    float fine_to_coarse, subgrid_size_f, subgrid_samples_f;
    float subgrids_sdf_value_range, subgrids_min_sdf_value;
    uint32_t* subgrid_start_slots;
    cudaTextureObject_t coarse_texture, subgrid_texture;
};
__global__ void index_to_int3(int* idx, int3* res, int nx, int ny, int total) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid >= total ) return;
    res[tid] = id_to_xyz(idx[tid], nx, ny);
}

struct CellLoc {
    int x_base, y_base, z_base;
    int ix, iy, iz;         // 精细网格整数坐标
    float tx, ty, tz;       // 三线性分数部分
    uint32_t start_slot;    // 子网格起始槽
};
__device__ CellLoc locate_cell(const TextureSDFData& sdf, float3 f) {
    CellLoc loc;
    // 粗网格尺寸
    int coarse_x = sdf.coarse_dims.x;
    int coarse_y = sdf.coarse_dims.y;
    int coarse_z = sdf.coarse_dims.z;

    // 精细网格整数坐标
    loc.ix = (int)floorf(f.x);
    loc.iy = (int)floorf(f.y);
    loc.iz = (int)floorf(f.z);

    // 所在粗网格块索引
    loc.x_base = (int)(f.x * sdf.fine_to_coarse);
    loc.y_base = (int)(f.y * sdf.fine_to_coarse);
    loc.z_base = (int)(f.z * sdf.fine_to_coarse);

    // 钳制到有效范围
    loc.x_base = min(max(loc.x_base, 0), coarse_x - 1);
    loc.y_base = min(max(loc.y_base, 0), coarse_y - 1);
    loc.z_base = min(max(loc.z_base, 0), coarse_z - 1);

    // 读取子网格起始槽
    loc.start_slot = sdf.subgrid_start_slots[
        idx3d(loc.x_base, loc.y_base, loc.z_base, coarse_x, coarse_y)
    ];

    // 分数坐标（相对于精细网格单元）
    loc.tx = f.x - floorf(f.x);
    loc.ty = f.y - floorf(f.y);
    loc.tz = f.z - floorf(f.z);

    return loc;
}
__device__ float texture_sample_sdf(const TextureSDFData& sdf, float3 local_pos) {
    // 钳制到包围盒
    float3 clamped = fmin3(fmax3(local_pos, sdf.sdf_box_lower), sdf.sdf_box_upper);

    float diff_mag = sqrtf(
        (local_pos.x - clamped.x) * (local_pos.x - clamped.x) +
        (local_pos.y - clamped.y) * (local_pos.y - clamped.y) +
        (local_pos.z - clamped.z) * (local_pos.z - clamped.z)
        );

    float3 f = make_float3(
        (clamped.x - sdf.sdf_box_lower.x) * sdf.inv_sdf_dx.x,
        (clamped.y - sdf.sdf_box_lower.y) * sdf.inv_sdf_dx.y,
        (clamped.z - sdf.sdf_box_lower.z) * sdf.inv_sdf_dx.z
        );

    CellLoc loc = locate_cell(sdf, f);

    float v000, v100, v010, v110, v001, v101, v011, v111;
    bool needs_scale = false;
    float tx = loc.tx, ty = loc.ty, tz = loc.tz;

    if ( loc.start_slot >= sdf::SLOT_LINEAR ) {
        // 从粗网格采样
        float cx = (float)loc.x_base;
        float cy = (float)loc.y_base;
        float cz = (float)loc.z_base;
        float3 coarse_f = make_float3(
            (loc.ix + loc.tx) * sdf.fine_to_coarse,
            (loc.iy + loc.ty) * sdf.fine_to_coarse,
            (loc.iz + loc.tz) * sdf.fine_to_coarse
            );
        tx = coarse_f.x - cx;
        ty = coarse_f.y - cy;
        tz = coarse_f.z - cz;

        // 纹理坐标：整数+0.5
        float3 c000 = make_float3(cx + 0.5f, cy + 0.5f, cz + 0.5f);
        float3 c100 = make_float3(cx + 1.5f, cy + 0.5f, cz + 0.5f);
        float3 c010 = make_float3(cx + 0.5f, cy + 1.5f, cz + 0.5f);
        float3 c110 = make_float3(cx + 1.5f, cy + 1.5f, cz + 0.5f);
        float3 c001 = make_float3(cx + 0.5f, cy + 0.5f, cz + 1.5f);
        float3 c101 = make_float3(cx + 1.5f, cy + 0.5f, cz + 1.5f);
        float3 c011 = make_float3(cx + 0.5f, cy + 1.5f, cz + 1.5f);
        float3 c111 = make_float3(cx + 1.5f, cy + 1.5f, cz + 1.5f);

        v000 = tex3D<float>(sdf.coarse_texture, c000.x, c000.y, c000.z);
        v100 = tex3D<float>(sdf.coarse_texture, c100.x, c100.y, c100.z);
        v010 = tex3D<float>(sdf.coarse_texture, c010.x, c010.y, c010.z);
        v110 = tex3D<float>(sdf.coarse_texture, c110.x, c110.y, c110.z);
        v001 = tex3D<float>(sdf.coarse_texture, c001.x, c001.y, c001.z);
        v101 = tex3D<float>(sdf.coarse_texture, c101.x, c101.y, c101.z);
        v011 = tex3D<float>(sdf.coarse_texture, c011.x, c011.y, c011.z);
        v111 = tex3D<float>(sdf.coarse_texture, c111.x, c111.y, c111.z);
    }
    else {
        // 从子网格纹理采样
        needs_scale = true;
        float block_x = (float)(loc.start_slot & 0x3FF);
        float block_y = (float)((loc.start_slot >> 10) & 0x3FF);
        float block_z = (float)((loc.start_slot >> 20) & 0x3FF);

        float lx = (float)loc.ix - (float)loc.x_base * sdf.subgrid_size_f;
        float ly = (float)loc.iy - (float)loc.y_base * sdf.subgrid_size_f;
        float lz = (float)loc.iz - (float)loc.z_base * sdf.subgrid_size_f;

        float ox = block_x * sdf.subgrid_samples_f + lx + 0.5f;
        float oy = block_y * sdf.subgrid_samples_f + ly + 0.5f;
        float oz = block_z * sdf.subgrid_samples_f + lz + 0.5f;

        v000 = tex3D<float>(sdf.subgrid_texture, ox, oy, oz);
        v100 = tex3D<float>(sdf.subgrid_texture, ox + 1.0f, oy, oz);
        v010 = tex3D<float>(sdf.subgrid_texture, ox, oy + 1.0f, oz);
        v110 = tex3D<float>(sdf.subgrid_texture, ox + 1.0f, oy + 1.0f, oz);
        v001 = tex3D<float>(sdf.subgrid_texture, ox, oy, oz + 1.0f);
        v101 = tex3D<float>(sdf.subgrid_texture, ox + 1.0f, oy, oz + 1.0f);
        v011 = tex3D<float>(sdf.subgrid_texture, ox, oy + 1.0f, oz + 1.0f);
        v111 = tex3D<float>(sdf.subgrid_texture, ox + 1.0f, oy + 1.0f, oz + 1.0f);
    }

    // 三线性插值
    float c00 = v000 + (v100 - v000) * tx;
    float c10 = v010 + (v110 - v010) * tx;
    float c01 = v001 + (v101 - v001) * tx;
    float c11 = v011 + (v111 - v011) * tx;
    float c0 = c00 + (c10 - c00) * ty;
    float c1 = c01 + (c11 - c01) * ty;
    float sdf_val = c0 + (c1 - c0) * tz;

    if ( needs_scale ) {
        sdf_val = sdf_val * sdf.subgrids_sdf_value_range + sdf.subgrids_min_sdf_value;
    }
    return sdf_val + diff_mag;
}

__device__ float texture_sample_sdf_at_voxel(const TextureSDFData& sdf, int ix, int iy, int iz) {
    int coarse_x = sdf.coarse_dims.x;
    int coarse_y = sdf.coarse_dims.y;
    int coarse_z = sdf.coarse_dims.z;

    int x_base = min(max((int)(ix * sdf.fine_to_coarse), 0), coarse_x - 1);
    int y_base = min(max((int)(iy * sdf.fine_to_coarse), 0), coarse_y - 1);
    int z_base = min(max((int)(iz * sdf.fine_to_coarse), 0), coarse_z - 1);

    uint32_t start_slot = sdf.subgrid_start_slots[
        idx3d(x_base, y_base, z_base, coarse_x, coarse_y)
    ];

    if ( start_slot < sdf::SLOT_LINEAR ) {
        float block_x = (float)(start_slot & 0x3FF);
        float block_y = (float)((start_slot >> 10) & 0x3FF);
        float block_z = (float)((start_slot >> 20) & 0x3FF);

        float lx = (float)ix - (float)x_base * sdf.subgrid_size_f;
        float ly = (float)iy - (float)y_base * sdf.subgrid_size_f;
        float lz = (float)iz - (float)z_base * sdf.subgrid_size_f;

        float ox = block_x * sdf.subgrid_samples_f + lx + 0.5f;
        float oy = block_y * sdf.subgrid_samples_f + ly + 0.5f;
        float oz = block_z * sdf.subgrid_samples_f + lz + 0.5f;

        float raw = tex3D<float>(sdf.subgrid_texture, ox, oy, oz);
        return raw * sdf.subgrids_sdf_value_range + sdf.subgrids_min_sdf_value;
    }
    else {
        // 粗网格路径：回退到通用采样
        float3 local_pos = make_float3(
            sdf.sdf_box_lower.x + ix * sdf.voxel_size.x,
            sdf.sdf_box_lower.y + iy * sdf.voxel_size.y,
            sdf.sdf_box_lower.z + iz * sdf.voxel_size.z
            );
        return texture_sample_sdf(sdf, local_pos);
    }
}
__device__ inline float texture_sample_sdf_local(const TextureSDFData& sdf, uint32_t start_slot, int lx, int ly, int lz) {
    float block_x = (float)(start_slot & 0x3FF);
    float block_y = (float)((start_slot >> 10) & 0x3FF);
    float block_z = (float)((start_slot >> 20) & 0x3FF);

    // 这里的 lx, ly, lz 范围是 0 到 8
    float ox = block_x * sdf.subgrid_samples_f + lx + 0.5f;
    float oy = block_y * sdf.subgrid_samples_f + ly + 0.5f;
    float oz = block_z * sdf.subgrid_samples_f + lz + 0.5f;

    float raw = tex3D<float>(sdf.subgrid_texture, ox, oy, oz);
    return raw * sdf.subgrids_sdf_value_range + sdf.subgrids_min_sdf_value;
}
__global__ void generate_isomesh_texture_kernel(
    const TextureSDFData* __restrict__ sdf_array,
    const int3* __restrict__ active_coarse_cells,
    int num_active,
    int subgrid_size,
    const int* __restrict__ tri_range_table,
    const int2* __restrict__ flat_edge_verts_table,
    const int3* __restrict__ corner_offsets_table,
    float isovalue,
    int* __restrict__ face_count,
    float3* __restrict__ vertices_out
) {
    int cell_idx = blockIdx.x;
    if ( cell_idx >= num_active ) return;

    int3 coarse = active_coarse_cells[cell_idx];
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    int local_z = threadIdx.z;

    int x_id = coarse.x * subgrid_size + local_x;
    int y_id = coarse.y * subgrid_size + local_y;
    int z_id = coarse.z * subgrid_size + local_z;

    const TextureSDFData& sdf = sdf_array[0];
    uint32_t slot = sdf.subgrid_start_slots[idx3d(coarse.x, coarse.y, coarse.z, sdf.coarse_dims.x, sdf.coarse_dims.y)];

    // 计算 cube index
    int cube_idx = 0;
    float vals[8];
    float3 corners[8];
    for ( int i = 0; i < 8; i++ ) {
        int3 co = corner_offsets_table[i];
        // vals[i] = texture_sample_sdf_at_voxel(sdf, x_id + co.x, y_id + co.y, z_id + co.z);
        vals[i] = texture_sample_sdf_local(sdf, slot, local_x + co.x, local_y + co.y, local_z + co.z);
        if ( isnan(vals[i]) ) return;
        if ( vals[i] <= isovalue ) cube_idx |= (1 << i);

        // 计算 corner 的世界坐标
        corners[i] = make_float3(
            sdf.sdf_box_lower.x + (x_id + co.x) * sdf.voxel_size.x,
            sdf.sdf_box_lower.y + (y_id + co.y) * sdf.voxel_size.y,
            sdf.sdf_box_lower.z + (z_id + co.z) * sdf.voxel_size.z
            );
    }

    int tri_start = tri_range_table[cube_idx];
    int tri_end = tri_range_table[cube_idx + 1];
    int num_faces = (tri_end - tri_start) / 3;
    if ( num_faces == 0 ) return;

    int base = atomicAdd(face_count, num_faces);

    for ( int f = 0; f < num_faces; f++ ) {
        int tri_offset = tri_start + f * 3;
        for ( int e = 0; e < 3; e++ ) {
            int v0 = flat_edge_verts_table[tri_offset + e].x;
            int v1 = flat_edge_verts_table[tri_offset + e].y;

            float val0 = vals[v0];
            float val1 = vals[v1];
            float t = (isovalue - val0) / (val1 - val0);
            float3 p = corners[v0] + (corners[v1] - corners[v0]) * t;
            vertices_out[base * 3 + f * 3 + e] = p;
        }
    }
}

// 每个 block 处理一个 coarse cell，block 内的线程覆盖 subgrid_size^3 个 fine voxel
__global__ void count_isomesh_faces_texture_kernel(
    const TextureSDFData* __restrict__ sdf_array,   // 长度为 1
    const int3* __restrict__ active_coarse_cells,   // 活跃 coarse cell 列表
    int num_active,
    int subgrid_size,
    const int* __restrict__ tri_range_table,   // [257]
    const int3* __restrict__ corner_offsets_table, // [8][3]
    float isovalue,
    int* __restrict__ face_count                     // 输出，长度为 1
) {
    // blockIdx.x 对应 active_coarse_cells 中的索引
    int cell_idx = blockIdx.x;
    if ( cell_idx >= num_active ) return;

    // 当前 coarse cell 的坐标
    int3 coarse = active_coarse_cells[cell_idx];

    // 全局 fine voxel 坐标
    int x_id = coarse.x * subgrid_size + threadIdx.x;
    int y_id = coarse.y * subgrid_size + threadIdx.y;
    int z_id = coarse.z * subgrid_size + threadIdx.z;

    const TextureSDFData& sdf = sdf_array[0];
    uint32_t slot = sdf.subgrid_start_slots[idx3d(coarse.x, coarse.y, coarse.z, sdf.coarse_dims.x, sdf.coarse_dims.y)];

    // 计算 cube index（8 个 corner 的符号位）
    int cube_idx = 0;
    for ( int i = 0; i < 8; i++ ) {
        int3 co = corner_offsets_table[i];
        // float v = texture_sample_sdf_at_voxel(sdf, x_id + co.x, y_id + co.y, z_id + co.z);
        float v = texture_sample_sdf_local(sdf, slot, threadIdx.x + co.x, threadIdx.y + co.y, threadIdx.z + co.z);
        if ( isnan(v) ) return;   // 如果超出窄带则跳过该 cube
        if ( v <= isovalue ) cube_idx |= (1 << i);
    }

    // 查询该 cube case 对应的三角形数量
    int tri_start = tri_range_table[cube_idx];
    int tri_end = tri_range_table[cube_idx + 1];
    int num_faces = (tri_end - tri_start) / 3;
    // printf("cube_idx:%d, num_faces:%d\n",cube_idx, num_faces);
    if ( num_faces > 0 ) {
        atomicAdd(face_count, num_faces);
    }
}
sdf::MeshResult sdf::SDF::compute_isomesh_from_texture_sdf(float isovalue) {
    int block = 256;
    int3 coarse_dims = params.coarse_dims;
    int cx = coarse_dims.x, cy = coarse_dims.y, cz = coarse_dims.z;

    thrust::device_vector<int> d_active_indices(params.subgrid_start_slots.size());
    auto new_end = thrust::copy_if(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(params.subgrid_start_slots.size()),
        params.subgrid_start_slots.begin(),
        d_active_indices.begin(), is_not_empty());
    int num_active = new_end - d_active_indices.begin();
    thrust::device_vector<int3> d_active_cells(num_active);
    index_to_int3<<<(num_active + block - 1) / block, block>>>(
        d_active_indices.data().get(), d_active_cells.data().get(),
        cx, cy, num_active);

    auto mc_data = mc_storage_instance();

    TextureSDFData sdf_data{
        .sdf_box_lower = params.min_extents,
        .sdf_box_upper = params.max_extents,
        .inv_sdf_dx = make_float3(
            1.0f / params.cell_size.x,
            1.0f / params.cell_size.y,
            1.0f / params.cell_size.z
            ),
        .voxel_size = params.cell_size,
        .coarse_dims = params.coarse_dims,
        .fine_to_coarse = 1.0f / params.subgrid_size,
        .subgrid_size_f = static_cast<float>(params.subgrid_size),
        .subgrid_samples_f = static_cast<float>(params.subgrid_size + 1),
        .subgrids_sdf_value_range = params.sdf_range,
        .subgrids_min_sdf_value = params.sdf_min_value,
        .subgrid_start_slots =
        thrust::raw_pointer_cast(params.subgrid_start_slots.data()),
        .coarse_texture = params.coarse_texture,
        .subgrid_texture = params.subgrid_texture
    };
    thrust::device_vector<TextureSDFData> d_sdf_array(1, sdf_data);

    // 第一步：计数
    thrust::device_vector<int> d_face_count(1);
    d_face_count[0] = 0;
    int subgrid_size = params.subgrid_size;
    count_isomesh_faces_texture_kernel<<<dim3(num_active, 1, 1),dim3(subgrid_size, subgrid_size, subgrid_size)>>>(
        d_sdf_array.data().get(), d_active_cells.data().get(), num_active, subgrid_size,
        mc_data.tri_range_table.data().get(), mc_data.corner_offsets_table.data().get(), isovalue, d_face_count.data().get()
        );

    // 第二步：生成顶点
    int num_faces = d_face_count[0];
    sdf::MeshResult mesh;
    thrust::device_vector<float3>& d_vertices = mesh.positions;
    d_vertices.resize(3 * num_faces);
    d_face_count[0] = 0;

    generate_isomesh_texture_kernel<<<dim3(num_active, 1, 1),dim3(subgrid_size, subgrid_size, subgrid_size)>>>(
        d_sdf_array.data().get(), d_active_cells.data().get(), num_active, subgrid_size,
        mc_data.tri_range_table.data().get(), mc_data.flat_edge_verts_table.data().get(),
        mc_data.corner_offsets_table.data().get(), isovalue,
        d_face_count.data().get(), d_vertices.data().get()
        );
    // int new_num_faces = d_face_count[0];

    return mesh;
}
