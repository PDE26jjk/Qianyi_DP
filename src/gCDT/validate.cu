// Device-side validation of the triangulation and the perturbation helper used
// to retry degenerate runs.
// Why this exists: the constraint-enforcement stage can, for degenerate
// constraint geometry (for example exactly axis-aligned rectangle constraints
// on an axis-aligned grid), emit overlapping triangles - the same constraint
// edge ends up in three triangles, two of them on the same side. The result is
// a non-manifold mesh with a small overlap.
// The pipeline therefore validates every result before returning it and retries
// with a tiny deterministic perturbation of the quantised vertices when the
// result is invalid (see pipeline.cu). One quantisation unit is 2^-30 of the
// normalised domain, so the perturbation is far below float32 precision and
// does not move any constraint in a way a caller could observe.

#include "cuda_runtime.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "utils.cuh"
#include "math_base.cuh"
#include "validate.h"

namespace {

struct EdgeSlot
{
	unsigned long long key;
	int count;
	int owner;
};

__host__ __device__ __forceinline__ unsigned long long hash64(unsigned long long x)
{
	x ^= x >> 33;
	x *= 0xff51afd7ed558ccdULL;
	x ^= x >> 33;
	x *= 0xc4ceb9fe1a85ec53ULL;
	x ^= x >> 33;
	return x;
}

__host__ __device__ __forceinline__ unsigned long long edgeKey(int u, int v)
{
	int lo = min(u, v);
	int hi = max(u, v);
	return ((unsigned long long)(unsigned int)lo << 32) | (unsigned int)hi;
}

__host__ __device__ __forceinline__ long long crossLL(const int2& a, const int2& b, const int2& c)
{
	return (long long)(b.x - a.x) * (long long)(c.y - a.y) -
	       (long long)(b.y - a.y) * (long long)(c.x - a.x);
}

__host__ __device__ __forceinline__ void triEdge(const int3& t, int e, int& u, int& v)
{
	u = (e == 0) ? t.x : ((e == 1) ? t.y : t.z);
	v = (e == 0) ? t.y : ((e == 1) ? t.z : t.x);
}

// soon as any edge is used more than twice.
__global__ void buildEdgeTable(const int3* tris, int numTri, EdgeSlot* table, int tableSize, int* flag)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= numTri * 3) return;

	int t = tid / 3;
	int e = tid % 3;
	int u, v;
	triEdge(tris[t], e, u, v);
	if (u == v)
	{
		atomicExch(flag, 1);
		return;
	}

	unsigned long long key = edgeKey(u, v);
	unsigned int pos = (unsigned int)(hash64(key) & (unsigned long long)(tableSize - 1));
	for (int probe = 0; probe < 64; ++probe)
	{
		unsigned long long cur = atomicCAS(&table[pos].key, 0ull, key);
		if (cur == 0ull || cur == key)
		{
			int c = atomicAdd(&table[pos].count, 1) + 1;
			if (c == 1) table[pos].owner = t;
			if (c > 2) atomicExch(flag, 1);
			return;
		}
		pos = (pos + 1u) & (unsigned int)(tableSize - 1);
	}
	atomicExch(flag, 1);  // saturated table: be conservative
}

// only costs a host-side confirmation.
__global__ void buildTriHashTable(const int3* tris, int numTri, EdgeSlot* table, int tableSize, int* flag)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= numTri) return;

	int3 tri = tris[t];
	int a = tri.x, b = tri.y, c = tri.z;
	if (a == b || b == c || a == c)
	{
		atomicExch(flag, 1);
		return;
	}
	if (a > b) { int s = a; a = b; b = s; }
	if (b > c) { int s = b; b = c; c = s; }
	if (a > b) { int s = a; a = b; b = s; }

	unsigned long long key = hash64(((unsigned long long)(unsigned int)a << 42) ^
	                                ((unsigned long long)(unsigned int)b << 21) ^
	                                (unsigned long long)(unsigned int)c);
	if (key == 0ull) key = 1ull;

	unsigned int pos = (unsigned int)(key & (unsigned long long)(tableSize - 1));
	for (int probe = 0; probe < 64; ++probe)
	{
		unsigned long long cur = atomicCAS(&table[pos].key, 0ull, key);
		if (cur == 0ull || cur == key)
		{
			int c2 = atomicAdd(&table[pos].count, 1) + 1;
			if (c2 == 1) table[pos].owner = t;
			if (c2 > 1) atomicExch(flag, 1);
			return;
		}
		pos = (pos + 1u) & (unsigned int)(tableSize - 1);
	}
	atomicExch(flag, 1);
}

__device__ __forceinline__ bool isHelperCorner(const int2& p, int res)
{
	// The four corners of the quantised square; triangles that touch them form
	// the frame that gcdtBuildMesh() drops before returning, so they are not
	// validated (the caller never sees them).
	return (p.x == 0 && p.y == 0) || (p.x == res && p.y == 0) ||
	       (p.x == res && p.y == res) || (p.x == 0 && p.y == res);
}

__device__ __forceinline__ int classifyTriangle(const Point2d* points, const int3* tris, int t,
                                                 const EdgeSlot* table, int tableSize,
                                                 const EdgeSlot* triTable, int triTableSize)
{
	// 1 = non-manifold edge, 2 = duplicate triangle, 3 = repeated index,
	// 4 = zero area, 0 = fine.
	int3 tri = tris[t];
	if (tri.x == tri.y || tri.y == tri.z || tri.x == tri.z) return 3;

	const int res = (1 << logPointRes) - 1;
	if (isHelperCorner(points[tri.x], res) || isHelperCorner(points[tri.y], res) ||
	    isHelperCorner(points[tri.z], res))
		return 0;  // frame triangle: dropped before the caller sees the mesh

	if (crossLL(points[tri.x], points[tri.y], points[tri.z]) == 0) return 4;

	for (int e = 0; e < 3; ++e)
	{
		int u, v;
		triEdge(tri, e, u, v);
		unsigned long long key = edgeKey(u, v);
		unsigned int pos = (unsigned int)(hash64(key) & (unsigned long long)(tableSize - 1));
		for (int probe = 0; probe < 64; ++probe)
		{
			unsigned long long cur = table[pos].key;
			if (cur == 0ull) break;
			if (cur == key)
			{
				if (table[pos].count > 2) return 1;
				break;
			}
			pos = (pos + 1u) & (unsigned int)(tableSize - 1);
		}
	}

	int a = tri.x, b = tri.y, c = tri.z;
	if (a > b) { int s = a; a = b; b = s; }
	if (b > c) { int s = b; b = c; c = s; }
	if (a > b) { int s = a; a = b; b = s; }
	unsigned long long key = hash64(((unsigned long long)(unsigned int)a << 42) ^
	                                ((unsigned long long)(unsigned int)b << 21) ^
	                                (unsigned long long)(unsigned int)c);
	if (key == 0ull) key = 1ull;
	unsigned int pos = (unsigned int)(key & (unsigned long long)(triTableSize - 1));
	for (int probe = 0; probe < 64; ++probe)
	{
		unsigned long long cur = triTable[pos].key;
		if (cur == 0ull) break;
		if (cur == key)
		{
			if (triTable[pos].count > 1) return 2;
			break;
		}
		pos = (pos + 1u) & (unsigned int)(triTableSize - 1);
	}
	return 0;
}

__global__ void countSuspiciousTriangles(const Point2d* points, const int3* tris, int numTri,
                                         const EdgeSlot* table, int tableSize,
                                         const EdgeSlot* triTable, int triTableSize,
                                         int* counters)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= numTri) return;
	const int cls = classifyTriangle(points, tris, t, table, tableSize, triTable, triTableSize);
	if (cls > 0) atomicAdd(counters + (cls - 1), 1);
}

__global__ void perturbPointsKernel(Point2d* points, int numPoints, unsigned int seed, int amplitude, int res)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= numPoints) return;

	unsigned long long h = hash64(((unsigned long long)seed << 32) ^ (unsigned long long)(unsigned int)tid);
	int dx = (int)(h % (unsigned int)(2 * amplitude + 1)) - amplitude;
	h = hash64(h);
	int dy = (int)(h % (unsigned int)(2 * amplitude + 1)) - amplitude;

	Point2d p = points[tid];
	points[tid].x = max(0, min(res - 1, p.x + dx));
	points[tid].y = max(0, min(res - 1, p.y + dy));
}

// caller can retry with a different perturbation.
__global__ void countDuplicateVertices(const Point2d* points, int numPoints, EdgeSlot* table, int tableSize,
                                       int* duplicates)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= numPoints) return;

	unsigned long long key = ((unsigned long long)(unsigned int)points[tid].x << 32) |
	                         (unsigned int)points[tid].y;
	// domain, so only the helper corner (0,0) can hit this.
	if (key == 0ull) key = 1ull;

	unsigned int pos = (unsigned int)(hash64(key) & (unsigned long long)(tableSize - 1));
	for (int probe = 0; probe < 64; ++probe)
	{
		unsigned long long cur = atomicCAS(&table[pos].key, 0ull, key);
		if (cur == 0ull || cur == key)
		{
			int c = atomicAdd(&table[pos].count, 1) + 1;
			if (c > 1) atomicAdd(duplicates, 1);
			return;
		}
		pos = (pos + 1u) & (unsigned int)(tableSize - 1);
	}
}

}  // namespace

int gcdtValidateMeshDetailed(const Point2d* points, int numPoints, const int3* tris, int numTriangles)
{
	(void)numPoints;
	if (numTriangles <= 0) return 0;

	int tableSize = 1;
	while (tableSize < numTriangles * 3 * 2) tableSize <<= 1;
	int triTableSize = 1;
	while (triTableSize < numTriangles * 2) triTableSize <<= 1;

	EdgeSlot* table = nullptr;
	EdgeSlot* triTable = nullptr;
	EdgeSlot* vertexTable = nullptr;
	int* flag = nullptr;
	int* badCount = nullptr;
	int vertexTableSize = 1;
	while (vertexTableSize < numPoints * 2) vertexTableSize <<= 1;
	utils::malloc(table, tableSize);
	utils::malloc(triTable, triTableSize);
	utils::malloc(vertexTable, vertexTableSize);
	utils::malloc(flag, 2);
	utils::malloc(badCount, 5);
	utils::memset(table, tableSize, 0);
	utils::memset(triTable, triTableSize, 0);
	utils::memset(vertexTable, vertexTableSize, 0);
	utils::memset(flag, 2, 0);
	utils::memset(badCount, 5, 0);

	const int block = 256;
	buildEdgeTable<<<(numTriangles * 3 + block - 1) / block, block>>>(
	    tris, numTriangles, table, tableSize, flag);
	getLastCudaError("buildEdgeTable");
	buildTriHashTable<<<(numTriangles + block - 1) / block, block>>>(
	    tris, numTriangles, triTable, triTableSize, flag + 1);
	getLastCudaError("buildTriHashTable");
	countSuspiciousTriangles<<<(numTriangles + block - 1) / block, block>>>(
	    points, tris, numTriangles, table, tableSize, triTable, triTableSize, badCount);
	getLastCudaError("countSuspiciousTriangles");
	if (numPoints > 0)
	{
		countDuplicateVertices<<<(numPoints + block - 1) / block, block>>>(
		    points, numPoints, vertexTable, vertexTableSize, badCount + 1);
		getLastCudaError("countDuplicateVertices");
	}

	int host[7] = { 0, 0, 0, 0, 0, 0, 0 };
	utils::memcpy(host, flag, 2, cudaMemcpyDeviceToHost);
	utils::memcpy(host + 2, badCount, 2, cudaMemcpyDeviceToHost);
	getLastCudaError("validate copy");

	utils::release(table, tableSize);
	utils::release(triTable, triTableSize);
	utils::release(vertexTable, vertexTableSize);
	utils::release(flag, 2);
	utils::release(badCount, 5);

	if (std::getenv("GCDT_VERBOSE"))
	{
		std::printf("[gCDT] validate: nonManifold=%d duplicate=%d repeatedIndex=%d zeroArea=%d coincidentVertices=%d\n",
		            host[2], host[3], host[4], host[5], host[6]);
	}
	const int badTriangles = host[2] + host[3] + host[4] + host[5] + host[6];
	const bool tableFlag = (host[0] != 0) || (host[1] != 0);
	return std::max(badTriangles, tableFlag ? 1 : 0);
}

bool gcdtValidateMesh(const Point2d* points, int numPoints, const int3* tris, int numTriangles,
                      int* badTriangles)
{
	int bad = gcdtValidateMeshDetailed(points, numPoints, tris, numTriangles);
	if (badTriangles) *badTriangles = bad;
	return bad == 0;
}

void gcdtPerturbPoints(Point2d* points, int numPoints, unsigned int seed, int amplitude)
{
	if (amplitude <= 0 || numPoints <= 0) return;
	const int block = 256;
	const int res = (1 << logPointRes) - 1;
	perturbPointsKernel<<<(numPoints + block - 1) / block, block>>>(points, numPoints, seed, amplitude, res);
	getLastCudaError("perturbPointsKernel");
}
