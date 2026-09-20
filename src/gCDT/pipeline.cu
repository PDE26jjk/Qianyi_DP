// Reusable driver for the whole gCDT pipeline.
// Differences to the demo entry point in kernel.cu:
//   * input is plain doubles and constraints by index (no scene file),
//   * the cub scratch and the device memory pool are sized from the scene,
//   * every result is validated and invalid results are retried with a tiny
//     perturbation of the quantised vertices,
//   * the caller receives host arrays plus the vertex-id map.

#include "cuda_runtime.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "utils.cuh"
#include "math_base.cuh"
#include "rawMesh.h"
#include "addConstraints.h"
#include "boostmap.h"
#include "fastDelaunay.h"
#include "validate.h"
#include "pipeline.h"
#include "internal_stages.h"

namespace {

const int kHelperCorners = 4;

// Corner coordinates of the quantised square, matching insertCornerPoints().
inline int2 helperCorner(int k)
{
	const int res = (1 << logPointRes) - 1;
	switch (k)
	{
	case 0: return int2{ 0, 0 };
	case 1: return int2{ res, 0 };
	case 2: return int2{ res, res };
	default: return int2{ 0, res };
	}
}

void normaliseInput(const double* xy, int numPoints, std::vector<double>& out)
{
	out.resize((size_t)numPoints * 2);
	double minx = 1e300, miny = 1e300, maxx = -1e300, maxy = -1e300;
	for (int i = 0; i < numPoints; ++i)
	{
		minx = std::min(minx, xy[2 * i]);
		maxx = std::max(maxx, xy[2 * i]);
		miny = std::min(miny, xy[2 * i + 1]);
		maxy = std::max(maxy, xy[2 * i + 1]);
	}
	const double cx = 0.5 * (minx + maxx);
	const double cy = 0.5 * (miny + maxy);
	double extent = std::max(maxx - minx, maxy - miny);
	if (!(extent > 0.0)) extent = 1.0;
	for (int i = 0; i < numPoints; ++i)
	{
		out[2 * i] = (xy[2 * i] - cx) / extent * 0.99 + 0.5;
		out[2 * i + 1] = (xy[2 * i + 1] - cy) / extent * 0.99 + 0.5;
	}
}

void quantiseReference(const std::vector<double>& xy, int numPoints, std::vector<int2>& out)
{
	const long long res = 1LL << logPointRes;
	out.resize(numPoints);
	for (int i = 0; i < numPoints; ++i)
	{
		long long x = (long long)(xy[2 * i] * (double)res);
		long long y = (long long)(xy[2 * i + 1] * (double)res);
		out[i].x = (int)std::max(0LL, std::min(res - 1, x));
		out[i].y = (int)std::max(0LL, std::min(res - 1, y));
	}
}

}  // namespace

bool gcdtBuildMesh(const double* xyInput, int numPoints, const int2* constraints, int numConstraints,
                   const GcdtBuildOptions& options, GcdtMesh& mesh, GcdtBuildStats* statsOut)
{
	GcdtBuildStats stats;
	mesh = GcdtMesh();
	mesh.numInputPoints = numPoints;

	const auto t0 = std::chrono::steady_clock::now();
	if (numPoints < 3)
	{
		if (statsOut) *statsOut = stats;
		return false;
	}

	getMemCacheRef().setCacheLimit(options.memoryLimitBytes);

	std::vector<double> xy;
	normaliseInput(xyInput, numPoints, xy);
	std::vector<int2> reference;
	quantiseReference(xy, numPoints, reference);
	auto setupDone = std::chrono::steady_clock::now();

	const int numVerts = numPoints + kHelperCorners;
	const int maxTriangles = numVerts * 4;
	const size_t cubBytes = options.cubBytes > 0
	                            ? options.cubBytes
	                            : utils::cubTempBytes(std::max(numVerts * 12 + numConstraints + 1, 1));

	unsigned char* cubTempStorage = nullptr;
	utils::malloc(cubTempStorage, (int)cubBytes);
	auto setupCubDone = std::chrono::steady_clock::now();
	const bool verbose = std::getenv("GCDT_VERBOSE") != nullptr;
	if (verbose)
	{
		std::printf("[gCDT] setup normalise=%.1f cub=%zuMB alloc=%.1f\n",
		            std::chrono::duration<double, std::milli>(setupDone - t0).count(),
		            cubBytes / 1048576u,
		            std::chrono::duration<double, std::milli>(setupCubDone - setupDone).count());
	}

	std::vector<int> hostMap(numVerts);
	for (int i = 0; i < numVerts; ++i) hostMap[i] = i;

	// Perturbation defaults on: exactly collinear constraint segments (for
	// example several holes whose edges share one supporting line - a perfectly
	// normal CAD input) drive the constraint stage into a degenerate branch, see
	// the root-cause note in validate.h. A sub-quantum perturbation leaves that
	// configuration behind without moving anything a caller can observe.
	const int amplitude = std::max(0, options.perturbAmplitude);
	const int attempts = amplitude > 0 ? std::max(1, options.maxAttempts) : 1;
	for (int attempt = 0; attempt < attempts; ++attempt)
	{
		stats.attempts = attempt + 1;

		// --- device state for this attempt ---------------------------------
		auto attemptAllocStart = std::chrono::steady_clock::now();
		double* d_xy = nullptr;
		utils::mallocAndCpy(d_xy, xy.data(), numPoints * 2);

		Point2d* points = nullptr;
		utils::malloc(points, numVerts);
		{
			long long res = 1LL << logPointRes;
			transPoints<<<(numPoints + 127) / 128, 128>>>(points, d_xy, res, numPoints);
			getLastCudaError("transPoints");
		}
		utils::release(d_xy, numPoints * 2);

		if (amplitude > 0)
		{
			const unsigned seed = 0x9E3779B9u * (unsigned)(attempt + 1);
			gcdtPerturbPoints(points, numPoints, seed, amplitude * (attempt + 1));
		}

		int2* cons = nullptr;
		if (numConstraints > 0) utils::mallocAndCpy(cons, constraints, numConstraints);

		int* origIndex = nullptr;
		utils::mallocAndCpy(origIndex, hostMap.data(), numVerts);

		int3* tris = nullptr;
		int3* adjTris = nullptr;
		int* sons = nullptr;
		utils::malloc(tris, maxTriangles);
		utils::malloc(adjTris, maxTriangles);
		utils::malloc(sons, maxTriangles);
		double tAlloc = std::chrono::duration<double, std::milli>(
		    std::chrono::steady_clock::now() - attemptAllocStart).count();

		auto stagesStart = std::chrono::steady_clock::now();
		Point2d* samplePoints = nullptr;
		int* sampleId = nullptr;
		int numSample = 0;
		double tSample = 0, tBefore = 0, tBm = 0, tSampleD = 0, tAfter = 0, tCons = 0, tNew = 0, tVal = 0;
		tSample = getSamplePoints(cubTempStorage, cubBytes, points, numPoints, samplePoints, sampleId, numSample);

		int numTriangles = 0;
		tBefore = rawMeshBefore(cubTempStorage, cubBytes, samplePoints, tris, sons, numTriangles, numSample);

		int3* newTrisSample = nullptr;
		utils::malloc(newTrisSample, numTriangles);
		filterTriangles(cubTempStorage, cubBytes, tris, newTrisSample, nullptr, nullptr, sons, numTriangles);
		tSampleD = delaunaySample(cubTempStorage, cubBytes, samplePoints, newTrisSample, adjTris, numSample + kHelperCorners, numTriangles);

		int* bitmapOffsets = nullptr;
		int* bitmapTris = nullptr;
		tBm = getBitmap(cubTempStorage, cubBytes, samplePoints, newTrisSample, bitmapOffsets, bitmapTris, numTriangles);
		convertSample<<<(numTriangles + 127) / 128, 128>>>(
		    newTrisSample, sampleId, numSample, numPoints, numTriangles);
		getLastCudaError("convertSample");
		utils::memcpy(tris, newTrisSample, numTriangles, cudaMemcpyDeviceToDevice);

		tAfter = rawMeshAfter(cubTempStorage, cubBytes, points, tris, adjTris, sons, bitmapOffsets, bitmapTris,
		                      numTriangles, numPoints);

		int3* consTris = nullptr;
		int numConsTris = 0;
		if (numConstraints > 0)
		{
			tCons = addConstraints(cubTempStorage, cubBytes, points, tris, adjTris, sons, bitmapOffsets, bitmapTris,
			                       cons, consTris, numConstraints, numPoints, numTriangles, numConsTris);
		}

		int3* newTris = nullptr;
		int3* newAdjTris = nullptr;
		utils::malloc(newTris, numTriangles);
		utils::malloc(newAdjTris, numTriangles);
		filterTriangles(cubTempStorage, cubBytes, tris, newTris, adjTris, newAdjTris, sons, numTriangles);
		if (numConsTris != 0)
			utils::memcpy(newTris + numTriangles, consTris, numConsTris, cudaMemcpyDeviceToDevice);
		numTriangles += numConsTris;

		tNew = delaunayNew(cubTempStorage, cubBytes, points, newTris, cons, numVerts, numTriangles, numConstraints,
		                   origIndex);
		double tStagesWall = std::chrono::duration<double, std::milli>(
		    std::chrono::steady_clock::now() - stagesStart).count();

		// Validate the geometry that is actually handed back: restore the
		// unperturbed coordinates (in the final permutation) first, so a
		// perturbation-induced degeneracy cannot slip through.
		{
			std::vector<int> mapTmp(numVerts);
			utils::memcpy(mapTmp.data(), origIndex, numVerts, cudaMemcpyDeviceToHost);
			std::vector<int2> clean(numVerts);
			for (int i = 0; i < numVerts; ++i)
			{
				const int src = mapTmp[i];
				clean[i] = (src < numPoints) ? reference[src] : helperCorner(src - numPoints);
			}
			utils::memcpy(points, clean.data(), numVerts, cudaMemcpyHostToDevice);
		}

		// --- validate -------------------------------------------------------
		auto valStart = std::chrono::steady_clock::now();
		int bad = gcdtValidateMeshDetailed(points, numVerts, newTris, numTriangles);
		tVal = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - valStart).count();
		stats.badTriangles = bad;
		if (verbose || bad != 0)
		{
			std::printf("[gCDT] attempt %d: %d triangles, %d suspicious | alloc=%.1f stagesWall=%.1f validate=%.1f "
			            "(sample=%.1f before=%.1f sampleD=%.1f bitmap=%.1f after=%.1f cons=%.1f new=%.1f)\n",
			            attempt, numTriangles, bad, tAlloc, tStagesWall, tVal,
			            tSample, tBefore, tSampleD, tBm, tAfter, tCons, tNew);
		}

		if (bad == 0)
		{
			auto tailStart = std::chrono::steady_clock::now();
			// Host copies: unperturbed coordinates arranged in the final
			// permutation, plus the vertex-id map.
			std::vector<int> map(numVerts);
			utils::memcpy(map.data(), origIndex, numVerts, cudaMemcpyDeviceToHost);
			mesh.points.resize(numVerts);
			for (int i = 0; i < numVerts; ++i)
			{
				const int src = map[i];
				mesh.points[i] = (src < numPoints) ? reference[src] : helperCorner(src - numPoints);
			}
			mesh.triangles.resize(numTriangles);
			utils::memcpy(mesh.triangles.data(), newTris, numTriangles, cudaMemcpyDeviceToHost);
			mesh.origIndex = std::move(map);

			if (options.dropHelperCorners)
			{
				// The vertices are permuted, so the helper corners are not
				// necessarily at the tail: decide by the original vertex id.
				auto isInputVertex = [&](int idx) { return mesh.origIndex[idx] < numPoints; };
				std::vector<int3> kept;
				kept.reserve(mesh.triangles.size());
				for (const int3& t : mesh.triangles)
				{
					if (isInputVertex(t.x) && isInputVertex(t.y) && isInputVertex(t.z)) kept.push_back(t);
				}
				std::vector<int> remap(numVerts, -1);
				int next = 0;
				for (const int3& t : kept)
				{
					const int v[3] = { t.x, t.y, t.z };
					for (int k = 0; k < 3; ++k)
						if (remap[v[k]] < 0) remap[v[k]] = next++;
				}
				std::vector<int2> compactPoints(next);
				std::vector<int> compactMap(next);
				for (int v = 0; v < numVerts; ++v)
				{
					if (remap[v] < 0) continue;
					compactPoints[remap[v]] = mesh.points[v];
					compactMap[remap[v]] = mesh.origIndex[v];
				}
				for (int3& t : kept)
				{
					t.x = remap[t.x];
					t.y = remap[t.y];
					t.z = remap[t.z];
				}
				mesh.points = std::move(compactPoints);
				mesh.origIndex = std::move(compactMap);
				mesh.triangles = std::move(kept);
			}

			utils::release(points, numVerts);
			utils::release(origIndex, numVerts);
			utils::release(tris, maxTriangles);
			utils::release(adjTris, maxTriangles);
			utils::release(sons, maxTriangles);
			utils::release(cons, numConstraints > 0 ? numConstraints : 0);
			utils::release(cubTempStorage, (int)cubBytes);
			double tTail = std::chrono::duration<double, std::milli>(
			    std::chrono::steady_clock::now() - tailStart).count();
			if (verbose) std::printf("[gCDT] accept: copies+release=%.1f ms\n", tTail);

			stats.valid = true;
			stats.msTotal = std::chrono::duration<double, std::milli>(
			    std::chrono::steady_clock::now() - t0).count();
			stats.msPerAttempt = stats.msTotal / stats.attempts;
			if (statsOut) *statsOut = stats;
			return true;
		}

		// --- reject and retry ----------------------------------------------
		utils::release(points, numVerts);
		utils::release(origIndex, numVerts);
		utils::release(tris, maxTriangles);
		utils::release(adjTris, maxTriangles);
		utils::release(sons, maxTriangles);
		utils::release(cons, numConstraints > 0 ? numConstraints : 0);
	}

	utils::release(cubTempStorage, (int)cubBytes);
	stats.valid = false;
	stats.msTotal = std::chrono::duration<double, std::milli>(
	    std::chrono::steady_clock::now() - t0).count();
	stats.msPerAttempt = stats.msTotal / stats.attempts;
	if (statsOut) *statsOut = stats;
	return false;
}
