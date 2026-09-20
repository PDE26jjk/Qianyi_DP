#pragma once

#include <cstddef>
#include <vector>

struct int2;
struct int3;

// Options for gcdtBuildMesh().
struct GcdtBuildOptions
{
	// Maximum number of attempts. Every attempt runs on a deterministically
	// perturbed vertex set (see perturbAmplitude); later attempts use a larger
	// perturbation, which leaves additional degenerate configurations behind.
	int maxAttempts = 4;
	// Perturbation amplitude in quantisation units (1 unit = 2^-30 of the
	// normalised domain); attempt i uses perturbAmplitude * (i + 1). The default
	// removes the degenerate branch described in validate.h - exactly collinear
	// constraint segments - which the raw pipeline handles incorrectly.
	// 0 disables perturbation (raw upstream behaviour, single attempt).
	int perturbAmplitude = 4;
	// Upper bound for device memory kept for reuse by the allocator. The live
	// set of the pipeline itself is not affected by this value.
	size_t memoryLimitBytes = 128ull * 1024ull * 1024ull;
	// Cub scratch size; 0 derives it from the scene size.
	size_t cubBytes = 0;
	// Drop the four helper corners (and their frame triangles) from the result,
	// keeping only the caller's own vertices. Recommended: the helper corners
	// are an implementation detail of the algorithm.
	bool dropHelperCorners = true;
};

struct GcdtBuildStats
{
	int attempts = 0;         // pipeline runs performed
	int badTriangles = 0;     // suspicious triangles in the last rejected attempt
	bool valid = false;       // the returned mesh passed validation
	double msTotal = 0.0;     // wall time of the whole call
	double msPerAttempt = 0.0;
};

// Result mesh. Vertices are quantised integer coordinates in [0, 2^30); use
// `origIndex` to map an output vertex back to the caller's vertex id. Entries
// >= numInputPoints are the four helper corners the algorithm appends; drop the
// triangles that touch them if the caller only wants a triangulation of its own
// vertices.
struct GcdtMesh
{
	std::vector<int2> points;
	std::vector<int3> triangles;
	std::vector<int> origIndex;
	int numInputPoints = 0;
};

// Builds a constrained Delaunay triangulation of `xyInput` (2*numPoints
// doubles, any range - the call normalises internally) with `constraints`
// (index pairs into the input vertices).
//
// The result is validated on the device before it is returned; invalid results
// are retried with a perturbed vertex set. Returns false when no attempt
// produced a valid mesh (the caller should fall back to another backend).
bool gcdtBuildMesh(const double* xyInput, int numPoints, const int2* constraints, int numConstraints,
                   const GcdtBuildOptions& options, GcdtMesh& mesh, GcdtBuildStats* stats = nullptr);
