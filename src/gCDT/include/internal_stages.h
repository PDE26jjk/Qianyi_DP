#pragma once

// Declarations of the pipeline stages implemented in kernel.cu and used by the
// reusable gcdtBuildMesh() driver.

struct int2;
struct int3;
typename Point2d;

double filterTriangles(void* cubTemp, size_t cubBytes, int3* tris, int3* newTris,
                       int3* adjTris, int3* newAdjTris, int* sons, int& numTriangles);
double delaunaySample(void* cubTemp, size_t cubBytes, Point2d* points, int3* tris,
                      int3* adjTris, int numPoints, int numTriangles);
double delaunayNew(void* cubTemp, size_t cubBytes, Point2d*& points, int3*& tris,
                   int2* cons, int numPoints, int numTriangles, int numCons, int* origIndex = nullptr);

__global__ void transPoints(Point2d* points, double* fpoints, long long res, int numP);
__global__ void convertSample(int3* tris, int* sampleId, int numSamples, int numPoints, int numT);
