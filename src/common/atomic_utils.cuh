#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_utils.h"
#include "vec_math.h"

static __device__ __forceinline__ void atomicAddFloat3(float3* address, float3 val) {
	atomicAdd(&address->x, val.x);
	atomicAdd(&address->y, val.y);
	atomicAdd(&address->z, val.z);
}

static __device__ void atomicAddMat3(Mat3* addr, const Mat3 val) {
    atomicAdd(&(addr->r[0].x), val.r[0].x); atomicAdd(&(addr->r[0].y), val.r[0].y); atomicAdd(&(addr->r[0].z), val.r[0].z);
    atomicAdd(&(addr->r[1].x), val.r[1].x); atomicAdd(&(addr->r[1].y), val.r[1].y); atomicAdd(&(addr->r[1].z), val.r[1].z);
    atomicAdd(&(addr->r[2].x), val.r[2].x); atomicAdd(&(addr->r[2].y), val.r[2].y); atomicAdd(&(addr->r[2].z), val.r[2].z);
}

static __device__ void atomicMinFloat(float* addr, float value) {
	int* addr_as_int = (int*)addr;
	int old = *addr_as_int;
	int expected;
	do {
		expected = old;
		// Note: Here it is assumed that the distance is a positive number; for IEEE754 positive floating-point numbers, the integer comparison logic is consistent with floating-point comparison.
		if (__int_as_float(expected) <= value) return;
		old = atomicCAS(addr_as_int, expected, __float_as_int(value));
	}
	while (old != expected);
}

// Source - https://stackoverflow.com/a/59329536
// Posted by Greg Kramida
// Retrieved 2026-01-29, License - CC BY-SA 4.0
__device__ static inline unsigned char atomicAdd(unsigned char* address, unsigned char val) {
	// offset, in bytes, of the char* address within the 32-bit address of the space that overlaps it
	size_t long_address_modulo = (size_t)address & 3;
	// the 32-bit address that overlaps the same memory
	auto* base_address = (unsigned int*)((unsigned char*)address - long_address_modulo);
	// A 0x3210 selector in __byte_perm will simply select all four bytes in the first argument in the same order.
	// The "4" signifies the position where the first byte of the second argument will end up in the output.
	unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
	// for selecting bytes within a 32-bit chunk that correspond to the char* address (relative to base_address)
	unsigned int selector = selectors[long_address_modulo];
	unsigned int long_old,long_assumed,long_val,replacement;

	long_old = *base_address;

	do {
		long_assumed = long_old;
		// replace bits in long_old that pertain to the char address with those from val
		long_val = __byte_perm(long_old, 0, long_address_modulo) + val;
		replacement = __byte_perm(long_old, long_val, selector);
		long_old = atomicCAS(base_address, long_assumed, replacement);
	}
	while (long_old != long_assumed);
	return __byte_perm(long_old, 0, long_address_modulo);
}

__device__ static inline unsigned char atomicOr(unsigned char* address, unsigned char val) {
	// offset, in bytes, of the char* address within the 32-bit address of the space that overlaps it
	size_t long_address_modulo = (size_t)address & 3;
	// the 32-bit address that overlaps the same memory
	auto* base_address = (unsigned int*)((unsigned char*)address - long_address_modulo);
	// A 0x3210 selector in __byte_perm will simply select all four bytes in the first argument in the same order.
	// The "4" signifies the position where the first byte of the second argument will end up in the output.
	unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
	// for selecting bytes within a 32-bit chunk that correspond to the char* address (relative to base_address)
	unsigned int selector = selectors[long_address_modulo];
	unsigned int long_old,long_assumed,long_val,replacement;

	long_old = *base_address;

	do {
		long_assumed = long_old;
		long_val = __byte_perm(long_old, 0, long_address_modulo) | (unsigned int)val;

		replacement = __byte_perm(long_old, long_val, selector);
		long_old = atomicCAS(base_address, long_assumed, replacement);
	}
	while (long_old != (unsigned int)long_assumed);
	return __byte_perm(long_old, 0, long_address_modulo);
}

