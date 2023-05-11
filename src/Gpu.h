#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "common.h"
#include "stddef.h"

#ifdef __cplusplus
extern "C" {
#endif

void GPU_DataCube_boxcar_filter(char *data, int word_size, size_t data_size, size_t *axis_size, size_t radius);

__global__ void g_DataCube_boxcar_filter(char *data, int word_size, size_t width, size_t height, size_t depth, size_t radius);

__device__ void d_filter_boxcar_1d_flt(float *data, float *data_copy, const size_t size, const size_t filter_radius);

__device__ inline size_t get_index( const size_t x, const size_t y, const size_t z, const size_t width, const size_t height)
{
	return x + width * (y + height * z);
}

#ifdef __cplusplus
}
#endif