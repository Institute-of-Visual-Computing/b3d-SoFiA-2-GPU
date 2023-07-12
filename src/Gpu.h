#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "common.h"
#include "stddef.h"

#ifdef __cplusplus
extern "C" {
#endif

void GPU_DataCube_filter(char *data, char *originalData, int word_size, size_t data_size, size_t *axis_size, size_t radiusGauss, size_t n_iter, size_t radiusBoxcar); 

void GPU_DataCube_filter_Chunked(char *data, char *originalData, int word_size, size_t data_size, size_t *axis_size, size_t radiusGauss, size_t n_iter, size_t radiusBoxcar, size_t number_of_chunks); 

void GPU_DataCube_boxcar_filter(char *data, int word_size, size_t data_size, size_t *axis_size, size_t radius);

void GPU_DataCube_gauss_filter(char *data, int word_size, size_t data_size, size_t *axis_size, size_t radius, size_t n_iter);

void GPU_DataCube_copy_mask_8_to_1(char* maskData1, char* maskData8, size_t *axis_size);


__global__ void g_DataCube_boxcar_filter(float *data, char *originalData, float *data_box, int word_size, const size_t startSlice, size_t width, size_t height, size_t depth, size_t radius, size_t chunck_type);

__global__ void g_DataCube_gauss_filter_XDir(float *data, float *data_box, int word_size, size_t width, size_t height, size_t depth, size_t radius, size_t n_iter);

__global__ void g_DataCube_gauss_filter_YDir(float *data, float *data_box, int word_size, size_t width, size_t height, size_t depth, size_t radius, size_t n_iter);

__global__ void g_DataCube_copy_mask_8_to_1(char* maskData1, char* maskData8, size_t width, size_t height, size_t depth);

__global__ void g_DataCube_copy_back_smoothed_cube(char *originalData, float *data, int word_size, size_t width, size_t height, size_t depth);

__global__ void g_DataCube_stat_mad(float *data, float *data_box, size_t width, size_t height, size_t depth, const double value, const size_t cadence, const int range);


__device__ void d_filter_boxcar_1d_flt(float *data, float *data_copy, const size_t size, const size_t filter_radius, const size_t jump);

__device__ void d_filter_chunk_boxcar_1d_flt(float *data, char *originalData, float *data_copy, const size_t startSlice, const size_t size, const size_t filter_radius, const size_t jump, size_t chunk_type);

__device__ inline size_t get_index( const size_t x, const size_t y, const size_t z, const size_t width, const size_t height)
{
	return x + width * (y + height * z);
}

// Inline function to set a specific bit in an array
inline void setByte(char* array, int index, bool value) 
{
    int arrayIndex = index / 8;  // Calculate the array index
    int bitOffset = index % 8;   // Calculate the bit offset within the byte

    // Set the bit using bitwise OR
    array[arrayIndex] |= (value << bitOffset);
}

#ifdef __cplusplus
}
#endif