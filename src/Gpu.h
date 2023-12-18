#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdint.h>

#include "common.h"
#include "stddef.h"


#ifdef __cplusplus
extern "C" {
#endif

#include "statistics_flt.h"
#include "Array_dbl.h"
#include "Array_siz.h"
#include "statistics_dbl.h"

void GPU_test_current();

void GPU_test_Gauss_X();

void GPU_test_Gauss_Y();

void GPU_test_Boxcar_Z();

void GPU_test_sdt_dev(float *data, size_t size, size_t cadence, const int range);

void GPU_test_median();

void GPU_test_copy_originalMask();

void GPU_test_flag_sources();

void GPU_DataCube_filter_flt(char *data, char *maskdata, size_t data_size, const size_t *axis_size, const Array_dbl *kernels_spat, const Array_siz *kernels_spec, const double maskScaleXY, const double rms, const size_t cadence, const int range, const double threshold);

// copies data into data_box. Values are set to zero where they are NaN
__global__ void g_copyData_removeBlanks(float *data_box, float *data, const size_t width, const size_t height, const size_t depth);

__global__ void g_addBlanks(float *data_box, float* data, const size_t width, const size_t height, const size_t depth);

// copies data into data_box. While doing this data is masked by the mask data and set to their flux value where the mask is not zero. Values are set to zero where they are NaN
__global__ void g_copyData_setMaskedScale1_removeBlanks(float *data_box, float *data, char *maskData1, const size_t width, const size_t height, const size_t depth, const float value);

// copies data into data_box. While doing this data is masked by the mask data and set to their flux value where the mask is not zero. Values are set to zero where they are NaN
__global__ void g_copyData_setMaskedScale8_removeBlanks(float *data_box, float *data, char *maskData8, const size_t width, const size_t height, const size_t depth, const float value);

__global__ void g_maskScale_remove_blanks_flt(float *data, char *mask, const size_t width, const size_t height, const size_t depth, const float value);

// Kernel to apply gaussian filter in x direction. Must be launched in row-wise blocks (so one block per row)
// It is assumed, that the width of the cube is not larger than 12000 entries
__global__ void g_filter_gauss_X_flt(float *data, const size_t width, const size_t height, const size_t depth, const size_t radius, const size_t n_iter);

// Kernel to apply gaussian filter in y direction. Must be launched in column-wise blocks (so one block per column)
// It is assumed, that the height of the cube is not larger than 12000 entries
__global__ void g_filter_gauss_Y_flt(float *data, const size_t width, const size_t height, const size_t depth, const size_t radius, const size_t n_iter);

__global__ void g_filter_boxcar_Z_flt(float *data, const size_t width, const size_t height, const size_t depth, const size_t radius);

__global__ void g_filter_boxcar_Z_flt_new(float *data, const size_t width, const size_t height, const size_t depth, const size_t radius);

// Needs to be executed in 32 x 32 thread blocks
__global__ void g_filter_XYZ_flt(float *data, const size_t width, const size_t height, const size_t depth, const size_t copy_depth, const size_t radius_g, const size_t radius_b, const size_t n_iter);

__global__ void g_Mask8(float *data_box, char *maskData8, const size_t width, const size_t height, const size_t depth, const double threshold, float *rms_smooth, const int8_t value);

__global__ void g_Mask1(float *data_box, char *maskData1, const size_t width, const size_t height, const size_t depth, const double threshold, float *rms_smooth, const int8_t value);

__global__ void g_FlagSources(float *mask32, uint32_t *bbCounter, uint32_t *bbPtr, const size_t width, const size_t height, const size_t depth, const size_t radius_z);

__global__ void g_MergeSourcesFirstPass(float *mask32, uint32_t *bbPtr, const size_t width, const size_t height, const size_t depth, const size_t radius_x, const size_t radius_y, const size_t radius_z);

void GPU_DataCube_filter(char *data, char *originalData, int word_size, size_t data_size, size_t *axis_size, size_t radiusGauss, size_t n_iter, size_t radiusBoxcar); 

void GPU_DataCube_filter_Chunked(char *data, char *originalData, int word_size, size_t data_size, size_t *axis_size, size_t radiusGauss, size_t n_iter, size_t radiusBoxcar, size_t number_of_chunks); 

void GPU_DataCube_boxcar_filter(char *data, int word_size, size_t data_size, size_t *axis_size, size_t radius);

void GPU_DataCube_gauss_filter(char *data, int word_size, size_t data_size, size_t *axis_size, size_t radius, size_t n_iter);

void GPU_DataCube_copy_mask_8_to_1(char* maskData1, char* maskData8, const size_t *axis_size);


__global__ void g_DataCube_boxcar_filter_flt(float *data, char *originalData, float *data_box, int word_size, const size_t startSlice, size_t width, size_t height, size_t depth, size_t radius, size_t chunck_type);

__global__ void g_DataCube_gauss_filter_XDir(float *data, float *data_box, int word_size, size_t width, size_t height, size_t depth, size_t radius, size_t n_iter);

__global__ void g_DataCube_gauss_filter_YDir(float *data, float *data_box, int word_size, size_t width, size_t height, size_t depth, size_t radius, size_t n_iter);

__global__ void g_DataCube_copy_mask_8_to_1(char *maskData1, char *maskData8, size_t width, size_t height, size_t depth);

__global__ void g_DataCube_copy_back_smoothed_cube(char *originalData, float *data, int word_size, size_t width, size_t height, size_t depth);

__global__ void g_DataCube_stat_mad_flt(float *data, float *data_box, size_t width, size_t height, size_t depth, const float value, const size_t cadence, const int range);

__global__ void g_DataCube_stat_mad_flt_2(float *data, float *data_box, size_t size, const float value, const size_t cadence, const int range, const float pivot);

__global__ void g_std_dev_val_flt(float *data, float *data_dst_duo, const size_t size, const float value, const size_t cadence, const int range);

__global__ void g_std_dev_val_flt_final_step(float *data_duo);


__device__ void d_filter_boxcar_1d_flt(float *data, float *data_copy, const size_t size, const size_t filter_radius, const size_t jump);

__device__ void d_filter_chunk_boxcar_1d_flt(float *data, char *originalData, float *data_copy, const size_t startSlice, const size_t size, const size_t filter_radius, const size_t jump, size_t chunk_type);

__device__ inline size_t get_index( const size_t x, const size_t y, const size_t z, const size_t width, const size_t height)
{
	return x + width * (y + height * z);
}

__device__ void d_sort_arr_flt(float *arr, size_t size);

void sort_arr_flt(float *arr, size_t size);


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