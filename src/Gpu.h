//#pragma once

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
#include "DataCube.h"

void GPU_test_current();

void GPU_test_Gauss_X();

void GPU_test_Gauss_Y();

void GPU_test_Boxcar_Z();

void GPU_test_sdt_dev(float *data, size_t size, size_t cadence, const int range);

void GPU_test_median(float *data, size_t size);

void GPU_test_median_recursive(float *data, size_t size, const size_t *axis_size, const int snRange);

void GPU_test_copy_originalMask();

void GPU_test_flag_sources();

void GPU_test_cpy_msk_1_to_8();

void GPU_test_transpose();

void GPU_test_hist_lone();

void GPU_test_hist(float *data, size_t size, size_t cadence, const int range);

void GPU_test_gausfit(float *data, size_t size, size_t cadence, const int range);

void GPU_test_convolve(float *data, size_t size, const size_t *axis_size);

void GPU_gaufit(const float *data, const size_t size, float *sigma_out, const size_t cadence, const int range);


void GPU_DataCube_filter_flt(char *data, char *maskdata, size_t data_size, const size_t *axis_size, const Array_dbl *kernels_spat, const Array_siz *kernels_spec, const double maskScaleXY, const noise_stat method, const double rms, const size_t cadence, const int range, const double threshold, const int scaleNoise, const noise_stat snStatistic, const int snRange);

// copies data into data_box. Values are set to zero where they are NaN
__global__ void g_copyData_removeBlanks(float *data_box, float *data, const size_t width, const size_t height, const size_t depth);

__global__ void g_addBlanks(float *data_box, float* data, const size_t width, const size_t height, const size_t depth);

// copies data into data_box. While doing this data is masked by the mask data and set to their flux value where the mask is not zero. Values are set to zero where they are NaN
__global__ void g_copyData_setMaskedScale1_removeBlanks(float *data_box, float *data, char *maskData1, const size_t width, const size_t height, const size_t depth, const float value);

// copies data into data_box. While doing this data is masked by the mask data and set to their flux value where the mask is not zero. Values are set to zero where they are NaN
__global__ void g_copyData_setMaskedScale1_removeBlanks_filter_boxcar_Z_flt(float *data_box, float *data, char *maskData1, const uint16_t width, const uint16_t height, const uint16_t depth, const float maskValue, const size_t radius);

__global__ void g_copyData_setMaskedScale1_removeBlanks_filter_gX_bcZ_flt(float *data_box, float *data, char *maskData1, const size_t width, const size_t height, const size_t depth, const float value, const size_t radius_g, const size_t radius_bc, const size_t n_iter);

// copies data into data_box. While doing this data is masked by the mask data and set to their flux value where the mask is not zero. Values are set to zero where they are NaN
__global__ void g_copyData_setMaskedScale8_removeBlanks(float *data_box, float *data, char *maskData8, const size_t width, const size_t height, const size_t depth, const float value);

__global__ void g_maskScale_remove_blanks_flt(float *data, char *mask, const size_t width, const size_t height, const size_t depth, const float value);

// Kernel to apply gaussian filter in x direction. Must be launched in row-wise blocks (so one block per row)
// It is assumed, that the width of the cube is not larger than 12000 entries
__global__ void g_filter_gauss_X_flt(float *data, const size_t width, const size_t height, const size_t depth, const size_t radius, const size_t n_iter);

__global__ void g_filter_gauss_X_flt_new(float *data, const size_t width, const size_t height, const size_t depth, const size_t radius, const size_t n_iter);

// Must be started with a grid Dimension of 1 in x and z direction and use height as y dimension for the grid
// Use only, when gauss filter size is not 0. Behaviour for a gauss filter size of 0 is  not supported. Use "g_copyData_setMaskedScale1_removeBlanks_filter_boxcar_Z_flt" instead
__global__ void g_cpyData_setMskScale1_rmBlnks_fltr_gX_bZ_flt_new(float *data_src, float *data_dst, char *maskData1, const uint16_t width, const uint16_t height, const uint16_t depth, const float maskValue, const uint16_t radius_g, const uint16_t radius_b, const uint16_t n_iter);

// Kernel to apply gaussian filter in y direction.
__global__ void g_filter_gauss_Y_flt(float *data, const size_t width, const size_t height, const size_t depth, const size_t radius, const size_t n_iter);

__global__ void g_filter_gauss_Y_flt_new(float *data, const size_t width, const size_t height, const size_t depth, const size_t radius, const size_t n_iter);

__global__ void g_filter_boxcar_Z_flt(float *data, const size_t width, const size_t height, const size_t depth, const size_t radius);

__global__ void g_filter_boxcar_Z_flt_new(float *data, const size_t width, const size_t height, const size_t depth, const size_t radius);

// Needs to be executed in 32 x 32 thread blocks
__global__ void g_filter_XYZ_flt(float *data, const size_t width, const size_t height, const size_t depth, const size_t copy_depth, const size_t radius_g, const size_t radius_b, const size_t n_iter);

__global__ void g_Mask8(float *data_box, char *maskData8, const size_t width, const size_t height, const size_t depth, const double threshold, float *rms_smooth, const int8_t value);

__global__ void g_Mask1(float *data_box, char *maskData1, const size_t width, const size_t height, const size_t depth, const double threshold, float *rms_smooth);

__global__ void g_find_max_min(const float *data_box, const size_t size, const size_t cadence, float *min_out, float *max_out);

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

__global__ void g_DataCube_copy_mask_1_to_8(char *mask8, char *mask1, size_t width, size_t height, size_t depth);

__global__ void g_DataCube_copy_back_smoothed_cube(char *originalData, float *data, int word_size, size_t width, size_t height, size_t depth);

__global__ void g_DataCube_stat_mad_flt(float *data, float *data_box, size_t width, size_t height, size_t depth, const float value, const size_t cadence, const int range);

__global__ void g_DataCube_stat_mad_flt_2(float *data, float *data_box, size_t size, const float value, const size_t cadence, const int range, const float pivot);

__global__ void g_std_dev_val_flt(float *data, float *data_dst_duo, const size_t size, const float value, const size_t cadence, const int range, const int scaleNoise);

__global__ void g_std_dev_val_flt_final_step(float *data_duo);

__global__ void g_mad_val_flt(float *data, float *data_dst_arr, unsigned int *counter, const size_t size, const size_t max_size, const float value, const size_t cadence, const int range);

__global__ void g_mad_val_flt_final_step(float *data, unsigned int *max_size);

__global__ void g_mad_val_hist_flt(float *data, const size_t size, unsigned int *bins, unsigned int *total_count, const float value, const size_t cadence, const int range, const unsigned int precision, const float *min_flt, const float *max_flt);

__global__ void g_mad_val_hist_flt_cpy_nth_bin(float *data, const size_t size, float *data_box, unsigned int *bins, unsigned int *total_count, unsigned int *counter, const float value, const size_t cadence, const int range, const unsigned int precision, const float *min_flt, const float *max_flt);

__global__ void g_mad_val_hist_flt_final_step(float *data, const unsigned int *sizePtr, unsigned int *total_count, unsigned int *bins);

__global__ void g_mad_val_hist_flt_scale_noise(float *data, const size_t size, const float value, const size_t cadence, const int range, const unsigned int precision, const unsigned int max_val_for_eval);

__global__ void g_create_histogram_flt(const float *data, const size_t size, const int range, unsigned int *histogram, const size_t n_bins, const float *data_min, const float *data_max, const size_t cadence);

__global__ void g_scale_max_min_with_second_moment_flt(const unsigned int *histogram, const size_t n_bins, const int range, float *data_min, float *data_max);

__global__ void g_calc_sigma_flt(const unsigned int *histogram, const size_t n_bins, const int range, const float *data_min, const float *data_max, float *sigma_out);

__global__ void g_gaufit_flt(const float *data, const size_t size, unsigned int *histogram, float *sigma_out, const size_t cadence, const int range, const float *min, const float *max);

__global__ void g_DataCube_transpose_inplace_flt(float *data, const size_t width, const size_t height, const size_t depth);

__global__ void g_test();

__device__ void d_filter_boxcar_1d_flt(float *data, float *data_copy, const size_t size, const size_t filter_radius, const size_t jump);

__device__ void d_filter_chunk_boxcar_1d_flt(float *data, char *originalData, float *data_copy, const size_t startSlice, const size_t size, const size_t filter_radius, const size_t jump, size_t chunk_type);

__device__ inline size_t get_index( const size_t x, const size_t y, const size_t z, const size_t width, const size_t height)
{
	return x + width * (y + height * z);
}

__device__ void d_sort_arr_flt(float *arr, size_t size);

__device__ static float atomicMax(float* address, float val);

__device__ static float atomicMin(float* address, float val);

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