#include "Gpu.h"

void GPU_DataCube_boxcar_filter(char *data, int word_size, size_t data_size, size_t *axis_size, size_t radius)
{
    // Error at start?
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error at start: %s\n", cudaGetErrorString(err));    
    }

    // check for CUDA capable device
    cudaFree(0);
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        exit(0);
    }

    // Allocate and copy Datacube data onto GPU
    float *d_data;
    float *data_box;

    err = cudaGetLastError();
    printf("%s\n", cudaGetErrorString(err));

    err = cudaMalloc((void**)&d_data, data_size * word_size * sizeof(char));
    if (err != cudaSuccess)
    {
        printf("%s\n", cudaGetErrorString(err));
        exit(0);
    }

    err = cudaMalloc((void**)&data_box, (data_size + axis_size[0] * axis_size[1] * 2 * radius) * sizeof(float));
    if (err != cudaSuccess)
    {
        printf("%s\n", cudaGetErrorString(err));
        exit(0);
    }

    cudaMemcpy(d_data, data, word_size * data_size * sizeof(char), cudaMemcpyHostToDevice);

    // Error after mem copy?
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error at mem Copy to device: %s\n", cudaGetErrorString(err));    
    }

    dim3 blockSize(32,32);
    dim3 gridSize((axis_size[0] + blockSize.x - 1) / blockSize.x,
                  (axis_size[1] + blockSize.y - 1) / blockSize.y);

    // Error before Kernel Launch?
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error before kernel launch: %s\n", cudaGetErrorString(err));    
    }

    // Launch CUDA Kernel
    cudaDeviceSynchronize();
    g_DataCube_boxcar_filter<<<gridSize, blockSize>>>(d_data, data_box, word_size, axis_size[0], axis_size[1], axis_size[2], radius);

    cudaDeviceSynchronize();

    // Error after Kernel Launch?
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after kernel launch: %s\n", cudaGetErrorString(err));    
    }

    cudaMemcpy(data, d_data, word_size * data_size * sizeof(char), cudaMemcpyDeviceToHost);
    // Error after backkcopy??
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after backcopy: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_data);
    cudaFree(data_box);

    // Error after free mem??
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after freeing memory: %s\n", cudaGetErrorString(err));    
    }
}

void GPU_DataCube_gauss_filter(char *data, int word_size, size_t data_size, size_t *axis_size, size_t radius, size_t n_iter)
{
printf("N-Iter: %lu\n", n_iter);

    // Error at start?
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error at start: %s\n", cudaGetErrorString(err));    
    }

    // check for CUDA capable device
    cudaFree(0);
    int deviceCount;
    //printf("wordSize: %i dataSize: %lu charSize: %lu\n", word_size, data_size, sizeof(char));
    //printf("Size per char: %lu, Size per word: %lu, Size per float: %lu\n", sizeof(char), sizeof(char) * word_size, sizeof(double));
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        exit(0);
    }

    // Allocate and copy Datacube data onto GPU
    float *d_data;
    float *data_box;

    err = cudaGetLastError();
    printf("%s\n", cudaGetErrorString(err));

    err = cudaMalloc((void**)&d_data, data_size * word_size * sizeof(char));
    if (err != cudaSuccess)
    {
        printf("%s\n", cudaGetErrorString(err));
        exit(0);
    }

    size_t x_overlap = axis_size[1] * axis_size[2] * 2 * radius;
    size_t y_overlap = axis_size[0] * axis_size[2] * 2 * radius;


    err = cudaMalloc((void**)&data_box, (data_size + (x_overlap > y_overlap ? x_overlap : y_overlap)) * sizeof(float));
    if (err != cudaSuccess)
    {
        printf("%s\n", cudaGetErrorString(err));
        exit(0);
    }

    cudaMemcpy(d_data, data, word_size * data_size * sizeof(char), cudaMemcpyHostToDevice);

    // Error after mem copy?
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error at mem Copy to device: %s\n", cudaGetErrorString(err));    
    }

    // Gauss Filter in X Direction
    dim3 blockSizeX(16,16);
    dim3 gridSizeX((axis_size[2] + blockSizeX.x - 1) / blockSizeX.x,
                  (axis_size[1] + blockSizeX.y - 1) / blockSizeX.y);

    // Error before Kernel Launch?
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error before Xkernel launch: %s\n", cudaGetErrorString(err));    
    }

    g_DataCube_gauss_filter_XDir<<<gridSizeX, blockSizeX>>>(d_data, data_box, word_size, axis_size[0], axis_size[1], axis_size[2], radius, n_iter);

    cudaDeviceSynchronize();

    // Error after Kernel Launch?
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after Xkernel launch: %s\n", cudaGetErrorString(err));    
    }

    // Gauss Filter in Y Direction
    dim3 blockSizeY(16,16);
    dim3 gridSizeY((axis_size[2] + blockSizeY.x - 1) / blockSizeY.x,
                  (axis_size[1] + blockSizeY.y - 1) / blockSizeY.y);

    // Error before Kernel Launch?
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error before Ykernel launch: %s\n", cudaGetErrorString(err));    
    }

    cudaDeviceSynchronize();

    g_DataCube_gauss_filter_YDir<<<gridSizeY, blockSizeY>>>(d_data, data_box, word_size, axis_size[0], axis_size[1], axis_size[2], radius, n_iter);

    cudaDeviceSynchronize();

    // Error after Kernel Launch?
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after Ykernel launch: %s\n", cudaGetErrorString(err));    
    }

    cudaMemcpy(data, d_data, word_size * data_size * sizeof(char), cudaMemcpyDeviceToHost);
    // Error after backkcopy??
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after backcopy: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_data);
    cudaFree(data_box);

    // Error after free mem??
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after freeing memory: %s\n", cudaGetErrorString(err));    
    }
}

__global__ void g_DataCube_boxcar_filter(float *data, float *data_box, int word_size, size_t width, size_t height, size_t depth, size_t radius)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t jump = width * height;

    if (x < width && y < height)
    {
        data = data + (x + y * width);
        data_box = data_box + (x + y * width);

        d_filter_boxcar_1d_flt(data, data_box, depth, radius, jump);
    }
}

__global__ void g_DataCube_gauss_filter_XDir(float *data, float *data_box, int word_size, size_t width, size_t height, size_t depth, size_t radius, size_t n_iter)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < depth && y < height)
    {
        data = data + (y * width + x * width * height);
        data_box = data_box + y * (width + 2 * radius) 
                            + x * (width + 2 * radius) * height;

        for(size_t i = n_iter; i--;) d_filter_boxcar_1d_flt(data, data_box, width, radius, 1);
    }
}

__global__ void g_DataCube_gauss_filter_YDir(float *data, float *data_box, int word_size, size_t width, size_t height, size_t depth, size_t radius, size_t n_iter)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < depth && y < width)
    {
        data = data + x * width * height + y;
        data_box = data_box + x * width * (height + 2 * radius) 
                            + y;

        for(size_t i = n_iter; i--;) d_filter_boxcar_1d_flt(data, data_box, height, radius, width);
    }
}


__device__ void d_filter_boxcar_1d_flt(float *data, float *data_copy, const size_t size, const size_t filter_radius, const size_t jump)
{
    // Define filter size
	const size_t filter_size = 2 * filter_radius + 1;
	const float inv_filter_size = 1.0 / filter_size;
	size_t i;
	
	// Make copy of data, taking care of NaN
	for(i = size; i--;) data_copy[(filter_radius + i) * jump] = FILTER_NAN(data[i * jump]);
	
	// Fill overlap regions with 0
	for(i = filter_radius; i--;) data_copy[i * jump] = data_copy[(size + filter_radius + i) * jump] = 0.0;
	
	// Apply boxcar filter to last data point
	data[(size - 1) * jump] = 0.0;
	for(i = filter_size; i--;) data[(size - 1) * jump] += data_copy[(size + i - 1) * jump];
	data[(size - 1) * jump] *= inv_filter_size;
	
	// Recursively apply boxcar filter to all previous data points
	for(i = size - 1; i--;) data[i * jump] = data[(i + 1) * jump] + (data_copy[i * jump] - data_copy[(filter_size + i) * jump]) * inv_filter_size;
	
	return;
}

__device__ void d_filter_boxcar_1d_flt_YDir(float *data, float *data_copy, const size_t size, const size_t filter_radius, const size_t size_y)
{
    // Define filter size
	const size_t filter_size = 2 * filter_radius + 1;
	const float inv_filter_size = 1.0 / filter_size;
	size_t i;
	
	// Make copy of data, taking care of NaN
	for(i = size; i--;) data_copy[filter_radius + i] = FILTER_NAN(data[i]);
	
	// Fill overlap regions with 0
	for(i = filter_radius; i--;) data_copy[i] = data_copy[size + filter_radius + i] = 0.0;
	
	// Apply boxcar filter to last data point
	data[size - 1] = 0.0;
	for(i = filter_size; i--;) data[size - 1] += data_copy[size + i - 1];
	data[size - 1] *= inv_filter_size;
	
	// Recursively apply boxcar filter to all previous data points
	for(i = size - 1; i--;) data[i] = data[i + 1] + (data_copy[i] - data_copy[filter_size + i]) * inv_filter_size;
	
	return;
}

__device__ void d_filter_boxcar_1d_flt_ZDir(float *data, float *data_copy, const size_t size, const size_t filter_radius, const size_t size_z)
{
    // Define filter size
	const size_t filter_size = 2 * filter_radius + 1;
	const float inv_filter_size = 1.0 / filter_size;
	size_t i;
	
	// Make copy of data, taking care of NaN
	for(i = size; i--;) data_copy[filter_radius + i] = FILTER_NAN(data[i]);
	
	// Fill overlap regions with 0
	for(i = filter_radius; i--;) data_copy[i] = data_copy[size + filter_radius + i] = 0.0;
	
	// Apply boxcar filter to last data point
	data[size - 1] = 0.0;
	for(i = filter_size; i--;) data[size - 1] += data_copy[size + i - 1];
	data[size - 1] *= inv_filter_size;
	
	// Recursively apply boxcar filter to all previous data points
	for(i = size - 1; i--;) data[i] = data[i + 1] + (data_copy[i] - data_copy[filter_size + i]) * inv_filter_size;
	
	return;
}

__device__ void d_filter_gauss_2d_flt(float *data, float *data_copy, float *data_row, float *data_col, const size_t size_x, const size_t size_y, const size_t n_iter, const size_t filter_radius)
{
	// Set up a few variables
	const size_t size_xy = size_x * size_y;
	float *ptr = data + size_xy;
	float *ptr2;
	
	// Run row filter (along x-axis)
	// This is straightforward, as the data are contiguous in x.
	while(ptr > data)
	{
		ptr -= size_x;
		for(size_t i = n_iter; i--;) d_filter_boxcar_1d_flt(ptr, data_row, size_x, filter_radius, 1);
	}
	
	// Run column filter (along y-axis)
	// This is more complicated, as the data are non-contiguous in y.
	for(size_t x = size_x; x--;)
	{
		// Copy data into column array
		ptr = data + size_xy - size_x + x;
		ptr2 = data_copy + size_y;
		while(ptr2 --> data_copy)
		{
			*ptr2 = *ptr;
			ptr -= size_x;
		}
		
		// Apply filter
		for(size_t i = n_iter; i--;) d_filter_boxcar_1d_flt(data_copy, data_col, size_y, filter_radius, 1);
		
		// Copy column array back into data array
		ptr = data + size_xy - size_x + x;
		ptr2 = data_copy + size_y;
		while(ptr2 --> data_copy)
		{
			*ptr = *ptr2;
			ptr -= size_x;
		}
	}
	
	return;
}