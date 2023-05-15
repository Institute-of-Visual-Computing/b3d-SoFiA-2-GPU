#include "Gpu.h"

void GPU_DataCube_boxcar_filter(char *data, int word_size, size_t data_size, size_t *axis_size, size_t radius)
{
    // Error at start?
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error at start: %s\n", cudaGetErrorString(err));    
    }

    //printf("Start GPU Stuff");
    // Allocate and copy Datacube data onto GPU
    cudaFree(0);
    int deviceCount;
    //printf("wordSize: %i dataSize: %lu charSize: %lu\n", word_size, data_size, sizeof(char));
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        exit(0);
    }

    int device = 0;  // Assuming you want to query the first CUDA device
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    
    //printf("Maximum block size: %d\n", deviceProp.maxThreadsPerBlock);
    //printf("Maximum grid size: (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);

    char *d_data;
    float *spectrum;
    float *data_box;

    err = cudaGetLastError();
    printf("%s\n", cudaGetErrorString(err));

    err = cudaMalloc((void**)&d_data, data_size * word_size * sizeof(char));
    if (err != cudaSuccess)
    {
        printf("%s\n", cudaGetErrorString(err));
        exit(0);
    }
    
    err = cudaMalloc((void**)&spectrum, data_size * sizeof(float));
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
    g_DataCube_boxcar_filter<<<gridSize, blockSize>>>(d_data, spectrum, data_box, word_size, axis_size[0], axis_size[1], axis_size[2], radius);

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
    cudaFree(spectrum);
    cudaFree(data_box);

    // Error after free mem??
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after freeing memory: %s\n", cudaGetErrorString(err));    
    }

    
}

__global__ void g_DataCube_boxcar_filter(char *data, float *spectrum, float *data_box, int word_size, size_t width, size_t height, size_t depth, size_t radius)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        spectrum = spectrum + x * depth + (y * depth * width);
        data_box = data_box + x * (depth + radius * 2) + (y * (depth + radius * 2) * width);

        for(size_t z = depth; z--;) *(spectrum + z) = (double)(*((float *)(data + get_index(x, y, z, width, height) * word_size)));

        d_filter_boxcar_1d_flt(spectrum, data_box, depth, radius);

        for(size_t z = depth; z--;) *((float *)(data + get_index(x, y, z, width, height) * word_size))   = (float)*(spectrum + z);
    }
}

__device__ void d_filter_boxcar_1d_flt(float *data, float *data_copy, const size_t size, const size_t filter_radius)
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
		for(size_t i = n_iter; i--;) d_filter_boxcar_1d_flt(ptr, data_row, size_x, filter_radius);
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
		for(size_t i = n_iter; i--;) d_filter_boxcar_1d_flt(data_copy, data_col, size_y, filter_radius);
		
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