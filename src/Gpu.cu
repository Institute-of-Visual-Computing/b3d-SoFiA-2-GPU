#include "Gpu.h"

void GPU_DataCube_boxcar_filter(char *data, int word_size, size_t data_size, size_t *axis_size, size_t radius)
{
    //printf("Start GPU Stuff");
    // Allocate and copy Datacube data onto GPU
    cudaFree(0);
    int deviceCount;
    printf("wordSize: %i dataSize: %lu charSize: %lu\n", word_size, data_size, sizeof(char));
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        exit(0);
    }
    char *d_data;
    printf("%lu\n", (data_size * word_size * sizeof(char)));
    cudaError_t err = cudaGetLastError();
    printf("%s\n", cudaGetErrorString(err));
    err = cudaMalloc((void**)&d_data, data_size * word_size * sizeof(char));
    if (err != cudaSuccess)
    {
        printf("%s\n", cudaGetErrorString(err));
        exit(0);
    }
    

    cudaMemcpy(d_data, data, word_size * data_size, cudaMemcpyHostToDevice);

    dim3 blockSize(64,64);
    dim3 gridSize((axis_size[0] + blockSize.x - 1) / blockSize.x,
                  (axis_size[1] + blockSize.y - 1) / blockSize.y);

    // Launch CUDA Kernel
    cudaDeviceSynchronize();
    g_DataCube_boxcar_filter<<<gridSize, blockSize>>>(d_data, word_size, axis_size[0], axis_size[1], axis_size[2], radius);

    cudaDeviceSynchronize();
    cudaMemcpy(data, d_data, word_size * data_size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    //printf("GPU Stuff Finished");
}

__global__ void g_DataCube_boxcar_filter(char *data, int word_size, size_t width, size_t height, size_t depth, size_t radius)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        // float *spectrum = (float *)malloc(depth * sizeof(float));
        // float  *data_box = (float *) malloc((depth + 2 * radius) * sizeof(float));

        float *spectrum;
        float *data_box;

        cudaMalloc((void**)&spectrum, depth * sizeof(float));
        cudaMalloc((void**)&data_box, (depth + 2 * radius) * sizeof(float));

        for(size_t z = depth; z--;) *(spectrum + z) = (double)(*((float *)(data + get_index(x, y, z, width, height) * word_size)));

        d_filter_boxcar_1d_flt(spectrum, data_box, depth, radius);

        for(size_t z = depth; z--;) *((float *)(data + get_index(x, y, z, width, height) * word_size))   = (float)*(spectrum + z);

        cudaFree(spectrum);
        cudaFree(data_box);
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