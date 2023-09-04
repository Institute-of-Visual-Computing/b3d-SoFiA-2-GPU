#include "Gpu.h"

void GPU_test_current()
{
    int arraySize = 32;
    float data[arraySize] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    u_int8_t mask[(int)ceil(arraySize / 8.0f)] = { 0b10101010, 0b01010101, 0b10101010, 0b01010101};

    printf("Array to test: ");

	for (int i = 0 ; i < arraySize; i++){printf("%f ", data[i]);}

	printf("\n");

    float *d_data;
    float *d_data_box;
    char *d_mask;

    cudaMalloc((void**)&d_data, arraySize * sizeof(float));
    cudaMalloc((void**)&d_data_box, arraySize * sizeof(float));
    cudaMalloc((void**)&d_mask, ceil(arraySize / 8.0f) * sizeof(char));

    cudaMemcpy(d_data, data, arraySize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, ceil(arraySize / 8.0f) * sizeof(char), cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Cuda error at start: %s\n", cudaGetErrorString(err));  
    }

    dim3 blockSize(2,1);
    dim3 gridSize(1,1);

    g_copyData_setMaskedScale1_removeBlanks<<<blockSize, gridSize>>>(d_data_box, d_data, d_mask, 16, 1, 2, 20);

    cudaDeviceSynchronize();
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Cuda error at start: %s\n", cudaGetErrorString(err));  
    }

    cudaMemcpy(data, d_data_box, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Array to test: ");

	for (int i = 0 ; i < arraySize; i++){printf("%f ", data[i]);}

	printf("\n");
}

void GPU_test_Gauss_X()
{
    float data[10] = {1,2,3,4,5,6,7,8,9,10};

    printf("Array to get median: ");

	for (int i = 0 ; i < 10; i++){printf("%f ", data[i]);}

	printf("\n");

    float *d_data;
    float *d_data_box;

    cudaMalloc((void**)&d_data, 10 * sizeof(float));
    cudaMalloc((void**)&d_data_box, 10 * sizeof(float));

    cudaMemset(d_data_box, 0, 10 * sizeof(float));

    cudaMemcpy(d_data, data, 10 * sizeof(float), cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Cuda error at start: %s\n", cudaGetErrorString(err));  
    }

    cudaDeviceSynchronize();

    dim3 blockSize(8,1);
    dim3 gridSize(1,2);

    g_filter_gauss_X_flt<<<gridSize, blockSize, 29 * sizeof(float)>>>(d_data, 5, 1, 2, 1, 5);

    cudaDeviceSynchronize();

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Cuda error at start: %s\n", cudaGetErrorString(err));  
    }

    cudaMemcpy(data, d_data, 10 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    printf("Array to get median: ");

	for (int i = 0 ; i < 10; i++){printf("%f ", data[i]);}

	printf("\n");

    cudaFree(d_data);
    cudaFree(d_data_box);

}

void GPU_test_Gauss_Y()
{
    float data[10] = {1,2,3,4,5,6,7,8,9,10};

    printf("Array to get median: ");

	for (int i = 0 ; i < 10; i++){printf("%f ", data[i]);}

	printf("\n");

    float *d_data;
    float *d_data_box;

    cudaMalloc((void**)&d_data, 10 * sizeof(float));
    cudaMalloc((void**)&d_data_box, 10 * sizeof(float));

    cudaMemset(d_data_box, 0, 10 * sizeof(float));

    cudaMemcpy(d_data, data, 10 * sizeof(float), cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Cuda error at start: %s\n", cudaGetErrorString(err));  
    }

    cudaDeviceSynchronize();

    dim3 blockSize(1,4);
    dim3 gridSize(5,1);

    g_filter_gauss_Y_flt<<<gridSize, blockSize, 29 * sizeof(float)>>>(d_data, 5, 2, 1, 1, 3);

    cudaDeviceSynchronize();

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Cuda error at start: %s\n", cudaGetErrorString(err));  
    }

    cudaMemcpy(data, d_data, 10 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    printf("Array to get median: ");

	for (int i = 0 ; i < 10; i++){printf("%f ", data[i]);}

	printf("\n");

    cudaFree(d_data);
    cudaFree(d_data_box);

}

void GPU_test_Boxcar_Z()
{
    float data[10] = {1,2,3,4,5,6,7,8,9,10};

    printf("Array to get median: ");

	for (int i = 0 ; i < 10; i++){printf("%f ", data[i]);}

	printf("\n");

    float *d_data;
    float *d_data_box;

    cudaMalloc((void**)&d_data, 10 * sizeof(float));
    cudaMalloc((void**)&d_data_box, 10 * sizeof(float));

    cudaMemset(d_data_box, 0, 10 * sizeof(float));

    cudaMemcpy(d_data, data, 10 * sizeof(float), cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Cuda error at start: %s\n", cudaGetErrorString(err));  
    }

    cudaDeviceSynchronize();

    dim3 blockSize(1,1,2);
    dim3 gridSize(5,1);

    g_filter_boxcar_Z_flt<<<gridSize, blockSize, 7 * sizeof(float)>>>(d_data, 5, 1, 2, 1);

    cudaDeviceSynchronize();

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Cuda error at start: %s\n", cudaGetErrorString(err));  
    }

    cudaMemcpy(data, d_data, 10 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    printf("Array to get median: ");

	for (int i = 0 ; i < 10; i++){printf("%f ", data[i]);}

	printf("\n");

    cudaFree(d_data);
    cudaFree(d_data_box);

}

void GPU_test_sdt_dev(float *data, size_t size, size_t cadence, const int range)
{
    printf("Ref: %.3e\n", std_dev_val_flt(data, size, 0, cadence, range));

    float *d_data;
    float *d_data_box;

    cudaMalloc((void**)&d_data, size * sizeof(float));
    cudaMalloc((void**)&d_data_box, 2 * sizeof(float));

    cudaMemset(d_data_box, 0, 2 * sizeof(float));

    cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Cuda error at start: %s\n", cudaGetErrorString(err));  
    }

    cudaDeviceSynchronize();

    dim3 blockSize(1024);
    dim3 gridSize(1024);

    g_std_dev_val_flt<<<gridSize, blockSize, blockSize.x * 2 * sizeof(float)>>>(d_data, d_data_box, size, 0, cadence, range);

    cudaDeviceSynchronize();

    g_std_dev_val_flt_final_step<<<1,1>>>(d_data_box);

    cudaDeviceSynchronize();

    float noise[2] = {0,0};
    cudaMemcpy(noise, d_data_box, 2 * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0 ; i < 2; i++){printf("noise: %.3e\n", noise[i]);;}

    cudaFree(d_data);
    cudaFree(d_data_box);
}

void GPU_test_median()
{
    float data[10] = {81,8,43,4,20,1,13,7,12,9};

    printf("Array to get median: ");

	for (int i = 0 ; i < 10; i++){printf("%f ", data[i]);}

	printf("\n");

    float *d_data;
    float *d_data_box;

    cudaMalloc((void**)&d_data, 10 * sizeof(float));
    cudaMalloc((void**)&d_data_box, 10 * sizeof(float));

    cudaMemset(d_data_box, 0, 10 * sizeof(float));

    cudaMemcpy(d_data, data, 10 * sizeof(float), cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Cuda error at start: %s\n", cudaGetErrorString(err));  
    }

    cudaDeviceSynchronize();

    dim3 blockSize(2);
    dim3 gridSize(1);

    g_std_dev_val_flt<<<gridSize, blockSize, blockSize.x * 2 * sizeof(float)>>>(d_data, d_data_box, 10, 0, 4, 0);

    cudaDeviceSynchronize();

    g_std_dev_val_flt_final_step<<<1,1>>>(d_data_box);

    cudaDeviceSynchronize();

    g_DataCube_stat_mad_flt<<<gridSize, blockSize, blockSize.x * 14 * sizeof(float)>>>(d_data, d_data_box, 10, 1, 1, 0, 1, 0);

    cudaDeviceSynchronize();

    cudaMemcpy(data, d_data_box, 10 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    printf("DataBox: ");

	for (int i = 0 ; i < 10; i++){printf("%f ", data[i]);}

	printf("\n");
}

void GPU_DataCube_filter_flt(char *data, char *maskdata, size_t data_size, const size_t *axis_size, const Array_dbl *kernels_spat, const Array_siz *kernels_spec, const double maskScaleXY, const double rms, const size_t cadence, const int range, const double threshold)
{
    printf("Starting GPU\n");

    // Error at start?
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error at start: %s\n", cudaGetErrorString(err));    
    }

    size_t n_iter;
    size_t radius_gauss;
    const double FWHM_CONST = 2.0 * sqrt(2.0 * log(2.0));

    size_t radius_boxcar;

    // check for CUDA capable device
    //cudaFree(0);
    int deviceCount;
    cudaDeviceProp prop;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        exit(0);
    }

    size_t width = axis_size[0];
    size_t height = axis_size[1];
    size_t depth = axis_size[2];

    // Define memory on GPU
    float *d_data;
    float *d_data_box;
    float *d_data_duo;
    char *d_mask_data;
    char *d_original_mask;

    size_t free_bytes, total_bytes;
    cudaError_t cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);

    if (cuda_status == cudaSuccess) {
        printf("Total GPU Memory: %fMB\n", total_bytes / (1024.0f * 1024.0f));
        printf("Free GPU Memory: %fMB\n", free_bytes / (1024.0f * 1024.0f));
    } else {
        printf("cudaMemGetInfo failed: %s\n", cudaGetErrorString(cuda_status));
    }

    // Allocate and copy values from Host to Device
    printf("Allocating %fMB for the data\n", data_size * sizeof(float) / (1024.0f * 1024.0f));
    cudaMalloc((void**)&d_data, data_size * sizeof(float));

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after malloc data: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);

    if (cuda_status == cudaSuccess) {
        printf("Total GPU Memory: %fMB\n", total_bytes / (1024.0f * 1024.0f));
        printf("Free GPU Memory: %fMB\n", free_bytes / (1024.0f * 1024.0f));
    } else {
        printf("cudaMemGetInfo failed: %s\n", cudaGetErrorString(cuda_status));
    }

    printf("Allocating %fMB for the box\n", data_size * sizeof(float) / (1024.0f * 1024.0f));
    cudaMalloc((void**)&d_data_box, data_size * sizeof(float));

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after malloc box: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);

    if (cuda_status == cudaSuccess) {
        printf("Total GPU Memory: %fMB\n", total_bytes / (1024.0f * 1024.0f));
        printf("Free GPU Memory: %fMB\n", free_bytes / (1024.0f * 1024.0f));
    } else {
        printf("cudaMemGetInfo failed: %s\n", cudaGetErrorString(cuda_status));
    }

    printf("Allocating %fMB for the duo\n", 2 * sizeof(float) / (1024.0f * 1024.0f));
    cudaMalloc((void**)&d_data_duo, 2 * sizeof(float));

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after malloc duo: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);

    if (cuda_status == cudaSuccess) {
        printf("Total GPU Memory: %fMB\n", total_bytes / (1024.0f * 1024.0f));
        printf("Free GPU Memory: %fMB\n", free_bytes / (1024.0f * 1024.0f));
    } else {
        printf("cudaMemGetInfo failed: %s\n", cudaGetErrorString(cuda_status));
    }

    printf("Allocating %fMB for the mask\n", data_size * sizeof(char) / (1024.0f * 1024.0f));
    cudaMalloc((void**)&d_original_mask, data_size * sizeof(char));

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after malloc mask: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);

    if (cuda_status == cudaSuccess) {
        printf("Total GPU Memory: %fMB\n", total_bytes / (1024.0f * 1024.0f));
        printf("Free GPU Memory: %fMB\n", free_bytes / (1024.0f * 1024.0f));
    } else {
        printf("cudaMemGetInfo failed: %s\n", cudaGetErrorString(cuda_status));
    }

    cudaMemcpy(d_data, data, data_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_box, data, data_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_original_mask, maskdata, data_size * sizeof(char), cudaMemcpyHostToDevice);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after copy: %s\n", cudaGetErrorString(err));    
    }

    // Set up Bitmask space efficient on GPU one byte handles 8 entries in from the cube,
    // since here we only need to mask pixels, not differ between individual sources
    size_t d_mask_size = ceil(width / 8.0f) * height * depth * sizeof(char);
	cudaMalloc((void**)&d_mask_data, d_mask_size);
    cudaMemset(d_mask_data, 0, d_mask_size);

    //GPU_DataCube_copy_mask_8_to_1(d_mask_data, d_original_mask, axis_size);


    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after seting up mask data: %s\n", cudaGetErrorString(err));    
    }

    dim3 blockSizeMS(32,32);
    dim3 gridSizeMS((width + blockSizeMS.x - 1) / blockSizeMS.x,
                    (height + blockSizeMS.y - 1) / blockSizeMS.y);

    dim3 blockSizeX(1024,1);
    dim3 gridSizeX(1,height);

    dim3 blockSizeY(1,1024);
    dim3 gridSizeY(width,1);

    dim3 blockSizeZ(1024,1);
    dim3 gridSizeZ(width,1);

    dim3 blockSizeNoise(1024);
    dim3 gridSizeNoise(1024);

    for(size_t i = 0; i < Array_dbl_get_size(kernels_spat); ++i)
	{
        for(size_t j = 0; j < Array_siz_get_size(kernels_spec); ++j)
		{
            cudaMemset(d_data_duo, 0, 2 * sizeof(float));

            if (Array_dbl_get(kernels_spat, i) || Array_siz_get(kernels_spec, j))
            {
                optimal_filter_size_dbl(Array_dbl_get(kernels_spat, i) / FWHM_CONST, &radius_gauss, &n_iter);
                radius_boxcar = Array_siz_get(kernels_spec, j) / 2;

                printf("[%.1f] x [%lu]\n", Array_dbl_get(kernels_spat, i), Array_siz_get(kernels_spec, j));

                // Copy maskScaled data from d_data to d_data_box and replace blanks
                if (maskScaleXY >= 0.0)
                {
                    printf("Starting Kernels\n");

                    g_copyData_setMaskedScale8_removeBlanks<<<gridSizeMS, blockSizeMS>>>(d_data_box, d_data, d_original_mask, width, height, depth, maskScaleXY * rms);
                    cudaDeviceSynchronize();

                    err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        printf("Cuda error after Mask Kernel: %s\n", cudaGetErrorString(err));    
                    }
                }
                else
                {
                    g_copyData_removeBlanks<<<gridSizeMS, blockSizeMS>>>(d_data_box, d_data, width, height, depth);
                    cudaDeviceSynchronize();
                }
                
                if(radius_gauss > 0)
                {
                    printf("Launching Gauss X…\n");

                    g_filter_gauss_X_flt<<<gridSizeX, blockSizeX, (radius_gauss * 3 + width * 2) * sizeof(float)>>>(d_data_box, width, height, depth, radius_gauss, n_iter);
                    cudaDeviceSynchronize();

                    err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        printf("Cuda error after X Kernel: %s\n", cudaGetErrorString(err));    
                    }
                }
                    
                if(radius_gauss > 0)
                {
                    printf("Launching Gauss Y…\n");

                    g_filter_gauss_Y_flt<<<gridSizeY, blockSizeY, (radius_gauss * 3 + height * 2) * sizeof(float)>>>(d_data_box, width, height, depth, radius_gauss, n_iter);
                    cudaDeviceSynchronize();

                    err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        printf("Cuda error after Y Kernel: %s\n", cudaGetErrorString(err));    
                    }
                }

                if(radius_boxcar > 0) 
                {
                    printf("Launching Boxcar Z…\n");

                    g_filter_boxcar_Z_flt<<<gridSizeZ, blockSizeZ, (radius_boxcar * 3 + depth * 2) * sizeof(float)>>>(d_data_box, width, height, depth, radius_boxcar);
                    cudaDeviceSynchronize();

                    err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        printf("Cuda error after Z Kernel: %s\n", cudaGetErrorString(err));    
                    }
                }

                printf("Starting Kernels for Mask\n");
                g_addBlanks<<<gridSizeMS, blockSizeMS>>>(d_data_box, d_data, width, height, depth);

                g_std_dev_val_flt<<<gridSizeNoise, blockSizeNoise, 1024 * 2 * sizeof(float)>>>(d_data_box, d_data_duo, data_size, 0.0f, cadence, range);
                cudaDeviceSynchronize();

                float noise[2] = {0,0};
                cudaMemcpy(noise, d_data_duo, 2 * sizeof(float), cudaMemcpyDeviceToHost);

                printf("noise: %.3e\n", noise[0]);
                printf("Count: %.3e\n", noise[1]);

                g_std_dev_val_flt_final_step<<<1,1>>>(d_data_duo);
                cudaDeviceSynchronize();

                cudaMemcpy(noise, d_data_duo, 2 * sizeof(float), cudaMemcpyDeviceToHost);

                printf("Final noise: %.3e\n\n", noise[0]);

                g_Mask8<<<gridSizeMS, blockSizeMS>>>(d_data_box, d_original_mask, width, height, depth, threshold, d_data_duo, 1);

                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    printf("Cuda error after noise calc: %s\n", cudaGetErrorString(err));
                }
            }
            else
            {
                g_std_dev_val_flt<<<gridSizeNoise, blockSizeNoise, 1024 * 2 * sizeof(float)>>>(d_data, d_data_duo, data_size, 0.0f, cadence, range);
                cudaDeviceSynchronize();
                g_std_dev_val_flt_final_step<<<1,1>>>(d_data_duo);
                cudaDeviceSynchronize();

                float noise[2] = {0,0};
                cudaMemcpy(noise, d_data_duo, 2 * sizeof(float), cudaMemcpyDeviceToHost);

                printf("noise: %.3e\n\n", noise[0]);

                g_Mask8<<<gridSizeMS, blockSizeMS>>>(d_data, d_original_mask, width, height, depth, threshold, d_data_duo, 1);

                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    printf("Cuda error after noise calc: %s\n", cudaGetErrorString(err));    
                }
            }
        }
    }

    cudaMemcpy(maskdata, d_original_mask, data_size * sizeof(char), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(d_data);
    cudaFree(d_data_box);
    cudaFree(d_mask_data);
    cudaFree(d_original_mask);
    cudaFree(d_data_duo);
}

__global__ void g_copyData_removeBlanks(float *data_box, float *data, const size_t width, const size_t height, const size_t depth)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {return;}

    size_t index = x + y * width;

    while (index < width * height * depth)
    {
        data_box[index] = FILTER_NAN(data[index]);
        index += width * height;
    }
}

__global__ void g_addBlanks(float *data_box, float* data, const size_t width, const size_t height, const size_t depth)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {return;}

    size_t index = x + y * width;

    while (index < width * height * depth)
    {
        if(IS_NAN(data[index])) data_box[index] = NAN;
        index += width * height;
    }
}

__global__ void g_copyData_setMaskedScale1_removeBlanks(float *data_box, float *data, char *maskData1, const size_t width, const size_t height, const size_t depth, const float value)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t maskWidth = ceil(width / 8.0f);
    size_t index = y * maskWidth + x;
    size_t page_size = width * height;
    size_t mask_page_size = maskWidth * height;

    if (x >= maskWidth || y >= height) {return;}

    size_t pageNumber = 0;
    while (index < mask_page_size * depth)
    {
        if (x == maskWidth - 1)
        {
            for (int i = 0; x * 8 + i < width; ++i)
            {
                data_box[index * 8 + i] = (*(maskData1 + index) & (1 << (7 - i))) ? copysign(value, data[index * 8 + i]) : data[index * 8 + i];
            }
        }
        else
        {
            data_box[x * 8 + y * width + 0] = (*(maskData1 + index) & (1 << 7)) ? copysign(value, data[x * 8 + y * width + 0]) : FILTER_NAN(data[x * 8 + y * width + 0]);
            data_box[x * 8 + y * width + 1] = (*(maskData1 + index) & (1 << 6)) ? copysign(value, data[x * 8 + y * width + 1]) : FILTER_NAN(data[x * 8 + y * width + 1]);
            data_box[x * 8 + y * width + 2] = (*(maskData1 + index) & (1 << 5)) ? copysign(value, data[x * 8 + y * width + 2]) : FILTER_NAN(data[x * 8 + y * width + 2]);
            data_box[x * 8 + y * width + 3] = (*(maskData1 + index) & (1 << 4)) ? copysign(value, data[x * 8 + y * width + 3]) : FILTER_NAN(data[x * 8 + y * width + 3]);
            data_box[x * 8 + y * width + 4] = (*(maskData1 + index) & (1 << 3)) ? copysign(value, data[x * 8 + y * width + 4]) : FILTER_NAN(data[x * 8 + y * width + 4]);
            data_box[x * 8 + y * width + 5] = (*(maskData1 + index) & (1 << 2)) ? copysign(value, data[x * 8 + y * width + 5]) : FILTER_NAN(data[x * 8 + y * width + 5]);
            data_box[x * 8 + y * width + 6] = (*(maskData1 + index) & (1 << 1)) ? copysign(value, data[x * 8 + y * width + 6]) : FILTER_NAN(data[x * 8 + y * width + 6]);
            data_box[x * 8 + y * width + 7] = (*(maskData1 + index) & (1 << 0)) ? copysign(value, data[x * 8 + y * width + 7]) : FILTER_NAN(data[x * 8 + y * width + 7]);
        }
        
        index += mask_page_size;
        pageNumber++;
    }
}

__global__ void g_copyData_setMaskedScale8_removeBlanks(float *data_box, float *data, char *maskData8, const size_t width, const size_t height, const size_t depth, const float value)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {return;}

    size_t index = x + y * width;

    while (index < width * height * depth)
    {
        data_box[index] = ((int8_t)(maskData8[index])) ? copysign(value, data[index]) : FILTER_NAN(data[index]);
        index += width * height;
    }
}

__global__ void g_maskScale_remove_blanks_flt(float *data, char *mask, const size_t width, const size_t height, const size_t depth, const float value)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t mask_width = ceil(width / 8.0f);
    size_t mask_slice_size = mask_width * height;

    if (x < width && y < height)
    {
        for (int z = 0; z < depth; ++z)
        {
            size_t maskIndex = z * mask_slice_size + y * mask_width + x / 8;
            size_t index = z * width * height + y * width + x;
            data[index] = data[index] * (1 - (mask[maskIndex] >> (7 - (x%8))) & 1) + value * ((mask[maskIndex] >> (7 - (x%8))) & 1);
            
        }
    }
}

__global__ void g_filter_gauss_X_flt(float *data, const size_t width, const size_t height, const size_t depth, const size_t radius, const size_t n_iter)
{
    size_t x = threadIdx.x;
    size_t y = blockIdx.y;

    extern __shared__ float s_data_GX_flt[];
    float *s_data_src = s_data_GX_flt + radius;
    float *s_data_dst = s_data_GX_flt + 2 * radius + width;

    if (x == 0) 
    {
        for (int i = radius; i--;)
        {
            *(s_data_GX_flt + i) = *(s_data_GX_flt + radius + width + i) = *(s_data_GX_flt + 2 * radius + 2 * width + i) = 0.0f;
        }
    }

    for (int z = 0; z < depth; ++z)
    {
        //inline size_t index = x + y * width + z * width * height;
        while(x < width && y < height)
        {
            *(s_data_src + x) = data[x + y * width + z * width * height];
            x += blockDim.x;
        }

        x = threadIdx.x;
        __syncthreads();

        for (int n = n_iter; n--;)
        {
            while(x < width && y < height)
            {
                *(s_data_dst + x) = *(s_data_src + x);
                for (int i = radius; i--;)
                {
                    *(s_data_dst + x) += *(s_data_src + x + (i + 1)) + *(s_data_src + x - (i + 1));
                }
                *(s_data_dst + x) /= 2 * radius + 1;
                x += blockDim.x;
            }

            x = threadIdx.x;
            float *tmp = s_data_src;
            s_data_src = s_data_dst;
            s_data_dst = tmp;
            __syncthreads();
        }

        while(x < width && y < height)
        {
            data[x + y * width + z * width * height] = *(s_data_src + x);
            x += blockDim.x;
        }
        x = threadIdx.x;
    }
}

__global__ void g_filter_gauss_Y_flt(float *data, const size_t width, const size_t height, const size_t depth, const size_t radius, const size_t n_iter)
{
    size_t x = blockIdx.x;
    size_t y = threadIdx.y;

    extern __shared__ float s_data_GY_flt[];
    float *s_data_src = s_data_GY_flt + radius;
    float *s_data_dst = s_data_GY_flt + 2 * radius + height;

    if (y == 0) 
    {
        for (int i = radius; i--;)
        {
            *(s_data_GY_flt + i) = *(s_data_GY_flt + radius + height + i) = *(s_data_GY_flt + 2 * radius + 2 * height + i) = 0.0f;
        }
    }

    for (int z = 0; z < depth; ++z)
    {
        //inline size_t index = x + y * width + z * width * height;
        while(x < width && y < height)
        {
            *(s_data_src + y) = data[x + y * width + z * width * height];
            y += blockDim.y;
        }

        y = threadIdx.y;
        __syncthreads();

        for (int n = n_iter; n--;)
        {
            while(x < width && y < height)
            {
                *(s_data_dst + y) = *(s_data_src + y);
                for (int i = radius; i--;)
                {
                    *(s_data_dst + y) += *(s_data_src + y + (i + 1)) + *(s_data_src + y - (i + 1));
                }
                *(s_data_dst + y) /= 2 * radius + 1;
                y += blockDim.y;
            }

            y = threadIdx.y;
            float *tmp = s_data_src;
            s_data_src = s_data_dst;
            s_data_dst = tmp;
            __syncthreads();
        }

        while(x < width && y < height)
        {
            data[x + y * width + z * width * height] = *(s_data_src + y);
            y += blockDim.y;
        }
        y = threadIdx.y;
    }
}

__global__ void g_filter_boxcar_Z_flt(float *data, const size_t width, const size_t height, const size_t depth, const size_t radius)
{
    size_t x = blockIdx.x;
    size_t z = threadIdx.x;

    extern __shared__ float s_data_BZ_flt[];
    float *s_data_src = s_data_BZ_flt + radius;
    float *s_data_dst = s_data_BZ_flt + 2 * radius + depth;

    if (z == 0) 
    {
        for (int i = radius; i--;)
        {
            *(s_data_BZ_flt + i) = *(s_data_BZ_flt + radius + depth + i) = *(s_data_BZ_flt + 2 * radius + 2 * depth + i) = 0.0f;
        }
    }

    for (int y = 0; y < height; ++y)
    {
        //inline size_t index = x + y * width + z * width * height;
        while(x < width && z < depth)
        {
            *(s_data_src + z) = data[x + y * width + z * width * height];
            z += blockDim.x;
        }

        z = threadIdx.x;
        __syncthreads();

        while(x < width && z < depth)
        {
            *(s_data_dst + z) = *(s_data_src + z);
            for (int i = radius; i--;)
            {
                *(s_data_dst + z) += *(s_data_src + z + (i + 1)) + *(s_data_src + z - (i + 1));
            }
            *(s_data_dst + z) /= 2 * radius + 1;
            z += blockDim.x;
        }

        z = threadIdx.x;
        __syncthreads();

        while(x < width && z < depth)
        {
            data[x + y * width + z * width * height] = *(s_data_dst + z);
            z += blockDim.x;
        }
        z = threadIdx.x;
    }
}

__global__ void g_Mask8(float *data_box, char *maskData8, const size_t width, const size_t height, const size_t depth, const double threshold, float *rms_smooth, const int8_t value)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {return;}

    size_t index = x + y * width;

    while (index < width * height * depth)
    {
        if (fabs(data_box[index]) > threshold * (*rms_smooth)) {maskData8[index] = (char)1;}
        index += width * height;
    }
}

void GPU_DataCube_filter(char *data, char *originalData, int word_size, size_t data_size, size_t *axis_size, size_t radiusGauss, size_t n_iter, size_t radiusBoxcar)
{


    GPU_DataCube_filter_Chunked(data, originalData, word_size, data_size, axis_size, radiusGauss, n_iter, radiusBoxcar, 1);
}

void GPU_DataCube_filter_Chunked(char *data, char *originalData, int word_size, size_t data_size, size_t *axis_size, size_t radiusGauss, size_t n_iter, size_t radiusBoxcar, size_t number_of_chunks)
{
    if (!radiusGauss && ! radiusBoxcar) {return;}

    // Error at start?
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error at start: %s\n", cudaGetErrorString(err));    
    }

    // check for CUDA capable device
    cudaFree(0);
    int deviceCount;
    cudaDeviceProp prop;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        exit(0);
    }

    cudaGetDeviceProperties(&prop, 0);

    // Allocate and copy Datacube data onto GPU
    float *d_data;
    float *d_data_box;

    size_t x_overlap = axis_size[1] * axis_size[2] * 2 * radiusGauss;
    size_t y_overlap = axis_size[0] * axis_size[2] * 2 * radiusGauss;
    size_t z_overlap = axis_size[0] * axis_size[1] * 2 * radiusBoxcar;

    size_t box_size =   (data_size 
                            + (x_overlap > y_overlap ? (x_overlap > z_overlap ? x_overlap : z_overlap) 
                            : (y_overlap > z_overlap ? y_overlap : z_overlap))
                        ) 
                        * sizeof(float);
    size_t slices_per_chunk = axis_size[2] / number_of_chunks;

    x_overlap = slices_per_chunk * axis_size[1] * 2 * radiusGauss;
    y_overlap = slices_per_chunk * axis_size[0] * 2 * radiusGauss;

    // if (prop.totalGlobalMem < 2 * box_size)
    // {
    //     number_of_chunks = ((2 * box_size) / prop.totalGlobalMem) + 1;
    //     slices_per_chunk /= number_of_chunks;
    //     slices_per_chunk++;
    // }

    // if (slices_per_chunk < 2 * radiusBoxcar + 1)
    // {
    //     printf("Insufficient memory on GPU to load enought slices of the cube to perform the boxcar filter.\n");
    //     exit(1);
    // }

    size_t chunk_overlap = x_overlap > y_overlap ? 
                        (x_overlap > z_overlap ? x_overlap : z_overlap) : 
                        (y_overlap > z_overlap ? y_overlap : z_overlap);

    err = cudaMalloc((void**)&d_data, (slices_per_chunk + 2 * radiusBoxcar) * axis_size[0] * axis_size[1] * word_size * sizeof(char));
    if (err != cudaSuccess)
    {
        printf("%s\n", cudaGetErrorString(err));
        exit(0);
    }

    err = cudaMemset(d_data, 0, (slices_per_chunk + 2 * radiusBoxcar) * axis_size[0] * axis_size[1] * word_size * sizeof(char));
    if (err != cudaSuccess)
    {
        printf("%s\n", cudaGetErrorString(err));
        exit(0);
    }

    err = cudaMalloc((void**)&d_data_box, (slices_per_chunk * axis_size[0] * axis_size[1] + chunk_overlap) * sizeof(float));
    if (err != cudaSuccess)
    {
        printf("%s\n", cudaGetErrorString(err));
        exit(0);
    }

    size_t remaining_slices = axis_size[2];
    int processed_chunks = 0;

    if (number_of_chunks > 1)
    {
        // TODO protect against thin cubes, where the first copy with a large boxcar filter may not succeed
        cudaMemcpy(d_data + radiusBoxcar * axis_size[0] * axis_size[1], originalData, (slices_per_chunk + radiusBoxcar) * axis_size[0] * axis_size[1] * word_size * sizeof(char), cudaMemcpyDeviceToDevice);

        // Error after mem copy?
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Cuda error at mem Copy to device: %s\n", cudaGetErrorString(err));    
        }

        // Error before Kernel Launch?
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Cuda error before Xkernel launch: %s\n", cudaGetErrorString(err));    
        }

        // Gauss Filter size in X Direction
        dim3 blockSizeX(ceil((float)32 / number_of_chunks),32);
        dim3 gridSizeX((slices_per_chunk + blockSizeX.x - 1) / blockSizeX.x ,
                    (axis_size[1] + blockSizeX.y - 1) / blockSizeX.y);

        if (radiusGauss && !radiusBoxcar) g_DataCube_gauss_filter_XDir<<<gridSizeX, blockSizeX, axis_size[0] * sizeof(float) + (axis_size[0] + 2 * radiusGauss) * sizeof(float)>>>(d_data + radiusBoxcar * axis_size[0] * axis_size[1], d_data_box, word_size, axis_size[0], axis_size[1], slices_per_chunk, radiusGauss, n_iter);

        cudaDeviceSynchronize();

        // Error after Kernel Launch?
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Cuda error after Xkernel launch: %s\n", cudaGetErrorString(err));    
        }

        // Gauss Filter in Y Direction
        dim3 blockSizeY(ceil((float)16 / number_of_chunks),16);
        dim3 gridSizeY((slices_per_chunk + blockSizeY.x - 1) / blockSizeY.x,
                    (axis_size[0] + blockSizeY.y - 1) / blockSizeY.y);

        // Error before Kernel Launch?
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Cuda error before Ykernel launch: %s\n", cudaGetErrorString(err));    
        }

        if (radiusGauss && !radiusBoxcar) g_DataCube_gauss_filter_YDir<<<gridSizeY, blockSizeY>>>(d_data + radiusBoxcar * axis_size[0] * axis_size[1], d_data_box, word_size, axis_size[0], axis_size[1], slices_per_chunk, radiusGauss, n_iter);

        cudaDeviceSynchronize();

        // Error after Kernel Launch?
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Cuda error after Ykernel launch: %s\n", cudaGetErrorString(err));    
        }

        // Boxcar Filter in Z Direction
        dim3 blockSizeZ(32,32);
        dim3 gridSizeZ((axis_size[0] + blockSizeZ.x - 1) / blockSizeZ.x,
                    (axis_size[1] + blockSizeZ.y - 1) / blockSizeZ.y);

        if (radiusBoxcar) g_DataCube_boxcar_filter_flt<<<gridSizeZ, blockSizeZ>>>(d_data, originalData, d_data_box, word_size, processed_chunks * slices_per_chunk, axis_size[0], axis_size[1], slices_per_chunk, radiusBoxcar, 0);

        cudaDeviceSynchronize();

        if (radiusGauss && !radiusBoxcar)
        {
            cudaMemcpy(originalData, d_data + radiusBoxcar * axis_size[0] * axis_size[1], slices_per_chunk * axis_size[0] * axis_size[1] * word_size * sizeof(char), cudaMemcpyDeviceToDevice);
            // Error after backkcopy??
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                printf("Cuda error after backcopy: %s\n", cudaGetErrorString(err));
            }
        }

        cudaMemcpy(data, d_data + radiusBoxcar * axis_size[0] * axis_size[1], slices_per_chunk * axis_size[0] * axis_size[1] * word_size * sizeof(char), cudaMemcpyDeviceToHost);
        // Error after backkcopy??
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Cuda error after backcopy: %s\n", cudaGetErrorString(err));
        }

        remaining_slices -= slices_per_chunk;
        processed_chunks++;
    }

    while(remaining_slices > slices_per_chunk)
    {
        //size_t remaining_slices = axis_size[2] - i * slices_per_chunk;
        size_t slices_to_copy = min(slices_per_chunk + radiusBoxcar, remaining_slices);

        cudaMemcpy(d_data + radiusBoxcar * axis_size[0] * axis_size[1], originalData + processed_chunks * slices_per_chunk * axis_size[0] * axis_size[1] * word_size, slices_to_copy *  axis_size[0] * axis_size[1] * word_size * sizeof(char), cudaMemcpyDeviceToDevice);

        // If there are not enought slices left at the end fill the overlap region with zeroes where neccessary
        if (slices_to_copy < (slices_per_chunk + radiusBoxcar))
        {
            float *zeroes = (float*)calloc((slices_per_chunk + radiusBoxcar - slices_to_copy) * axis_size[0] * axis_size[1], sizeof(float));
            //err = cudaMemcpy(d_data + (radiusBoxcar + slices_to_copy) * axis_size[0] * axis_size[1], zeroes, (slices_per_chunk + radiusBoxcar - slices_to_copy) * axis_size[0] * axis_size[1] * sizeof(float), cudaMemcpyHostToDevice);
            err = cudaMemset(d_data + (radiusBoxcar + slices_to_copy) * axis_size[0] * axis_size[1], 0, (slices_per_chunk + radiusBoxcar - slices_to_copy) * axis_size[0] * axis_size[1] * word_size);
            if (err != cudaSuccess)
            {
                printf("Cuda error at memSet on device: %s\n", cudaGetErrorString(err));
            }
            cudaDeviceSynchronize();
        }
        
        // Error after mem copy?
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Cuda error at mem Copy to device: %s\n", cudaGetErrorString(err));    
        }

        // Error before Kernel Launch?
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Cuda error before Xkernel launch: %s\n", cudaGetErrorString(err));    
        }

        // Gauss Filter size in X Direction
        dim3 blockSizeX(ceil((float)32 / number_of_chunks),32);
        dim3 gridSizeX((slices_per_chunk + blockSizeX.x - 1) / blockSizeX.x ,
                    (axis_size[1] + blockSizeX.y - 1) / blockSizeX.y);

        if (radiusGauss && !radiusBoxcar) g_DataCube_gauss_filter_XDir<<<gridSizeX, blockSizeX, axis_size[0] * sizeof(float) + (axis_size[0] + 2 * radiusGauss) * sizeof(float)>>>(d_data + radiusBoxcar * axis_size[0] * axis_size[1], d_data_box, word_size, axis_size[0], axis_size[1], slices_per_chunk, radiusGauss, n_iter);

        cudaDeviceSynchronize();

        // Error after Kernel Launch?
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Cuda error after Xkernel launch: %s\n", cudaGetErrorString(err));    
        }

        // Gauss Filter in Y Direction
        dim3 blockSizeY(ceil((float)16 / number_of_chunks),16);
        dim3 gridSizeY((slices_per_chunk + blockSizeY.x - 1) / blockSizeY.x,
                    (axis_size[0] + blockSizeY.y - 1) / blockSizeY.y);

        // Error before Kernel Launch?
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Cuda error before Ykernel launch: %s\n", cudaGetErrorString(err));    
        }

        if (radiusGauss && !radiusBoxcar) g_DataCube_gauss_filter_YDir<<<gridSizeY, blockSizeY>>>(d_data + radiusBoxcar * axis_size[0] * axis_size[1], d_data_box, word_size, axis_size[0], axis_size[1], slices_per_chunk, radiusGauss, n_iter);

        cudaDeviceSynchronize();

        // Error after Kernel Launch?
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Cuda error after Ykernel launch: %s\n", cudaGetErrorString(err));    
        }

        // Boxcar Filter in Z Direction
        dim3 blockSizeZ(32,32);
        dim3 gridSizeZ((axis_size[0] + blockSizeZ.x - 1) / blockSizeZ.x,
                    (axis_size[1] + blockSizeZ.y - 1) / blockSizeZ.y);

        if (radiusBoxcar) g_DataCube_boxcar_filter_flt<<<gridSizeZ, blockSizeZ>>>(d_data, originalData, d_data_box, word_size, processed_chunks * slices_per_chunk, axis_size[0], axis_size[1], slices_per_chunk, radiusBoxcar, 1);

        cudaDeviceSynchronize();

        if (radiusGauss && !radiusBoxcar)
        {
            cudaMemcpy(originalData, d_data + radiusBoxcar * axis_size[0] * axis_size[1], slices_per_chunk * axis_size[0] * axis_size[1] * word_size * sizeof(char), cudaMemcpyDeviceToDevice);
            // Error after backkcopy??
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                printf("Cuda error after backcopy: %s\n", cudaGetErrorString(err));
            }
        }

        cudaMemcpy(data + processed_chunks * slices_per_chunk * axis_size[0] * axis_size[1] * word_size, d_data + radiusBoxcar * axis_size[0] * axis_size[1], slices_per_chunk * axis_size[0] * axis_size[1] * word_size * sizeof(char), cudaMemcpyDeviceToHost);
        // Error after backkcopy??
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Cuda error after backcopy: %s\n", cudaGetErrorString(err));
        }

        remaining_slices -= slices_per_chunk;

        processed_chunks++;
    }

    // err = cudaMalloc((void**)&d_data, data_size * word_size * sizeof(char));
    // if (err != cudaSuccess)
    // {
    //     printf("%s\n", cudaGetErrorString(err));
    //     exit(0);
    // }


    // err = cudaMalloc((void**)&d_data_box, (data_size + 
    //                                         (x_overlap > y_overlap ? (x_overlap > z_overlap ? x_overlap : z_overlap) 
    //                                         : (y_overlap > z_overlap ? y_overlap : z_overlap))) 
    //                                         * sizeof(float));
    // if (err != cudaSuccess)
    // {
    //     printf("%s\n", cudaGetErrorString(err));
    //     exit(0);
    // }

    // for (int i = 0; i < blockSizeX.x; i++)
    // {
    //     for (int j = 0; j < blockSizeX.y; j++)
    //     {
    //         cudaMemcpy(d_data 
    //                         + min(i * axis_size[0] * axis_size[1] * gridSizeX.x + (gridSizeX.x - 1) * axis_size[0] * axis_size[1], (axis_size[2]- 1) * axis_size[0] * axis_size[1])
    //                         + min(j * axis_size[0] * gridSizeX.y + (gridSizeX.y - 1) * axis_size[0], (axis_size[1] - 1) * axis_size[0])
    //                         + axis_size[0] - 1, 
    //                         &flag, sizeof(char), cudaMemcpyHostToDevice);
    //     }
    // }

    size_t last_chunk_size = remaining_slices;

    if (last_chunk_size > 0)
    {
        cudaMemcpy(d_data + radiusBoxcar * axis_size[0] * axis_size[1], originalData + processed_chunks * slices_per_chunk * axis_size[0] * axis_size[1] * word_size, last_chunk_size * axis_size[0] * axis_size[1] * word_size * sizeof(char), cudaMemcpyDeviceToDevice);
        // Error after mem copy?
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Cuda error at last mem Copy to device: %s\n", cudaGetErrorString(err));    
        }

        // Error before Kernel Launch?
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Cuda error before Xkernel launch: %s\n", cudaGetErrorString(err));    
        }

        // Gauss Filter size in X Direction
        dim3 blockSizeX(ceil((float)32 / number_of_chunks),32);
        dim3 gridSizeX((last_chunk_size + blockSizeX.x - 1) / blockSizeX.x ,
                    (axis_size[1] + blockSizeX.y - 1) / blockSizeX.y);

        if (radiusGauss && !radiusBoxcar) g_DataCube_gauss_filter_XDir<<<gridSizeX, blockSizeX, axis_size[0] * sizeof(float) + (axis_size[0] + 2 * radiusGauss) * sizeof(float)>>>(d_data + radiusBoxcar * axis_size[0] * axis_size[1], d_data_box, word_size, axis_size[0], axis_size[1], last_chunk_size, radiusGauss, n_iter);

        cudaDeviceSynchronize();

        // Error after Kernel Launch?
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Cuda error after Xkernel launch: %s\n", cudaGetErrorString(err));    
        }

        // Gauss Filter in Y Direction
        dim3 blockSizeY(ceil((float)16 / number_of_chunks),16);
        dim3 gridSizeY((last_chunk_size + blockSizeY.x - 1) / blockSizeY.x,
                    (axis_size[0] + blockSizeY.y - 1) / blockSizeY.y);

        // Error before Kernel Launch?
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Cuda error before Ykernel launch: %s\n", cudaGetErrorString(err));    
        }

        if (radiusGauss && !radiusBoxcar) g_DataCube_gauss_filter_YDir<<<gridSizeY, blockSizeY>>>(d_data + radiusBoxcar * axis_size[0] * axis_size[1], d_data_box, word_size, axis_size[0], axis_size[1], last_chunk_size, radiusGauss, n_iter);

        cudaDeviceSynchronize();

        // Error after Kernel Launch?
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Cuda error after Ykernel launch: %s\n", cudaGetErrorString(err));    
        }

        // Boxcar Filter in Z Direction
        dim3 blockSizeZ(32,32);
        dim3 gridSizeZ((axis_size[0] + blockSizeZ.x - 1) / blockSizeZ.x,
                    (axis_size[1] + blockSizeZ.y - 1) / blockSizeZ.y);

        if (radiusBoxcar) g_DataCube_boxcar_filter_flt<<<gridSizeZ, blockSizeZ>>>(d_data, originalData, d_data_box, word_size, processed_chunks * slices_per_chunk, axis_size[0], axis_size[1], last_chunk_size, radiusBoxcar, 2);

        cudaDeviceSynchronize();

        if (radiusGauss && !radiusBoxcar)
        {
            //g_DataCube_copy_back_smoothed_cube(originalData, d_data, word_size, axis_size[0], axis_size[1], last_chunk_size);
        }

        cudaMemcpy(data + processed_chunks * slices_per_chunk * axis_size[0] * axis_size[1] * word_size, d_data + radiusBoxcar * axis_size[0] * axis_size[1], last_chunk_size * axis_size[0] * axis_size[1] * word_size * sizeof(char), cudaMemcpyDeviceToHost);
        // Error after backkcopy??
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Cuda error after backcopy: %s\n", cudaGetErrorString(err));
        }
    }

    cudaFree(d_data);
    cudaFree(d_data_box);

    // Error after free mem??
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after freeing memory: %s\n", cudaGetErrorString(err));    
    }
}

void GPU_DataCube_copy_mask_8_to_1(char* maskData1, char* maskData8, const size_t *axis_size)
{
    dim3 blockSize(32,32);
    dim3 gridSize((ceil(axis_size[0] / 8.0f) + blockSize.x - 1) / blockSize.x,
                            (axis_size[1] + blockSize.y - 1) / blockSize.y);

    g_DataCube_copy_mask_8_to_1<<<gridSize, blockSize>>>(maskData1, maskData8, axis_size[0], axis_size[1], axis_size[2]);
    cudaDeviceSynchronize();
}

__global__ void g_DataCube_boxcar_filter_flt(float *data, char *originalData, float *data_box, int word_size, const size_t startSlice, size_t width, size_t height, size_t depth, size_t radius, size_t chunck_type)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t jump = width * height;

    if (x < width && y < height)
    {
        data = data + (x + y * width);
        data_box = data_box + (x + y * width);

        d_filter_chunk_boxcar_1d_flt(data, originalData, data_box, startSlice, depth, radius, jump, chunck_type);
    }
}

__global__ void g_DataCube_gauss_filter_XDir(float *data, float *data_box, int word_size, size_t width, size_t height, size_t depth, size_t radius, size_t n_iter)
{
    size_t thread_count = blockDim.x * blockDim.y;
    size_t thread_index = threadIdx.x * blockDim.y + threadIdx.y;

    const size_t filter_size = 2 * radius + 1;
	const float inv_filter_size = 1.0 / filter_size;

    extern __shared__ float s_data[];
    float *s_data_box = &s_data[width];

    size_t start_index = blockIdx.x * blockDim.x * width * height + blockIdx.y * blockDim.y * width;    

    for (int iter = 0; iter < blockDim.x * blockDim.y; iter++)
    {
        if (blockIdx.y * blockDim.y + (iter % blockDim.y) >= height) continue;

        size_t data_index = start_index + (iter % blockDim.y) * width + (iter / blockDim.y) * width * height;

        if (data_index >= width * height * depth) continue;

        for (size_t i = 0; i < (float)width / thread_count; i++)
        {
            size_t j = thread_index + thread_count * i;
            if (j < width)
            {
                s_data[j] = s_data_box[radius + j] = data[data_index + j];
            }
        }

         for (int i = radius; i--;) s_data_box[i] = s_data_box[radius + width + i] = 0.0;

        __syncthreads();

        for (size_t k = n_iter; k--;)
        {
            for (int i = 0; i < (float)width / thread_count; i++)
            {
                int j = thread_index + thread_count * i;
                if (j < width)
                {
                    s_data[j] = 0.0;
                    for(int f = filter_size; f--;) s_data[j] += s_data_box[j + f];
                    s_data[j] *= inv_filter_size;
                }
            }

            __syncthreads();

            for (int i = 0; i < (float)width / thread_count; i++)
            {
                int j = thread_index + thread_count * i;
                if (j < width)
                {
                    s_data_box[radius + j] = s_data[j];
                }
            }

            __syncthreads();
        }

        for (int i = 0; i < width / (float)thread_count; i++)
        {
            int j = thread_index + thread_count * i;
            if (j < width)
            {
                data[data_index + j] = s_data[j];
            }
        }
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

__global__ void g_DataCube_copy_mask_8_to_1(char* maskData1, char* maskData8, size_t width, size_t height, size_t depth)
{
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = 0;

    size_t jump = width * height;
    size_t jump1 = ((width + 7) / 8) * height;

    while (z < depth)
    {
        size_t x = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

        if (x < width && y < height)
        {
            size_t indexSrc = width * y + x + z * jump;
            size_t indexDst = ((width + 7) / 8) * y + x / 8 + z * jump1;
            int8_t *srcPtr = (int8_t*)maskData8 + indexSrc;
            char *dstPtr = maskData1 + indexDst;

            u_int8_t result = 0;

            result |= (*srcPtr++ != 0) << 7;
            result |= (*srcPtr++ != 0) << 6;
            result |= (*srcPtr++ != 0) << 5;
            result |= (*srcPtr++ != 0) << 4;
            result |= (*srcPtr++ != 0) << 3;
            result |= (*srcPtr++ != 0) << 2;
            result |= (*srcPtr++ != 0) << 1;
            result |= (*srcPtr != 0) << 0;

            *dstPtr = (char)result;

            //x += blockDim.x * 8;
        }

        z++;
    }
}

__global__ void g_DataCube_copy_back_smoothed_cube(char *originalData, float *data, int word_size, size_t width, size_t height, size_t depth)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t jump = width * height;

    if (x < width && y < height)
    {
        size_t z = 0;
        size_t index = x + y * width;
        while (z < depth)
        {
            originalData[index + z * jump] = IS_NAN(originalData[index + z * jump]) ? originalData[index + z * jump] : data[index + z * jump];
            z++;
        }
    }
}

__global__ void g_DataCube_stat_mad_flt(float *data, float *data_box, size_t width, size_t height, size_t depth, const float value, const size_t cadence, const int range)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t index = y * blockDim.x * gridDim.x + x;
    const size_t local_index = y * blockDim.x + x;

    const size_t thread_count = blockDim.x * blockDim.y;
    const size_t thread_count_total = gridDim.x * gridDim.y * thread_count;
    const size_t max_medians_per_thread = ((((width * height * depth + cadence - 1) / cadence + thread_count_total - 1) / thread_count_total) + 4) / 5;

    float *ptr = data + width * height * depth - 1 - index * cadence;
    float *ptr_box_start = data_box + index * (max_medians_per_thread + 1);
    float *ptr_box = ptr_box_start + 1;

    extern __shared__ float s_data_mad[];
    float *s_data_start = s_data_mad + local_index * 6;
    float *s_data_median_start = s_data_mad + thread_count * 6 + local_index;
    int counter = 0;

    if (range == 0)
    {
        while (ptr >= data)
        {
            while (ptr >= data && counter < 5)
            {
                if (IS_NOT_NAN(*ptr))
                {
                    *(s_data_start + counter) = fabs(*ptr - value);
                    counter++;
                }
                ptr -= thread_count_total * cadence;
            }

            *(s_data_start + 5) = 0;
            
            if (counter > 0)
            {
                d_sort_arr_flt(s_data_start, counter);
                *s_data_median_start = counter % 2 != 0 ? 
                                        *(s_data_start + counter / 2) : 
                                        (*(s_data_start + counter / 2) + *(s_data_start + (counter / 2 - 1))) / 2;
                *(s_data_start + 5) = 1;
            }
            __syncthreads();

            counter = 2;
            while (local_index % counter == 0 && counter <= thread_count)
            {
                *(s_data_start + 5) = *(s_data_start + 5) + *(s_data_start + 5 + 6 * counter / 2);
                counter *= 2;
            }

            counter = 0;

            __syncthreads();

            if (local_index == 0)
            {
                d_sort_arr_flt(s_data_median_start, (int)*(s_data_start + 5));
                // data_box[(int)atomicAdd(data_box + width * height * depth - 1, 1)] = *s_data_median_start;
                //thrust::sort(s_data_median_start, s_data_median_start + (int)*s_data_start);
                data_box[(int)atomicAdd(data_box + width * height * depth - 1, 1)] = (int)*(s_data_start + 5) % 2 != 0 ? 
                                                                                    *(s_data_median_start + (int)*(s_data_start + 5) / 2) :
                                                                                    (*(s_data_median_start + (int)*(s_data_start + 5) / 2) + *(s_data_median_start + (int)*(s_data_start + 5) / 2 - 1)) / 2;
            }
            __syncthreads();
        }
    }
}

__global__ void g_DataCube_stat_mad_flt_2(float *data, float *data_box, size_t size, const float value, const size_t cadence, const int range, const float pivot)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_count = blockDim.x * gridDim.x;

    extern __shared__ float s_data_stat_mad[];
}

__global__ void g_std_dev_val_flt(float *data, float *data_dst_duo, const size_t size, const float value, const size_t cadence, const int range)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_count = blockDim.x * gridDim.x;

    extern __shared__ float s_data_sdf[];
    float *s_data_sdf_start = s_data_sdf + threadIdx.x * 2;
    *s_data_sdf_start = 0.0f;
    *(s_data_sdf_start + 1) = 0.0f;

    const float *ptr = data + size + index * cadence;
    const float *ptr2 = data + cadence * thread_count - 1;

    while (ptr > ptr2)
    {
        ptr -= cadence * thread_count;

        if((range == 0 && IS_NOT_NAN(*ptr)) || (range < 0 && *ptr < 0.0) || (range > 0 && *ptr > 0.0))
		{
			*s_data_sdf_start += (*ptr - value) * (*ptr - value);
			++*(s_data_sdf_start + 1);
		}
    }

    __syncthreads();

    int counter = 2;
    while (counter / 2 < blockDim.x)
    {
        if (threadIdx.x % counter == 0)
        {
            if (*(s_data_sdf_start + counter) < 0.0f) {atomicAdd(data_dst_duo + 2, 1);}
            *s_data_sdf_start += *(s_data_sdf_start + counter);
            *(s_data_sdf_start + 1) += *(s_data_sdf_start + 1 + counter);
        }
        counter *= 2;
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(data_dst_duo, *s_data_sdf_start);
        atomicAdd(data_dst_duo + 1, *(s_data_sdf_start + 1));
    }
}

__global__ void g_std_dev_val_flt_final_step(float *data_duo)
{
    *data_duo = sqrt(*data_duo / *(data_duo + 1));
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

__device__ void d_filter_chunk_boxcar_1d_flt(float *data, char *originalData, float *data_copy, const size_t startSlice, const size_t size, const size_t filter_radius, const size_t jump, size_t chunk_type)
{
    // Define filter size
	const size_t filter_size = 2 * filter_radius + 1;
	const float inv_filter_size = 1.0 / filter_size;
	size_t i;

    if (chunk_type != 2)
    {
        for(i = filter_radius; i--;) data_copy[i * jump] = FILTER_NAN(data[i * jump]);
        for(i = filter_radius; i--;) data_copy[(size + filter_radius + i) * jump] = FILTER_NAN(data[(size + filter_radius + i) * jump]);
    }
    else
    {
        for(i = filter_radius; i--;) data_copy[i * jump] = FILTER_NAN(data[i * jump]);
        for(i = filter_radius; i--;) data_copy[(size + filter_radius + i) * jump] = 0.0;
    }

    // // Fill overlap regions
    // if (chunk_type == 0)
    // {
    //     for(i = filter_radius; i--;) data_copy[i * jump] = 0.0;
    //     for(i = filter_radius; i--;) data_copy[(size + filter_radius + i) * jump] = FILTER_NAN(data[(size + filter_radius + i) * jump]);
    // }
    // else if (chunk_type == 1)
    // {
    //     for(i = filter_radius; i--;) data_copy[i * jump] = FILTER_NAN(data[i * jump]);
    //     for(i = filter_radius; i--;) data_copy[(size + filter_radius + i) * jump] = FILTER_NAN(data[(size + filter_radius + i) * jump]);
    // }
    // else if (chunk_type == 2)
    // {
    //     for(i = filter_radius; i--;) data_copy[i * jump] = FILTER_NAN(data[i * jump]);
    //     for(i = filter_radius; i--;) data_copy[(size + filter_radius + i) * jump] = 0.0;
    // }

    // Write elements at the end of the data chunk back to the front end overlap for next chunk
    if (chunk_type != 2)
    {
        for(i = 0; i < filter_radius; i++) data[i * jump] = data[(size + i) * jump];
    }

	// Make copy of data, taking care of NaN
	for(i = size; i--;) data_copy[(filter_radius + i) * jump] = FILTER_NAN(data[(filter_radius + i) * jump]);
	
	// Apply boxcar filter to last data point
	data[(size + filter_radius - 1) * jump] = 0.0;
	for(i = filter_size; i--;) data[(size + filter_radius - 1) * jump] += data_copy[(size + i - 1) * jump];
	data[(size + filter_radius - 1) * jump] *= inv_filter_size;
	
	// Recursively apply boxcar filter to all previous data points
	for(i = size - 1; i--;) data[(filter_radius + i) * jump] = data[(filter_radius + i + 1) * jump] + (data_copy[i * jump] - data_copy[(filter_size + i) * jump]) * inv_filter_size;
	
	return;
}

__device__ void d_sort_arr_flt(float *arr, size_t size)
{
    float tmp;
    for (int i = size + 1; --i;)
    {
        int j = size - 1;
        while(j > size - i)
        {
            tmp = arr[j - 1];
            if (arr[j - 1] > arr[j])
            {
                arr[j - 1] = arr[j];
                arr[j] = tmp;
            }
            --j;
        }
    }
}

void sort_arr_flt(float *arr, size_t size)
{
    float tmp;
    for (int i = size + 1; --i;)
    {
        int j = size - 1;
        while(j > size - i)
        {
            tmp = arr[j - 1];
            if (arr[j - 1] > arr[j])
            {
                arr[j - 1] = arr[j];
                arr[j] = tmp;
            }
            --j;
        }
    }
}