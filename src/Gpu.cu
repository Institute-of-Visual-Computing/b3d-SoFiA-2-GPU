#include "Gpu.h"

void GPU_test_current()
{
    dim3 blockSize(32);
    dim3 gridSize(1);

    g_test<<<gridSize, blockSize>>>();

    cudaDeviceSynchronize();
}

__global__ void g_test()
{
    uint i = threadIdx.x % 2 == 0 ? 1 << threadIdx.x : 0;
    int x = threadIdx.x;

    for (int offset = 16; offset > 0; offset /= 2)
    {
        i |= __shfl_up_sync(0xffffffff, i, offset);
    }

    //i = i >> threadIdx.x;

    i = __popc(i);

    printf("i for thread %d: %u\n", x, i);
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

    dim3 blockSize(32,32);
    dim3 gridSize(1,1);

    g_filter_gauss_X_flt_new<<<gridSize, blockSize, 1000 * sizeof(float)>>>(d_data, 10, 1, 1, 1, 1);

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

    g_filter_boxcar_Z_flt_new<<<gridSize, blockSize, 5 * 3 * sizeof(float)>>>(d_data, 1, 1, 10, 1);

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

    g_std_dev_val_flt<<<gridSize, blockSize, blockSize.x * 2 * sizeof(float)>>>(d_data, d_data_box, size, 0, cadence, range, 0);

    cudaDeviceSynchronize();

    g_std_dev_val_flt_final_step<<<1,1>>>(d_data_box);

    cudaDeviceSynchronize();

    float noise[2] = {0,0};
    cudaMemcpy(noise, d_data_box, 2 * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0 ; i < 2; i++){printf("noise: %.3e\n", noise[i]);;}

    cudaFree(d_data);
    cudaFree(d_data_box);
}

void GPU_test_median(float *data, size_t size)
{
    //size_t size = 999999;
    //float data[size];// = {100,200,3,4,5,6,7,8,9,10,11};

    //for (int i = 0; i < size; i++) {data[i] = size - i;}

    printf("Array to get median: ");

	for (int i = 0 ; i < 11; i++){printf("%f ", data[i]);}

	printf("\n");

    printf("True Median: %f\n", mad_val_flt(data, size, 0.0, 4, 0));

    float *d_data;
    float *d_data_box;
    unsigned int *d_counter;

    cudaMalloc((void**)&d_data, size * sizeof(float));
    cudaMalloc((void**)&d_data_box, size * sizeof(float));
    cudaMalloc((void**)&d_counter, sizeof(unsigned int));

    cudaMemset(d_data_box, 0, size * sizeof(float));
    cudaMemset(d_counter, 0, sizeof(unsigned int));

    cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Cuda error at start: %s\n", cudaGetErrorString(err));  
    }

    cudaDeviceSynchronize();

    dim3 blockSize(1024);
    dim3 gridSize(16);

    g_mad_val_flt<<<gridSize, blockSize, (blockSize.x + 1) * sizeof(float)>>>(d_data, d_data_box, d_counter, size, size, 0, 4, 0);

    unsigned int count;
    cudaMemcpy(&count, d_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Median values count: %u\n", count);

    cudaDeviceSynchronize();

    g_mad_val_flt_final_step<<<1,256>>>(d_data_box, d_counter);

    cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        printf("Cuda error after kernels %s\n", cudaGetErrorString(err));  
    }

    //cudaMemcpy(data, d_data_box, 12 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(data + size, d_data_box, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        printf("Cuda error after kernels %s\n", cudaGetErrorString(err));  
    }

    cudaDeviceSynchronize();

    printf("DataBox: ");

	for (int i = 0 ; i < 11; i++){printf("%f ", data[i]);}

	printf("\n");

    printf("Median: %f\n", data[size]);
}

void GPU_test_median_recursive(float *data, size_t size, const size_t *axis_size, const int snRange)
{
    uint precision = 32;
    float h_max;
    float h_min;

    max_min_flt(data, size, &h_max, &h_min);

    float *d_data;
    float *d_data_box;
    unsigned int *d_bins;
    unsigned int *d_counter;

    cudaMalloc((void**)&d_data, size * sizeof(float));
    cudaMalloc((void**)&d_data_box, size * sizeof(float));
    cudaMalloc((void**)&d_bins, precision * sizeof(unsigned int));
    cudaMalloc((void**)&d_counter, sizeof(unsigned int));

    cudaMemset(d_data_box, 0, size * sizeof(float));
    cudaMemset(d_bins, 0, precision * sizeof(unsigned int));
    cudaMemset(d_counter, 0, sizeof(unsigned int));

    cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Cuda error at start: %s\n", cudaGetErrorString(err));  
    }

    cudaDeviceSynchronize();

    dim3 blockSize(256);
    dim3 gridSize(16);

    float true_rms = mad_val_flt(data, size, 0.0, 1, snRange);

    printf("True RMS: %0.20e\n", true_rms);

    int selectedBin = 0;
    unsigned int h_bins[precision];
    unsigned int tmpCount;
    unsigned int h_count = 0;
    unsigned int median_counter = 0;
    unsigned int breakCond = 500;

    float *d_med_arr;
    unsigned int *d_med_ptr;
    cudaMalloc((void**)&d_med_arr, breakCond * sizeof(float));
    cudaMalloc((void**)&d_med_ptr, sizeof(unsigned int));
    cudaMemset(d_med_ptr, 0, sizeof(unsigned int));

    h_bins[0] = breakCond + 1;

    int iteration = 0;
    float my_max = snRange == 0 ? max(fabs(h_min), fabs(h_max)) : (snRange < 0 ? max(fabs(h_min), 0.0f) : max(fabs(h_max), 0.0f));
    while (h_bins[selectedBin] > breakCond && iteration < 20)
    {
        selectedBin = 0;
        cudaMemset(d_bins, 0, precision * sizeof(unsigned int));
        cudaMemset(d_counter, 0, sizeof(unsigned int));
        g_bin_data<<<gridSize, blockSize, (blockSize.x / 32) * precision * sizeof(unsigned int) >>>(d_data, size, d_bins, precision, h_min, h_max, snRange, 1, d_counter, iteration != 0);

        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Cuda error after noise scaling: %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(h_bins, d_bins, precision * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&tmpCount, d_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        if (iteration == 0)
        {
            median_counter = tmpCount / 2;
        }

        unsigned int k = median_counter - h_count + 1;


        for (int i = h_bins[0]; i < k; i+=h_bins[++selectedBin])
        {
            h_count += h_bins[selectedBin];
        }

        float bucket_size = iteration != 0 ? (h_max - h_min) / precision : my_max / precision;

        if (iteration == 0)
        {
            h_min = bucket_size * selectedBin;
            h_max = bucket_size * (selectedBin + 1);
        }
        else
        {
            h_min = h_min + bucket_size * selectedBin;
            h_max = h_min + bucket_size;
        }

        iteration++;

        if(h_bins[selectedBin] < breakCond)
        {
            g_cpy_bin<<<gridSize, blockSize>>>(d_data, size, d_med_arr, d_med_ptr, h_min, h_max, snRange, 1);
            g_median_final<<<1,1>>>(d_med_arr, d_med_ptr,  median_counter - h_count);
            cudaDeviceSynchronize();
        }
    }
    cudaDeviceSynchronize();
}

void GPU_MAD(float *d_data, size_t size, float *d_dstMAD, const int snRange, unsigned int precision, const size_t cadence, float h_min, float h_max)
{
    unsigned int *d_bins;
    unsigned int *d_counter;
    float *d_med_arr;
    unsigned int *d_med_ptr;

    int selectedBin = 0;
    unsigned int h_bins[precision];
    unsigned int tmpCount;
    unsigned int h_count = 0;
    unsigned int median_counter = 0;
    unsigned int breakCond = 500;

    cudaMalloc((void**)&d_bins, precision * sizeof(unsigned int));
    cudaMalloc((void**)&d_counter, sizeof(unsigned int));
    cudaMalloc((void**)&d_med_arr, breakCond * sizeof(float));
    cudaMalloc((void**)&d_med_ptr, sizeof(unsigned int));

    cudaMemset(d_med_ptr, 0, sizeof(unsigned int));

    cudaError_t err = cudaGetLastError();

    dim3 blockSize(256);
    dim3 gridSize(16);

    h_bins[0] = breakCond + 1;
    int iteration = 0;
    float my_max = snRange == 0 ? max(fabs(h_min), fabs(h_max)) : (snRange < 0 ? max(fabs(h_min), 0.0f) : max(fabs(h_max), 0.0f));

    while (h_bins[selectedBin] > breakCond && iteration < 1000)
    {
        selectedBin = 0;
        cudaMemset(d_bins, 0, precision * sizeof(unsigned int));
        cudaMemset(d_counter, 0, sizeof(unsigned int));
        g_bin_data<<<gridSize, blockSize, (blockSize.x / 32) * precision * sizeof(unsigned int)>>>(d_data, size, d_bins, precision, h_min, h_max, snRange, cadence, d_counter, iteration != 0);

        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Cuda error after new MAD: %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(h_bins, d_bins, precision * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&tmpCount, d_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        if (iteration == 0)
        {
            median_counter = tmpCount / 2;
        }

        unsigned int k = median_counter - h_count + 1;

        for (int i = h_bins[0]; i < k; i+=h_bins[++selectedBin])
        {
            h_count += h_bins[selectedBin];
        }

        float bucket_size = iteration != 0 ? (h_max - h_min) / precision : my_max / precision;

        if (iteration == 0)
        {
            h_min = bucket_size * selectedBin;
            h_max = bucket_size * (selectedBin + 1);
        }
        else
        {
            h_min = h_min + bucket_size * selectedBin;
            h_max = h_min + bucket_size;
        }

        iteration++;
        cudaDeviceSynchronize();
    }

    g_cpy_bin<<<gridSize, blockSize>>>(d_data, size, d_med_arr, d_med_ptr, h_min, h_max, snRange, cadence);
    g_median_final<<<1,1>>>(d_med_arr, d_med_ptr,  median_counter - h_count);
    cudaMemcpy(d_dstMAD, d_med_arr, sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
}

void GPU_test_hist(float *data, size_t size, size_t cadence, const int range)
{
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Cuda error at start: %s\n", cudaGetErrorString(err));  
    }

    //size_t size = 11;
    unsigned int precision = 1024;
    
    //float data[size] = {20,30,3,4,5,6,7,8,9,10,11};
    unsigned int bins[precision];

    //for (int i = 0; i < size; i++) {data[i] = size - i;}

    printf("Array to get median: ");

	for (int i = 0 ; i < 11; i++){printf("%f ", data[i]);}

	printf("\n");

    printf("True Median: %.3e\n", mad_val_flt(data, size, 0.0, cadence, range));

    float *d_data;
    float *d_data_box;
    float *d_min;
    float *d_max;
    unsigned int *d_bins;
    unsigned int *d_bin_total_values;
    unsigned int *d_counter;

    float min = INFINITY;
    float max = -INFINITY;

    cudaMalloc((void**)&d_data, size * sizeof(float));
    cudaMalloc((void**)&d_data_box, size * sizeof(float));
    cudaMalloc((void**)&d_min,sizeof(float));
    cudaMalloc((void**)&d_max,sizeof(float));
    cudaMalloc((void**)&d_bins, precision * sizeof(unsigned int));
    cudaMalloc((void**)&d_bin_total_values, sizeof(unsigned int));
    cudaMalloc((void**)&d_counter, sizeof(unsigned int));

    cudaMemset(d_bins, 0, precision * sizeof(unsigned int));
    cudaMemset(d_bin_total_values, 0, sizeof(unsigned int));
    cudaMemset(d_counter, 0, sizeof(unsigned int));

    cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_min, &min, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &max, sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        printf("Cuda error after Memalloc: %s\n", cudaGetErrorString(err));  
    }

    cudaDeviceSynchronize();

    dim3 blockSize(1024);
    dim3 gridSize(32);

    min = min_flt(data, size);
    max = max_flt(data, size);

    //cudaMemcpy(d_min, &min, sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_max, &max, sizeof(float), cudaMemcpyHostToDevice);

    printf("Min : %f\nMax: %f\n", min, max);

    unsigned int nth = (size / 2) / cadence;

    g_find_max_min<<<1024, 1024>>>(d_data, size, cadence, d_min, d_max);

    cudaDeviceSynchronize();

    min = 0;
    max = 0;

    cudaMemcpy(&min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max, d_max, sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    printf("Min : %f\nMax: %f\n", min, max);

    g_mad_val_hist_flt<<<gridSize, blockSize, precision * sizeof(unsigned int)>>>(d_data, size, d_bins, d_bin_total_values, 0.0, cadence, range, precision, d_min, d_max);

    cudaDeviceSynchronize();

    g_mad_val_hist_flt_cpy_nth_bin<<<gridSize, blockSize, blockSize.x * sizeof(float)>>>(d_data, size, d_data_box, d_bins, d_bin_total_values, d_counter, 0.0, cadence, range, precision, d_min, d_max);

    g_mad_val_hist_flt_final_step<<<1,1>>>(d_data_box, d_counter, d_bin_total_values, d_bins);

    float median;
    cudaMemcpy(&median, d_data_box, sizeof(float), cudaMemcpyDeviceToHost);

    unsigned int counter;
    cudaMemcpy(&counter, d_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaMemcpy(bins, d_bins, precision * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        printf("Cuda error after back cpy %s\n", cudaGetErrorString(err));  
    }

    printf("GPU Median: %.3e\n", median / MAD_TO_STD);

    // printf("Bins: ");

	// for (int i = 0 ; i < precision; i++){printf("%u ", bins[i]);}
    
    // printf("\n");

    printf("Counter: %u\n", counter);
}

void GPU_test_gausfit(float *data, size_t size, size_t cadence, const int range)
{
    float *d_data;
    float *d_sigma_out;

    cudaMalloc((void**)&d_data, size * sizeof(float));
    cudaMalloc((void**)&d_sigma_out, sizeof(float));

    cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice);

    printf("True GaussFit: %.3e\n", gaufit_flt(data, size, cadence, range));

    GPU_gaufit(d_data, size, d_sigma_out, cadence, range);

    // cudaError_t err = cudaGetLastError();

    // if (err != cudaSuccess)
    // {
    //     printf("Cuda error at start: %s\n", cudaGetErrorString(err));  
    // }

    // //size_t size = 11;
    // unsigned int precision = 101;
    
    // //float data[size] = {20,30,3,4,5,6,7,8,9,10,11};
    // unsigned int bins[precision];

    // //for (int i = 0; i < size; i++) {data[i] = size - i;}

    // printf("True GaussFit: %.3e\n", gaufit_flt(data, size, cadence, range));

    // float *d_data;
    // unsigned int *d_histogram;
    // float *d_min;
    // float *d_max;
    // float *d_sigma;

    // float min = INFINITY;
    // float max = -INFINITY;

    // cudaMalloc((void**)&d_data, size * sizeof(float));
    // cudaMalloc((void**)&d_histogram, precision * sizeof(unsigned int));
    // cudaMalloc((void**)&d_min,sizeof(float));
    // cudaMalloc((void**)&d_max,sizeof(float));
    // cudaMalloc((void**)&d_sigma,sizeof(float));

    // cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_min, &min, sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_max, &max, sizeof(float), cudaMemcpyHostToDevice);

    // cudaMemset(d_histogram, 0, precision * sizeof (unsigned int));

    // if (err != cudaSuccess)
    // {
    //     printf("Cuda error after Memalloc: %s\n", cudaGetErrorString(err));  
    // }

    // cudaDeviceSynchronize();

    // dim3 blockSize(1024);
    // dim3 gridSize(32);

    // min = min_flt(data, size);
    // max = max_flt(data, size);

    // //cudaMemcpy(d_min, &min, sizeof(float), cudaMemcpyHostToDevice);
    // //cudaMemcpy(d_max, &max, sizeof(float), cudaMemcpyHostToDevice);

    // printf("Min CPU: %f\nMax CPU: %f\n", min, max);

    // g_find_max_min<<<1024, 1024>>>(d_data, size, cadence, d_min, d_max);

    // cudaDeviceSynchronize();

    // min = 0;
    // max = 0;

    // cudaMemcpy(&min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&max, d_max, sizeof(float), cudaMemcpyDeviceToHost);

    // cudaDeviceSynchronize();

    // printf("Min GPU: %f\nMax GPU: %f\n", min, max);

    // g_gaufit_flt<<<1, 32>>>(d_data, size, d_histogram, d_sigma, cadence, range, d_min, d_max);

    // cudaDeviceSynchronize();

    // float sigma;

    // cudaMemcpy(&sigma, d_sigma, sizeof(float), cudaMemcpyDeviceToHost);

    // cudaDeviceSynchronize();

    // if (err != cudaSuccess)
    // {
    //     printf("Cuda error after back cpy %s\n", cudaGetErrorString(err));  
    // }

    // // printf("Bins: ");

	// // for (int i = 0 ; i < precision; i++){printf("%u ", bins[i]);}
    
    // // printf("\n");

    // printf("GPU GaussFit: %.3e\n", sigma);
}

void GPU_test_flag_sources()
{
    float mask32[27] = {
                        -1, 0, -1,
                        -1, 0, -1,
                        0, -1, -1,
                    
                        -1, -1, 0,
                        0, -1, -1,
                        0, 0, 0,
                    
                        -1, 0, -1,
                        0, -1, -1,
                        0, -1, -1
                        };

    u_int32_t sources[30];

    for (int i = 0; i < 27; i++)
    {
        printf("%f ", mask32[i]);
        if (i % 3 == 2) {printf("\n");}
        if (i % 9 == 8){printf("\n");}
    }
    printf("\n");

    float *d_mask32;
    uint32_t *d_BBs;
    uint32_t *d_BBcounter;

    cudaError_t err;

    printf("Started GPU\n");
    
    cudaMalloc((void**)&d_mask32, 27 * sizeof(float));
    cudaMalloc((void**)&d_BBs, 30 * 3 * sizeof(uint32_t));
    cudaMalloc((void**)&d_BBcounter, sizeof(uint32_t));

    if ((err = cudaGetLastError()) != cudaSuccess)
    {
        printf("Cuda error after Malloc: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaMemcpy(d_mask32, mask32, 27 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_BBs, 0, 30 * 3 * sizeof(uint32_t));
    cudaMemset(d_BBcounter, 0, sizeof(uint32_t));
    cudaDeviceSynchronize();

    if ((err = cudaGetLastError()) != cudaSuccess)
    {
        printf("Cuda error after Memset: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    dim3 blockSize(3,3);
    dim3 gridSize(1,1);

    g_FlagSources<<<gridSize, blockSize>>>(d_mask32, d_BBcounter, d_BBs, 3, 3, 3, 1);
    cudaDeviceSynchronize();

    if ((err = cudaGetLastError()) != cudaSuccess)
    {
        printf("Cuda error after Flagging: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    g_MergeSourcesFirstPass<<<gridSize, blockSize>>>(d_mask32, d_BBs, 3, 3, 3, 1, 1, 1);
    cudaDeviceSynchronize();

    if ((err = cudaGetLastError()) != cudaSuccess)
    {
        printf("Cuda error after Merging: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaMemcpy(mask32, d_mask32, 27 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sources, d_BBs, 30 * sizeof(u_int32_t), cudaMemcpyDeviceToHost);

    if ((err = cudaGetLastError()) != cudaSuccess)
    {
        printf("Cuda error after back copy: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    for (int i = 0; i < 27; i++)
    {
        printf("%f ", mask32[i]);
        if (i % 3 == 2) {printf("\n");}
        if (i % 9 == 8){printf("\n");}
    }
    printf("\n");

    for (int i = 0; i < 30; i+=3)
    {
        printf("%lu, %lu, %lu\n", sources[i], sources[i + 1], sources[i + 2]);
    }
    
    cudaFree(d_mask32);
    cudaFree(d_BBs);
    cudaFree(d_BBcounter);

    if ((err = cudaGetLastError()) != cudaSuccess)
    {
        printf("Cuda error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    printf("Finished GPU\n");
}

void GPU_test_transpose()
{
    size_t sizeX = 1024;
    size_t sizeY = 1024;
    size_t sizeZ = 448;
    size_t size = sizeX * sizeY * sizeZ;
    float *data = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < size; i++) data[i] = i;

    printf("Matrix to transpose:");

	// for (int i = 0 ; i < size; i+=32)
    // {
    //     if (i % 256 == 0) printf("\n");
    //     printf("%g \t%g \t%g \t%g \t%g \t%g \t%g \t%g\t\t", data[i+0], data[i+1], data[i+2], data[i+3], data[i+4], data[i+5], data[i+6], data[i+7]);
    //     printf("%g \t%g \t%g \t%g \t%g \t%g \t%g \t%g\t\t", data[i+8], data[i+9], data[i+10], data[i+11], data[i+12], data[i+13], data[i+14], data[i+15]);
    //     printf("%g \t%g \t%g \t%g \t%g \t%g \t%g \t%g\t\t", data[i+16], data[i+17], data[i+18], data[i+19], data[i+20], data[i+21], data[i+22], data[i+23]);
    //     printf("%g \t%g \t%g \t%g \t%g \t%g \t%g \t%g\n", data[i+24], data[i+25], data[i+26], data[i+27], data[i+28], data[i+29], data[i+30], data[i+31]);
    // }

	printf("\n");

    float *d_data;
    float *d_data_cpy;

    cudaMalloc((void**)&d_data, size * sizeof(float));
    cudaMalloc((void**)&d_data_cpy, size * sizeof(float));

    cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_cpy, d_data, size * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Cuda error at memcpy: %s\n", cudaGetErrorString(err));  
    }

    cudaDeviceSynchronize();

    dim3 blockSize(32,16);
    dim3 gridSize((sizeX + blockSize.x - 1) / blockSize.x, (sizeY + blockSize.x - 1) / blockSize.x, 4);

    g_DataCube_transpose_inplace_flt<<<gridSize, blockSize>>>(d_data, sizeX, sizeY, sizeZ);

    cudaDeviceSynchronize();

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Cuda error after kernel: %s\n", cudaGetErrorString(err));  
    }

    cudaMemcpy(data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    printf("Transposed Matrix:");

	// for (int i = 0 ; i < size; i+=32)
    // {
    //     if (i % 128 == 0) printf("\n");
    //     printf("%g \t%g \t%g \t%g \t%g \t%g \t%g \t%g\t\t", data[i+0], data[i+1], data[i+2], data[i+3], data[i+4], data[i+5], data[i+6], data[i+7]);
    //     printf("%g \t%g \t%g \t%g \t%g \t%g \t%g \t%g\t\t", data[i+8], data[i+9], data[i+10], data[i+11], data[i+12], data[i+13], data[i+14], data[i+15]);
    //     printf("%g \t%g \t%g \t%g \t%g \t%g \t%g \t%g\t\t", data[i+16], data[i+17], data[i+18], data[i+19], data[i+20], data[i+21], data[i+22], data[i+23]);
    //     printf("%g \t%g \t%g \t%g \t%g \t%g \t%g \t%g\n", data[i+24], data[i+25], data[i+26], data[i+27], data[i+28], data[i+29], data[i+30], data[i+31]);
    // }

	printf("\n");

    cudaFree(d_data);
}

void GPU_test_convolve(float *data, size_t size, const size_t *axis_size)
{
    float *d_data;
    uint8_t radius = 1;
    size_t n_iter = 3;

    cudaMalloc((void**)&d_data, size * sizeof(float));

    cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice);

    g_filter_gauss_Y_flt<<<dim3(1,1), dim3(32), 2 * ((n_iter + 1) * 2 * radius + 1) * sizeof(float)>>>(d_data, axis_size[0], axis_size[1], axis_size[2], radius, n_iter);
    cudaDeviceSynchronize();
}

void GPU_test_what_is_wrong(DataCube *self, char *data, char *maskdata, size_t data_size, const size_t *axis_size, const Array_dbl *kernels_spat, const Array_siz *kernels_spec, const double maskScaleXY, const noise_stat method, const double rms, const size_t cadence, const int range, const double threshold, const int scaleNoise, const noise_stat snStatistic, const int snRange)
{
    printf("CPU at %i %i %i: %.10e\n\n", 115, 193, 35, DataCube_get_data_flt(self, 115, 193, 35));

    size_t width = axis_size[0];
    size_t height = axis_size[1];
    size_t depth = axis_size[2];

    float *d_data;
    float *d_data_box;
    unsigned int *d_counter;
    char *d_mask_data;

    // Set up Bitmask space efficient on GPU one byte handles 8 entries in from the cube,
    // since here we only need to mask pixels, not differ between individual sources
    size_t d_mask_size = ((width + 7) / 8) * height * depth * sizeof(char);
    printf("Allocating %fMB for the mask\n", d_mask_size / (1024.0f * 1024.0f));
	cudaMalloc((void**)&d_mask_data, d_mask_size);
    cudaMemset(d_mask_data, 0, d_mask_size);


    cudaMalloc((void**)&d_data, data_size * sizeof(float));
    cudaMalloc((void**)&d_data_box, data_size * sizeof(float));
    cudaMalloc((void**)&d_counter, sizeof(unsigned int));

    cudaMemset(d_data_box, 0, data_size * sizeof(float));
    cudaMemset(d_counter, 0, sizeof(unsigned int));

    cudaMemcpy(d_data, data, data_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Cuda error at start: %s\n", cudaGetErrorString(err));  
    }

    cudaDeviceSynchronize();

    dim3 blockSizeXZ(256,1);
    dim3 gridSizeXZ(1,height,1);

    dim3 blockSizeY(64,8);
    dim3 gridSizeY((width + 7) / 8,1, 16);

    size_t radius_gauss;
    size_t radius_boxcar;
    size_t n_iter;

    const double FWHM_CONST = 2.0 * sqrt(2.0 * log(2.0));
    //optimal_filter_size_dbl(Array_dbl_get(kernels_spat, 0) / FWHM_CONST, &radius_gauss, &n_iter);
    optimal_filter_size_dbl(2, &radius_gauss, &n_iter);

    radius_boxcar = 3;
    printf("RadiusGauss: %lu, n_iter: %lu\n", radius_gauss, n_iter);

    g_cpyData_setMskScale1_rmBlnks_fltr_gX_bZ_flt_new<<<gridSizeXZ, blockSizeXZ, ((radius_gauss > 0 ? 1 : 0) * (radius_gauss * 3 + width * 2) + (radius_boxcar > 0 ? 1 : 0) * width) * sizeof(float)>>>
    (d_data, d_data_box, d_mask_data, width, height, depth, maskScaleXY * rms, radius_gauss, radius_boxcar, n_iter);

    g_filter_gauss_Y_flt_new<<<gridSizeY, blockSizeY, (9 * height + 10 * radius_gauss) * sizeof(float)>>>(d_data_box, width, height, depth, radius_gauss, n_iter);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after Y Kernel: %s\n", cudaGetErrorString(err));    
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after kernel: %s\n", cudaGetErrorString(err));
    }
    
    DataCube *maskCube = DataCube_blank(DataCube_get_axis_size(self, 0), DataCube_get_axis_size(self, 1), DataCube_get_axis_size(self, 2), 8, false);
	DataCube_copy_wcs(self, maskCube);
	DataCube_puthd_str(maskCube, "BUNIT", " ");

    // Smoothing required; create a copy of the original cube
    DataCube *smoothedCube = DataCube_copy(self);
    
    // Set flux of already detected pixels to maskScaleXY * rms
    if(maskScaleXY >= 0.0) DataCube_set_masked_8(smoothedCube, maskCube, maskScaleXY * rms);

    // Spatial and spectral smoothing
    //if(Array_dbl_get(kernels_spat, i) > 0.0) DataCube_gaussian_filter(smoothedCube, Array_dbl_get(kernels_spat, i) / FWHM_CONST);
    //if(Array_siz_get(kernels_spec, j) > 0)   DataCube_boxcar_filter(smoothedCube, Array_siz_get(kernels_spec, j) / 2);

    DataCube_boxcar_filter(smoothedCube, radius_boxcar);
    DataCube_gaussian_filter(smoothedCube, 2);

    cudaDeviceSynchronize();

    cudaMemcpy(data, d_data_box, data_size * sizeof(float), cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after back cpy: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    for (int i = 10; i--;)
    {
        int x = rand() % width;
        int y = rand() % height;
        int z = rand() % depth;

        printf("CPU at %i %i %i: %.10e\n", x, y, z, DataCube_get_data_flt(smoothedCube, x, y, z));
        printf("GPU at %i %i %i: %.10e\n", x, y, z, DataCube_get_data_flt(self, x, y, z));
        if (DataCube_get_data_flt(smoothedCube, x, y, z) == DataCube_get_data_flt(self, x, y, z))
        {
            printf("OK\n");
        }
        else
        {
            printf("Wrong!\n");
        }
        printf("\n");
    }
}

void GPU_gaufit(const float *d_data, const size_t size, float *d_sigma_out, const size_t cadence, const int range)
{
    // Determine maximum and minimum
    float *data_min;
	float *data_max;
    unsigned int *histogram;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error at start: %s\n", cudaGetErrorString(err));    
    }

    cudaMalloc((void**)&data_min,sizeof(float));
    cudaMalloc((void**)&data_max,sizeof(float));

    float inf = INFINITY;
    float ninf = -INFINITY;
    cudaMemcpy(data_min, &inf, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data_max, &ninf, sizeof(float), cudaMemcpyHostToDevice);

    g_find_max_min<<<1024, 1024>>>(d_data, size, 1, data_min, data_max);
	// Create initial histogram
	const size_t n_bins = 101;

    cudaMalloc((void**)&histogram, n_bins * sizeof(unsigned int));
    cudaMemset(histogram, 0, n_bins * sizeof(unsigned int));

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after malloc: %s\n", cudaGetErrorString(err));    
    }

    g_create_histogram_flt<<<1024, 1024, n_bins * sizeof(unsigned int)>>>(d_data, size, range, histogram, n_bins, data_min, data_max, cadence);

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after first hist creation: %s\n", cudaGetErrorString(err));    
    }

    g_scale_max_min_with_second_moment_flt<<<1, 32>>>(histogram, n_bins, range, data_min, data_max);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after moment scale: %s\n", cudaGetErrorString(err));    
    }
    
    // Regenerate histogram with new scaling
    cudaMemset(histogram, 0, n_bins * sizeof(unsigned int));
    g_create_histogram_flt<<<1024, 1024, n_bins * sizeof(unsigned int)>>>(d_data, size, range, histogram, n_bins, data_min, data_max, cadence);
    g_calc_sigma_flt<<<1, 32>>>(histogram, n_bins, range, data_min, data_max, d_sigma_out);

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error at End: %s\n", cudaGetErrorString(err));    
    }
}

void GPU_test_cpy_msk_1_to_8()
{
    u_int8_t data[4] = {1,3,255,128+32+4+1};
    u_int8_t result[28];

    printf("Array to get median: ");

	for (int i = 0 ; i < 4; i++){printf("%i ", data[i]);}

	printf("\n");

    char *mask1;
    char *mask8;

    cudaMalloc((void**)&mask1, 4 * sizeof(char));
    cudaMalloc((void**)&mask8, 28 * sizeof(char));

    cudaMemset(mask1, 0, 4 * sizeof(char));
    cudaMemset(mask8, 0, 28 * sizeof(char));

    cudaMemcpy(mask1, data, 4 * sizeof(char), cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Cuda error at start: %s\n", cudaGetErrorString(err));  
    }

    cudaDeviceSynchronize();

    dim3 blockSize(32,32);
    dim3 gridSize(1,1);

    g_DataCube_copy_mask_1_to_8<<<gridSize, blockSize>>>(mask8, mask1, 7,2,2);

    cudaDeviceSynchronize();

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Cuda error at start: %s\n", cudaGetErrorString(err));  
    }

    cudaMemcpy(result, mask8, 28 * sizeof(char), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    printf("Array to get median: ");

	for (int i = 0 ; i < 28; i++){printf("%i ", result[i]);}

	printf("\n");

    cudaFree(mask1);
    cudaFree(mask8);
}



void GPU_DataCube_filter_flt(char *data, char *maskdata, size_t data_size, const size_t *axis_size, const Array_dbl *kernels_spat, const Array_siz *kernels_spec, const double maskScaleXY, const noise_stat method, const double rms, const size_t cadence, const int range, const double threshold, const int scaleNoise, const noise_stat snStatistic, const int snRange)
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

    // For STAT_MAD
    float *d_data_min;
    float *d_data_max;
    float *d_median_arr;
    float *d_median_arr_swap;
    unsigned int mad_precision = 64;
    unsigned int *d_bins;
    unsigned int *d_med_total_values;
    unsigned int *d_med_counter;

    // size_t free_bytes, total_bytes;
    // cudaError_t cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);

    // if (cuda_status == cudaSuccess) {
    //     printf("Total GPU Memory: %fMB\n", total_bytes / (1024.0f * 1024.0f));
    //     printf("Free GPU Memory: %fMB\n", free_bytes / (1024.0f * 1024.0f));
    // } else {
    //     printf("cudaMemGetInfo failed: %s\n", cudaGetErrorString(cuda_status));
    // }

    // Allocate and copy values from Host to Device
    printf("Allocating %fMB for the data\n", data_size * sizeof(float) / (1024.0f * 1024.0f));
    cudaMalloc((void**)&d_data, data_size * sizeof(float));

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after malloc data: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);

    // if (cuda_status == cudaSuccess) {
    //     printf("Total GPU Memory: %fMB\n", total_bytes / (1024.0f * 1024.0f));
    //     printf("Free GPU Memory: %fMB\n", free_bytes / (1024.0f * 1024.0f));
    // } else {
    //     printf("cudaMemGetInfo failed: %s\n", cudaGetErrorString(cuda_status));
    // }

    printf("Allocating %fMB for the box\n", data_size * sizeof(float) / (1024.0f * 1024.0f));
    cudaMalloc((void**)&d_data_box, data_size * sizeof(float));

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after malloc box: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);

    // if (cuda_status == cudaSuccess) {
    //     printf("Total GPU Memory: %fMB\n", total_bytes / (1024.0f * 1024.0f));
    //     printf("Free GPU Memory: %fMB\n", free_bytes / (1024.0f * 1024.0f));
    // } else {
    //     printf("cudaMemGetInfo failed: %s\n", cudaGetErrorString(cuda_status));
    // }

    printf("Allocating %fMB for the duo\n", 2 * sizeof(float) / (1024.0f * 1024.0f));
    cudaMalloc((void**)&d_data_duo, 2 * sizeof(float));

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after malloc duo: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);

    // if (cuda_status == cudaSuccess) {
    //     printf("Total GPU Memory: %fMB\n", total_bytes / (1024.0f * 1024.0f));
    //     printf("Free GPU Memory: %fMB\n", free_bytes / (1024.0f * 1024.0f));
    // } else {
    //     printf("cudaMemGetInfo failed: %s\n", cudaGetErrorString(cuda_status));
    // }

    if (method == NOISE_STAT_MAD)
    {
        size_t medArraySize = ((range == 0) ? (data_size / cadence) : (data_size / (2 * cadence)));
        printf("med Array size is: %lu with range: %i\n", medArraySize, range);
        printf("Allocating %fMB for median calculation\n", medArraySize * sizeof(float) / (1024.0f * 1024.0f));
        cudaMalloc((void**)&d_median_arr, medArraySize * sizeof(float));
        cudaMalloc((void**)&d_median_arr_swap, medArraySize * sizeof(float));
        cudaMalloc((void**)&d_med_total_values, sizeof(unsigned int));
        cudaMalloc((void**)&d_med_counter, sizeof(unsigned int));
        cudaMalloc((void**)&d_bins, mad_precision * sizeof(unsigned int));
        cudaMalloc((void**)&d_data_min, sizeof(float));
        cudaMalloc((void**)&d_data_max, sizeof(float));
        
        cudaMemset(d_med_total_values, 0, sizeof(unsigned int));
        cudaMemset(d_med_counter, 0, sizeof(unsigned int));

        float inf = INFINITY;
        float ninf = -INFINITY;
        cudaMemcpy(d_data_min, &inf, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_data_max, &ninf, sizeof(float), cudaMemcpyHostToDevice);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Cuda error after malloc for median: %s\n", cudaGetErrorString(err));
            exit(1);
        }
    }

    // Set up Bitmask space efficient on GPU one byte handles 8 entries in from the cube,
    // since here we only need to mask pixels, not differ between individual sources
    size_t d_mask_size = ((width + 7) / 8) * height * depth * sizeof(char);
    printf("Allocating %fMB for the mask\n", d_mask_size / (1024.0f * 1024.0f));
	cudaMalloc((void**)&d_mask_data, d_mask_size);
    cudaMemset(d_mask_data, 0, d_mask_size);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after malloc mask: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);

    // if (cuda_status == cudaSuccess) {
    //     printf("Total GPU Memory: %fMB\n", total_bytes / (1024.0f * 1024.0f));
    //     printf("Free GPU Memory: %fMB\n", free_bytes / (1024.0f * 1024.0f));
    // } else {
    //     printf("cudaMemGetInfo failed: %s\n", cudaGetErrorString(cuda_status));
    // }

    cudaMemcpy(d_data, data, data_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_box, d_data, data_size * sizeof(float), cudaMemcpyDeviceToDevice);

    // Set up Bitmask space efficient on GPU one byte handles 8 entries in from the cube,
    // since here we only need to mask pixels, not differ between individual sources

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after copy: %s\n", cudaGetErrorString(err));    
    }

    //GPU_DataCube_copy_mask_8_to_1(d_mask_data, d_original_mask, axis_size);


    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after seting up mask data: %s\n", cudaGetErrorString(err));    
    }

    dim3 blockSizeMS(256,1);
    dim3 gridSizeMS((width + blockSizeMS.x - 1) / blockSizeMS.x,
                    (height + blockSizeMS.y - 1) / blockSizeMS.y);

    dim3 blockSizeM1(256,1);
    dim3 gridSizeM1((width + blockSizeM1.x - 1) / blockSizeM1.x,
                    (height + blockSizeM1.y - 1) / blockSizeM1.y);

    dim3 blockSizeX(32,16);
    dim3 gridSizeX(1,(height + 1) / 2, 2);

    dim3 blockSizeXZ(256,1);
    dim3 gridSizeXZ(1,height,8);

    dim3 blockSizeY(64,8);
    dim3 gridSizeY((width + 7) / 8,1, 16);

    dim3 blockSizeYC(32,16);
    dim3 gridSizeYC((width + blockSizeYC.x - 1) / blockSizeYC.x, 1, 64);

    dim3 blockSizeZ(256,1);
    dim3 gridSizeZ((width + blockSizeZ.x - 1) / blockSizeZ.x,
                    (height + blockSizeZ.y - 1) / blockSizeZ.y);

    dim3 blockSizeT(32, 16);
    dim3 gridSizeT((width + blockSizeT.x - 1) / blockSizeT.x,
                    (height + blockSizeT.x - 1) / blockSizeT.x);


    dim3 blockSizeNoise(1024);
    dim3 gridSizeNoise(1024);

    float h_min;
    float h_max;
    if (method == NOISE_STAT_MAD)
    {
        g_find_max_min<<<512, 1024>>>(d_data, data_size, cadence, d_data_min, d_data_max);
        cudaMemcpy(&h_min, d_data_min, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_max, d_data_max, sizeof(float), cudaMemcpyDeviceToHost);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error after finding min and max: %s\n", cudaGetErrorString(err));    
    }

    cudaDeviceSynchronize();

    for(size_t i = 0; i < Array_dbl_get_size(kernels_spat); ++i)
	{
        for(size_t j = 0; j < Array_siz_get_size(kernels_spec); ++j)
		{
            cudaMemset(d_data_duo, 0, 2 * sizeof(float));
            if (method == NOISE_STAT_MAD)
            {
                cudaMemset(d_bins, 0, mad_precision * sizeof(unsigned int));
                cudaMemset(d_med_counter, 0, sizeof(unsigned int));
                cudaMemset(d_med_total_values, 0, sizeof(unsigned int));
            }

            if (Array_dbl_get(kernels_spat, i) || Array_siz_get(kernels_spec, j))
            {
                optimal_filter_size_dbl(Array_dbl_get(kernels_spat, i) / FWHM_CONST, &radius_gauss, &n_iter);
                radius_boxcar = Array_siz_get(kernels_spec, j) / 2;

                printf("[%.1f] x [%lu]\n", Array_dbl_get(kernels_spat, i), Array_siz_get(kernels_spec, j));

                printf("Gauss: %lu, Boxcar: %lu\n", radius_gauss, radius_boxcar);

                // Copy maskScaled data from d_data to d_data_box and replace blanks
                if (maskScaleXY >= 0.0 && false)
                {
                    printf("Starting Kernels\n");
                    err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        printf("Cuda error before starting next round: %s\n", cudaGetErrorString(err));    
                    }

                    if (radius_boxcar > 0 && true)
                    {
                        printf("Launching data copy, flux masking, NAN deletion and Boxcar Z…\n");

                        g_copyData_setMaskedScale1_removeBlanks_filter_boxcar_Z_flt<<<gridSizeZ, blockSizeZ, (2 * radius_boxcar + 1) * blockSizeZ.x * blockSizeZ.y * sizeof(float)>>>(d_data_box, d_data, d_mask_data, width, height, depth, maskScaleXY * rms, radius_boxcar);
                        cudaDeviceSynchronize();

                        err = cudaGetLastError();
                        if (err != cudaSuccess)
                        {
                            printf("Cuda error after Z Kernel: %s\n", cudaGetErrorString(err));    
                        }
                    }
                    else
                    {
                        printf("Launching data copy, flux masking and NAN deletion\n");
                        //g_copyData_setMaskedScale8_removeBlanks<<<gridSizeMS, blockSizeMS>>>(d_data_box, d_data, d_original_mask, width, height, depth, maskScaleXY * rms);
                        g_copyData_setMaskedScale1_removeBlanks<<<gridSizeMS, blockSizeMS>>>(d_data_box, d_data, d_mask_data, width, height, depth, maskScaleXY * rms);
                        cudaDeviceSynchronize();
                    }

                    err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        printf("Cuda error after Mask Kernel: %s\n", cudaGetErrorString(err));    
                    }
                }
                else
                {
                    //g_copyData_removeBlanks<<<gridSizeMS, blockSizeMS>>>(d_data_box, d_data, width, height, depth);
                    printf("Launching data copy, flux masking, NAN deletion Gauss X and Boxcar Z…\n");
                    g_cpyData_setMskScale1_rmBlnks_fltr_gX_bZ_flt_new<<<gridSizeXZ, blockSizeXZ, ((radius_gauss > 0 ? 1 : 0) * (radius_gauss * 3 + width * 2) + (radius_boxcar > 0 ? 1 : 0) * width) * sizeof(float)>>>
                    (d_data, d_data_box, d_mask_data, width, height, depth, maskScaleXY * rms, radius_gauss, radius_boxcar, n_iter);
                    err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        printf("Cuda error after X-Z Kernel: %s\n", cudaGetErrorString(err));    
                    }
                    cudaDeviceSynchronize();
                }
                
                if(radius_gauss > 0)
                {
                    // printf("Launching Gauss X…\n");
                    // g_filter_gauss_X_flt_new<<<gridSizeX, blockSizeX, (4 * width + 5 * radius_gauss) * sizeof(float)>>>(d_data_box, width, height, depth, radius_gauss, n_iter);
                    // cudaDeviceSynchronize();

                    // err = cudaGetLastError();
                    // if (err != cudaSuccess)
                    // {
                    //     printf("Cuda error after X Kernel: %s\n", cudaGetErrorString(err));    
                    // }
                }
                    
                if(radius_gauss > 0)
                {
                    // printf("Transposing...\n");

                    // g_DataCube_transpose_inplace_flt<<<gridSizeT, blockSizeT>>>(d_data_box, width, height, depth);

                    // printf("Launching Gauss X for Y…\n");
                    // g_filter_gauss_X_flt_new<<<gridSizeX, blockSizeX, (4 * width + 5 * radius_gauss) * sizeof(float)>>>(d_data_box, width, height, depth, radius_gauss, n_iter);

                    // printf("Transposing...\n");

                    // g_DataCube_transpose_inplace_flt<<<gridSizeT, blockSizeT>>>(d_data_box, width, height, depth);


                    // printf("Launching Gauss Y…\n");

                    //g_filter_gauss_Y_flt<<<gridSizeYC, blockSizeYC, ((blockSizeYC.x + 1) * (n_iter * 2 * radius_gauss + 1) + blockSizeYC.x * (blockSizeYC.y - 1)) * sizeof(float)>>>(d_data_box, width, height, depth, radius_gauss, n_iter);
                    g_filter_gauss_Y_flt_new<<<gridSizeY, blockSizeY, (9 * height + 10 * radius_gauss) * sizeof(float)>>>(d_data_box, width, height, depth, radius_gauss, n_iter);
                    cudaDeviceSynchronize();

                    err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        printf("Cuda error after Y Kernel: %s\n", cudaGetErrorString(err));    
                    }
                }

                // if(radius_boxcar > 0 && false) 
                // {
                //     printf("Launching Boxcar Z…\n");

                //     g_filter_boxcar_Z_flt_new<<<gridSizeZ, blockSizeZ, (2 * radius_boxcar + 1) * blockSizeZ.x * blockSizeZ.y * sizeof(float)>>>(d_data_box, width, height, depth, radius_boxcar);
                //     cudaDeviceSynchronize();

                //     err = cudaGetLastError();
                //     if (err != cudaSuccess)
                //     {
                //         printf("Cuda error after Z Kernel: %s\n", cudaGetErrorString(err));    
                //     }
                // }

                
                printf("Starting Kernels for Mask\n");
                g_addBlanks<<<gridSizeM1, blockSizeM1>>>(d_data_box, d_data, width, height, depth);

                if (scaleNoise == 1)
                {
                    printf("Correcting for noise variations along spectral axis.\n");
                    if (snStatistic == NOISE_STAT_STD)
                    {
                        g_std_dev_val_flt<<<axis_size[2], blockSizeNoise, blockSizeNoise.x * 2 * sizeof(float)>>>(d_data_box, d_data_duo, axis_size[0] * axis_size[1], 0.0f, 1, snRange, scaleNoise);
                    }
                    else if (snStatistic == NOISE_STAT_MAD)
                    {
                        uint scale_precision = 4096;
                        uint max_values = 512;
                        
                        g_mad_val_hist_flt_scale_noise<<<axis_size[2], 32, scale_precision * sizeof(uint) + max_values * sizeof(float)>>>(d_data_box, axis_size[0] * axis_size[1], 0.0f, 1, snRange, scale_precision, max_values);
                        
                        cudaDeviceSynchronize();
                        err = cudaGetLastError();
                        if (err != cudaSuccess)
                        {
                            printf("Cuda error after noise scaling: %s\n", cudaGetErrorString(err));
                        }

                        cudaDeviceSynchronize();
                    }
                }

                if (method == NOISE_STAT_STD)
                {
                    g_std_dev_val_flt<<<gridSizeNoise, blockSizeNoise, blockSizeNoise.x * 2 * sizeof(float)>>>(d_data_box, d_data_duo, data_size, 0.0f, cadence, range, 0);
                    g_std_dev_val_flt_final_step<<<1,1>>>(d_data_duo);
                }
                else if (method == NOISE_STAT_MAD)
                {
                    // g_mad_val_flt<<<1, blockSizeNoise, (blockSizeNoise.x + 1) * sizeof(float)>>>(d_data_box, d_median_arr, d_med_counter, data_size, (range == 0) ? (data_size / cadence) : (data_size / (2 * cadence)), 0.0f, cadence, range);
                    // unsigned int count;
                    // cudaMemcpy(&count, d_med_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                    // printf("Median values count: %u\n", count);
                    // g_mad_val_flt_final_step<<<1,1024>>>(d_median_arr, d_med_counter);
                    // cudaMemcpy(d_data_duo, d_median_arr, 1 * sizeof(double), cudaMemcpyDeviceToDevice);
                    printf("Starting MAD\n");
                    //g_mad_val_hist_flt<<<gridSizeNoise, blockSizeNoise, mad_precision * sizeof(unsigned int)>>>(d_data_box, data_size, d_bins, d_med_total_values, 0.0f, cadence, range, mad_precision, d_data_min, d_data_max);
                    //g_mad_val_hist_flt_cpy_nth_bin<<<gridSizeNoise, 32, 32 * sizeof(float)>>>(d_data_box, data_size, d_median_arr, d_bins, d_med_total_values, d_med_counter, 0.0f, cadence, range, mad_precision, d_data_min, d_data_max); 

                    // unsigned int med_counter;
                    // cudaMemcpy(&med_counter, d_med_counter, sizeof(unsigned int),cudaMemcpyDeviceToHost);
                    // cudaMemset(d_bins, 0, mad_precision * sizeof(unsigned int));
                    // cudaMemset(d_med_counter, 0, sizeof(unsigned int));
                    // cudaMemset(d_med_total_values, 0, sizeof(unsigned int));

                    // g_mad_val_hist_flt<<<gridSizeNoise, blockSizeNoise, mad_precision * sizeof(unsigned int)>>>(d_median_arr, med_counter, d_bins, d_med_total_values, 0.0f, cadence, range, mad_precision, d_data_min, d_data_max);
                    // g_mad_val_hist_flt_cpy_nth_bin<<<gridSizeNoise, blockSizeNoise, blockSizeNoise.x * sizeof(float)>>>(d_data_box, data_size, d_median_arr, d_bins, d_med_total_values, d_med_counter, 0.0f, cadence, range, mad_precision, d_data_min, d_data_max);
                    
                    //g_mad_val_hist_flt_final_step<<<1,1>>>(d_median_arr, d_med_counter, d_med_total_values, d_bins);
                    GPU_MAD(d_data_box, data_size, d_data_duo, snRange, mad_precision, cadence, h_min, h_max);

                    cudaDeviceSynchronize();

                    //cudaMemcpy(d_data_duo, d_median_arr, 1 * sizeof(float), cudaMemcpyDeviceToDevice);
                }
                else
                {
                    GPU_gaufit(d_data_box, data_size, d_data_duo, cadence, range);
                }

                float noise[2] = {0,0};
                //cudaMemcpy(noise, d_data_duo, 2 * sizeof(float), cudaMemcpyDeviceToHost);

                cudaMemcpy(noise, d_data_duo, 2 * sizeof(float), cudaMemcpyDeviceToHost);

                printf("Final noise: %.3e\n\n", noise[0]);

                //g_Mask8<<<gridSizeMS, blockSizeMS>>>(d_data_box, d_original_mask, width, height, depth, threshold, d_data_duo, 1);
                g_Mask1<<<gridSizeM1, blockSizeM1>>>(d_data_box, d_mask_data, width, height, depth, threshold, d_data_duo);

                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    printf("Cuda error after noise calc: %s\n", cudaGetErrorString(err));
                }
            }
            else
            {
                if (method == NOISE_STAT_STD)
                {
                    err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        printf("Cuda error before first noise calc: %s\n", cudaGetErrorString(err));    
                    }
                    g_std_dev_val_flt<<<gridSizeNoise, blockSizeNoise, 1024 * 2 * sizeof(float)>>>(d_data, d_data_duo, data_size, 0.0f, cadence, range, 0);
                    g_std_dev_val_flt_final_step<<<1,1>>>(d_data_duo);
                }
                else if (method == NOISE_STAT_MAD)
                {
                    // cudaDeviceSynchronize();
                    // g_mad_val_flt<<<1, blockSizeNoise, (blockSizeNoise.x + 1) * sizeof(float)>>>(d_data_box, d_median_arr, d_med_counter, data_size, ((range == 0) ? (data_size / cadence) : (data_size / (2 * cadence))), 0.0f, cadence, range);
                    // cudaDeviceSynchronize();
                    // unsigned int count;
                    // cudaMemcpy(&count, d_med_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                    // printf("Median values count: %u\n", count);
                    // err = cudaGetLastError();
                    // if (err != cudaSuccess)
                    // {
                    //     printf("Cuda error after first noise step: %s\n", cudaGetErrorString(err));    
                    // }
                    // cudaDeviceSynchronize();
                    // g_mad_val_flt_final_step<<<1,1024>>>(d_median_arr, d_med_counter);
                    // err = cudaGetLastError();
                    // if (err != cudaSuccess)
                    // {
                    //     printf("Cuda error after second noise step: %s\n", cudaGetErrorString(err));    
                    // }
                    // cudaDeviceSynchronize();
                    // cudaMemcpy(d_data_duo, d_median_arr, 1 * sizeof(float), cudaMemcpyDeviceToDevice);
                    // cudaDeviceSynchronize();

                    printf("Starting MAD\n");
                    //g_mad_val_hist_flt<<<gridSizeNoise, blockSizeNoise, mad_precision * sizeof(unsigned int)>>>(d_data, data_size, d_bins, d_med_total_values, 0.0f, cadence, range, mad_precision, d_data_min, d_data_max);
                    //g_mad_val_hist_flt_cpy_nth_bin<<<gridSizeNoise, 32, 32 * sizeof(float)>>>(d_data, data_size, d_median_arr, d_bins, d_med_total_values, d_med_counter, 0.0f, cadence, range, mad_precision, d_data_min, d_data_max);
                    //g_mad_val_hist_flt_final_step<<<1,1>>>(d_median_arr, d_med_counter, d_med_total_values, d_bins);

                    GPU_MAD(d_data, data_size, d_data_duo, snRange, mad_precision, cadence, h_min, h_max);

                    cudaDeviceSynchronize();

                    //cudaMemcpy(d_data_duo, d_median_arr, 1 * sizeof(float), cudaMemcpyDeviceToDevice);
                }
                else
                {
                    GPU_gaufit(d_data, data_size, d_data_duo, cadence, range);
                }

                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    printf("Cuda error after first noise calc: %s\n", cudaGetErrorString(err));
                    exit(1);
                }

                float noise[2] = {0,0};
                cudaMemcpy(noise, d_data_duo, 2 * sizeof(float), cudaMemcpyDeviceToHost);

                printf("Initial noise: %.3e\n\n", noise[0]);

                //g_Mask8<<<gridSizeMS, blockSizeMS>>>(d_data, d_original_mask, width, height, depth, threshold, d_data_duo, 1);
                g_Mask1<<<gridSizeM1, blockSizeM1>>>(d_data, d_mask_data, width, height, depth, threshold, d_data_duo);
                cudaDeviceSynchronize();

                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    printf("Cuda error after first Masking: %s\n", cudaGetErrorString(err));
                }
            }
        }
    }

    //g_DataCube_copy_mask_1_to_8<<<gridSizeMS, blockSizeMS>>>(d_original_mask, d_mask_data, width, height, depth);
    cudaDeviceSynchronize();
    cudaMemcpy(maskdata, d_mask_data, d_mask_size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(d_data);
    cudaFree(d_data_box);
    cudaFree(d_mask_data);
    cudaFree(d_data_duo);

    if(method == NOISE_STAT_MAD)
    {
        cudaFree(d_data_min);
        cudaFree(d_data_max);
        cudaFree(d_median_arr);
        cudaFree(d_median_arr_swap);
        cudaFree(d_bins);
        cudaFree(d_med_total_values);
        cudaFree(d_med_counter);
    }

    printf("Finished GPU\n");
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
        if(IS_NAN(data[index])) 
        {
            data_box[index] = NAN;
        }
        index += width * height;
    }
}

__global__ void g_copyData_setMaskedScale1_removeBlanks(float *data_box, float *data, char *maskData1, const size_t width, const size_t height, const size_t depth, const float value)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = 0;

    size_t index = x + y * width;
    size_t index1 = (x / 8) + y * ((width + 7) / 8);

    if (x < width && y < height)
    {
        while (z < depth)
        {
            data_box[index] = ((maskData1[index1] & (1 << (7 - (x % 8)))) >> (7 - (x % 8))) * copysign(value, FILTER_NAN(data[index])) + (((maskData1[index1] & (1 << (7 - (x % 8)))) >> (7 - (x % 8))) ^ 1) * FILTER_NAN(data[index]);

            index += width * height;
            index1 += ((width + 7) / 8) * height;
            z += 1;
        }
    }
}

__global__ void g_copyData_setMaskedScale1_removeBlanks_filter_gX_bcZ_flt(float *data_box, float *data, char *maskData1, const size_t width, const size_t height, const size_t depth, const float value, const size_t radius_g, const size_t radius_bc, const size_t n_iter)
{
    const size_t localY = threadIdx.y / 8;
    size_t y = blockIdx.y * 2 + localY;
    const size_t x0 = threadIdx.x + blockDim.x * (threadIdx.y % 8);
    size_t x = x0;
    size_t z = blockIdx.z;

    const size_t xPage = 8 * blockDim.x;

    extern __shared__ float s_data_GX_flt_GnB[];
    float *s_data_bc = s_data_GX_flt_GnB;
    float *s_data_src = s_data_bc + width + radius_g * (1 + localY) + width * localY;
    float *s_data_dst = s_data_src + width * 2 + radius_g * 2;

    if (x0 == 0)
    {
        for (int i = radius_g; i--;)
        {
            *(s_data_GX_flt_GnB + i) = *(s_data_src + width + i) = *(s_data_dst + width + i) = 0.0f;
        }
    }

    while (z < depth)
    {
        while (y < height && x < width)
        {
            *(s_data_src + x) = data[x + y * width + z * width * height];
            x += xPage;
        }
        x = x0;
        __syncthreads();

        for (int i = n_iter; i--;)
        {
            while (y < height && x < width)
            {
                *(s_data_dst + x) = *(s_data_src + x);
                for (int i = radius_g; i--;)
                {
                    *(s_data_dst + x) += *(s_data_src + x + (i + 1)) + *(s_data_src + x - (i + 1));
                }
                *(s_data_dst + x) /= (2 * radius_g + 1);
                x += xPage;
            }
            x = x0;
            float *tmp = s_data_src;
            s_data_src = s_data_dst;
            s_data_dst = tmp;
            __syncthreads();
        }

        while (y < height && x < width)
        {
            data[x + y * width + z * width * height] = *(s_data_src + x);
            x += xPage;
        }
        x = x0;
        z+=gridDim.z;
    }
}

__global__ void g_copyData_setMaskedScale1_removeBlanks_filter_boxcar_Z_flt(float *data_box, float *data, char *maskData1, const uint16_t width, const uint16_t height, const uint16_t depth, const float maskValue, const size_t radius)
{
    const uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint16_t y = blockIdx.y * blockDim.y + threadIdx.y;

    const int filter_size = (2 * radius + 1);
    float value = 0.0f;
    const float invers = 1.0f / filter_size;
    extern __shared__ float s_data_BZ_flt_new[];
    float *s_data_start = s_data_BZ_flt_new + (threadIdx.x + threadIdx.y * blockDim.x) * filter_size;
    u_int8_t ptr = 0;    

    if (x < width && y < height)
    {
        for (size_t z = depth; z--;)
        {
            if (z < depth - filter_size)
            {
                //value -= data[x + y * width + (z + filter_size) * width * height];
                value -= s_data_start[ptr];
            }

            float locvar = FILTER_NAN(data[x + y * width + z * width * height]);
            char maskvar = maskData1[(x / 8) + y * ((width + 7) / 8) + z * ((width + 7) / 8) * height];

            locvar = ((maskvar & (1 << (7 - (x % 8)))) >> (7 - (x % 8))) * copysign(maskValue, locvar) + (((maskvar & (1 << (7 - (x % 8)))) >> (7 - (x % 8))) ^ 1) * locvar;

            value += s_data_start[ptr++] = locvar;
            ptr = ptr % filter_size;

            if (z < depth - radius)
            {
                data_box[x + y * width + (z + radius) * width * height] = value * invers;
            }
        }

        for (size_t z = radius; z--;)
        {
            //value -= data[x + y * width + (z + radius + 1) * width * height];
            value -= s_data_start[ptr++];
            ptr = ptr % filter_size;
            data_box[x + y * width + z * width * height] = value * invers;
        }
    }
}

__global__ void g_copyData_setMaskedScale8_removeBlanks(float *data_box, float *data, char *maskData8, const size_t width, const size_t height, const size_t depth, const float value)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {return;}

    size_t index = x + y * width;
    float currentData;

    while (index < width * height * depth)
    {
        currentData = data[index];
        data_box[index]= ((int8_t)(maskData8[index])) * copysign(value, currentData) + ((int8_t)(maskData8[index]) ^ 1) * FILTER_NAN(currentData);
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
        //size_t index = x + y * width + z * width * height;
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

__global__ void g_filter_gauss_X_flt_new(float *data, const size_t width, const size_t height, const size_t depth, const size_t radius, const size_t n_iter)
{
    const size_t localY = threadIdx.y / 8;
    size_t y = blockIdx.y * 2 + localY;
    const size_t x0 = threadIdx.x + blockDim.x * (threadIdx.y % 8);
    size_t x = x0;
    size_t z = blockIdx.z;

    const size_t xPage = 8 * blockDim.x;

    extern __shared__ float s_data_GX_flt_new[];
    float *s_data_src = s_data_GX_flt_new + radius * (1 + localY) + width * localY;
    float *s_data_dst = s_data_src + width * 2 + radius * 2;

    if (x0 == 0)
    {
        for (int i = radius; i--;)
        {
            *(s_data_GX_flt_new + i) = *(s_data_src + width + i) = *(s_data_dst + width + i) = 0.0f;
        }
    }

    while (z < depth)
    {
        while (y < height && x < width)
        {
            *(s_data_src + x) = data[x + y * width + z * width * height];
            x += xPage;
        }
        x = x0;
        __syncthreads();

        for (int n = n_iter; n--;)
        {
            while (y < height && x < width)
            {
                *(s_data_dst + x) = *(s_data_src + x);
                for (int i = radius; i--;)
                {
                    *(s_data_dst + x) += *(s_data_src + x + (i + 1)) + *(s_data_src + x - (i + 1));
                }
                *(s_data_dst + x) /= (2 * radius + 1);
                x += xPage;
            }
            x = x0;
            float *tmp = s_data_src;
            s_data_src = s_data_dst;
            s_data_dst = tmp;
            __syncthreads();
        }

        while (y < height && x < width)
        {
            data[x + y * width + z * width * height] = *(s_data_src + x);
            x += xPage;
        }
        x = x0;
        z += gridDim.z;
    }
}

__global__ void g_cpyData_setMskScale1_rmBlnks_fltr_gX_bZ_flt_new(float *data_src, float *data_dst, char *maskData1, const uint16_t width, const uint16_t height, const uint16_t depth, const float maskValue, uint16_t radius_g, const uint16_t radius_b, const uint16_t n_iter)
{
    uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t z = (blockIdx.z * depth) / gridDim.z;
    const uint32_t max_z = ((blockIdx.z + 1) * depth) / gridDim.z;

    extern __shared__ float s_data_gX_bZ_flt_new[];
    const int filter_size_b = (2 * radius_b + 1);
    const int filter_size_g = (2 * radius_g + 1);
    const float invers_b = 1.0f / filter_size_b;
    const float invers_g = 1.0f / filter_size_g;
    float *s_data_src = s_data_gX_bZ_flt_new + radius_g;
    float *s_data_dst = s_data_src + width + radius_g;
    float *z_sum = s_data_gX_bZ_flt_new + 3 * radius_g * (int)__saturatef(radius_g) + 2 * width * (int)__saturatef(radius_g);

    if (y >= height) return;

    if (radius_b > 0)
    {
        while (x < width)
        {
            float locvar = FILTER_NAN(data_src[x + y * width + z * width * height]);
            char maskvar = maskData1[(x / 8) + y * ((width + 7) / 8) + z * ((width + 7) / 8) * height];

            if (maskValue >= 0) {locvar = ((maskvar & (1 << (7 - (x % 8)))) >> (7 - (x % 8))) * copysign(maskValue, locvar) + (((maskvar & (1 << (7 - (x % 8)))) >> (7 - (x % 8))) ^ 1) * locvar;}
            //else {locvar = FILTER_NAN(locvar);}

            z_sum[x] = locvar;
            x += blockDim.x;
        }
        x = blockIdx.x * blockDim.x + threadIdx.x;

        for (int i = radius_b; --i;)
        {
            while (x < width)
            {
                float locvar = FILTER_NAN(data_src[x + y * width + (z + i) * width * height]); // TODO protect for large radii with too small parts (i.e. z+i > depth)
                char maskvar = maskData1[(x / 8) + y * ((width + 7) / 8) + (z + i) * ((width + 7) / 8) * height];

                if (maskValue >= 0) {locvar = ((maskvar & (1 << (7 - (x % 8)))) >> (7 - (x % 8))) * copysign(maskValue, locvar) + (((maskvar & (1 << (7 - (x % 8)))) >> (7 - (x % 8))) ^ 1) * locvar;}
                //else {locvar = FILTER_NAN(locvar);}

                z_sum[x] += locvar;
                x += blockDim.x;
            }
        x = blockIdx.x * blockDim.x + threadIdx.x;
        }

        if (z != 0) // TODO protect this, if radius is bigger than z size of parts it will spill
        {
            for (int i = radius_b + 1; --i;)
            {
                while (x < width)
                {
                    float locvar = FILTER_NAN(data_src[x + y * width + (z - i) * width * height]); // TODO protect for large radii with too small parts (i.e. z-i < 0)
                    char maskvar = maskData1[(x / 8) + y * ((width + 7) / 8) + (z - i) * ((width + 7) / 8) * height];

                    if (maskValue > 0) {locvar = ((maskvar & (1 << (7 - (x % 8)))) >> (7 - (x % 8))) * copysign(maskValue, locvar) + (((maskvar & (1 << (7 - (x % 8)))) >> (7 - (x % 8))) ^ 1) * locvar;}
                    //else {locvar = FILTER_NAN(locvar);}

                    z_sum[x] += locvar;
                    x += blockDim.x;
                }
            x = blockIdx.x * blockDim.x + threadIdx.x;
            }
        }
    }

    if (radius_g > 0 && x < radius_g)
    {
        *(s_data_gX_bZ_flt_new + x) = *(s_data_src + width + x) = *(s_data_dst + width + x) = 0.0f;
    }

    while (z < max_z)
    {
        while (x < width)
        {
            float locvar = FILTER_NAN(radius_b > 0 ? (z + radius_b < depth ? data_src[x + y * width + (z + radius_b) * width * height] : 0.0f) : data_src[x + y * width + z * width * height]);
            char maskvar = radius_b > 0 ? (z + radius_b < depth ? maskData1[(x / 8) + y * ((width + 7) / 8) + (z + radius_b) * ((width + 7) / 8) * height] : 0) : maskData1[(x / 8) + y * ((width + 7) / 8) + z * ((width + 7) / 8) * height];

            if (maskValue > 0) {locvar = ((maskvar & (1 << (7 - (x % 8)))) >> (7 - (x % 8))) * copysign(maskValue, locvar) + (((maskvar & (1 << (7 - (x % 8)))) >> (7 - (x % 8))) ^ 1) * locvar;}
            else {locvar = FILTER_NAN(locvar);}

            if (radius_b > 0)
            {
                z_sum[x] += locvar;
            }

            if (radius_g > 0)
            {
                s_data_src[x] = radius_b > 0 ? z_sum[x] * invers_b : locvar;
            }
            else if (radius_g <= 0 && radius_b <= 0)
            {
                data_dst[x + y * width + z * width * height] = locvar;
            }

            x += blockDim.x;
        }
        x = blockIdx.x * blockDim.x + threadIdx.x;

        __syncthreads();

        if (radius_g > 0)
        {
            for (int i = n_iter; i--;)
            {
                while (x < width)
                {
                    float value = *(s_data_src + x);
                    for (int i = radius_g; i--;)
                    {
                        value += *(s_data_src + x + (i + 1)) + *(s_data_src + x - (i + 1));
                    }
                    *(s_data_dst + x) = value * invers_g;
                    x += blockDim.x;
                }
                x = blockIdx.x * blockDim.x + threadIdx.x;
                float *tmp = s_data_src;
                s_data_src = s_data_dst;
                s_data_dst = tmp;
                __syncthreads();
            }
        }

        if (radius_g > 0 || radius_b > 0)
        {
            while (x < width)
            {
                data_dst[x + y * width + z * width * height] = radius_g > 0 ? *(s_data_src + x) : z_sum[x] * invers_b;
                x += blockDim.x;
            }
            x = blockIdx.x * blockDim.x + threadIdx.x;
        }

        if (radius_b > 0 && z >= radius_b)
        {
            while (x < width)
            {
                float sub_val = FILTER_NAN(data_src[x + y * width + (z - radius_b) * width * height]);
                char maskvar = maskData1[(x / 8) + y * ((width + 7) / 8) + (z - radius_b) * ((width + 7) / 8) * height];

                if (maskValue > 0) {sub_val = ((maskvar & (1 << (7 - (x % 8)))) >> (7 - (x % 8))) * copysign(maskValue, sub_val) + (((maskvar & (1 << (7 - (x % 8)))) >> (7 - (x % 8))) ^ 1) * sub_val;}
                //else {sub_val = FILTER_NAN(sub_val);}

                z_sum[x] -= sub_val;
                x += blockDim.x;
            }
            x = blockIdx.x * blockDim.x + threadIdx.x;            
        }
        z++;
    }  
}

__global__ void g_filter_gauss_Y_flt(float *data, const size_t width, const size_t height, const size_t depth, const size_t radius, const size_t n_iter)
{
    const uint32_t threadcount = blockDim.x * blockDim.y;
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = threadIdx.y;
    uint32_t z = blockIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    const uint16_t loc_index = y * blockDim.x + threadIdx.x;

    const uint8_t filter_size = 2 * radius + 1;
    const float inverse_filter = 1.0f / filter_size;
    float inverse_datapoints = 1.0f / pow((2 * radius + 1), n_iter);

    extern __shared__ float s_data_GY_flt[];

    uint32_t *box_filter = (uint32_t*)s_data_GY_flt;
    uint32_t *tmp_filter = box_filter + (n_iter + 1) * 2 * radius + 1;
    float *s_data = s_data_GY_flt + n_iter * 2 * radius + 1;

    if (IS_ODD(n_iter)) 
    {
        uint32_t *tmp = tmp_filter;
        tmp_filter = box_filter;
        box_filter = tmp;
    }

    int loc_x = loc_index;
    while (loc_x < (n_iter + 1) * 2 * radius + 1)
    {
        box_filter[loc_x] = tmp_filter[loc_x] = loc_x > (n_iter * radius - 1) && loc_x < ((n_iter + 2) * radius + 1) ? 1 : 0;
        loc_x += threadcount;
    }

    __syncthreads();

    // if (loc_index == 0 && x == 0)
    // {
    //     printf("[");
    //     for (int i = 0; i < (n_iter + 1) * 2 * radius + 1; i++)
    //     {
    //         printf(" %u", box_filter[i]);
    //     }
    //     printf(" ]\n");
    // }
    // __syncthreads();

    for (int i = n_iter; --i;)
    {
        loc_x = loc_index + radius;
        while (loc_x < (n_iter + 1) * 2 * radius + 1 - radius)
        {
            uint32_t value = box_filter[loc_x];
            for (int k = radius; k--;)
            {
                value += box_filter[loc_x + (k + 1)] + box_filter[loc_x - (k + 1)];
            }
            tmp_filter[loc_x] = value;
            loc_x += threadcount;
        }
        uint32_t *tmp = tmp_filter;
        tmp_filter = box_filter;
        box_filter = tmp;
        __syncthreads();

        // if (loc_index == 0 && x == 0)
        // {
        //     printf("[");
        //     for (int i = 0; i < (n_iter + 1) * 2 * radius + 1; i++)
        //     {
        //         printf(" %u", box_filter[i]);
        //     }
        //     printf(" ]\n");
        // }
        // __syncthreads();
    }

    loc_x = loc_index;
    while (loc_x < n_iter * 2 * radius + 1)
    {
        tmp_filter[loc_x] = box_filter[loc_x + radius];
        loc_x += threadcount;
    }

    box_filter = tmp_filter;

    __syncthreads();

    // if (loc_index == 0 && x == 0 && y == 0 && z == 0)
    // {
    //     printf("[");
    //     for (int i = 0; i < n_iter * 2 * radius + 1; i++)
    //     {
    //         printf(" %u", box_filter[i]);
    //     }
    //     printf(" ]\n");
    // }

    // __syncthreads();

    while (z < depth)
    {
        int ptr = n_iter * radius + threadIdx.y;

        while (y < n_iter * radius)
        {
            s_data[y * blockDim.x + threadIdx.x] = 0.0f;
            y += blockDim.y;
        }

        y = threadIdx.y;

        while (y < n_iter * radius + 1 + (blockDim.y - 1) && y < height)
        {
            s_data[(n_iter * radius + y) * blockDim.x + threadIdx.x] = data[x + y * width + z * width * height];
            y += blockDim.y;
        }

        y = threadIdx.y;

        while (y < height + blockDim.y - 1)
        {
            if (y < height)
            {
                float value = s_data[ptr * blockDim.x + threadIdx.x] * box_filter[n_iter * radius];

                for (int i = n_iter * radius; i--;)
                {
                    value += s_data[((ptr + (i+1)) % (n_iter * 2 * radius + 1 + (blockDim.y - 1))) * blockDim.x + threadIdx.x] * box_filter[n_iter * radius + (i + 1)]
                            + s_data[((ptr + (n_iter * 2 * radius + 1 + (blockDim.y - 1)) - (i+1)) % (n_iter * 2 * radius + 1 + (blockDim.y - 1))) * blockDim.x + threadIdx.x] * box_filter[n_iter * radius - (i + 1)];
                }

                if (x == 0 && y == 0 && z == 0) {printf("value: %f inverse_Datapoints: %f n_iter: %lu\n", value, inverse_datapoints, n_iter);}

                data[x + y * width + z * width * height] = value * inverse_datapoints;
            }
            y += blockDim.y;

            __syncthreads();

            if (y < height)
            {
                ptr = (ptr + blockDim.y) % (n_iter * 2 * radius + 1 + (blockDim.y - 1));
                if (y < height - n_iter * radius)
                {
                    s_data[((ptr + n_iter * radius) * blockDim.x) % (n_iter * 2 * radius + 1 + (blockDim.y - 1)) + threadIdx.x] = data[x + (y + n_iter * radius) * width + z * width * height];
                }
                else
                {
                    s_data[((ptr + n_iter * radius) * blockDim.x) % (n_iter * 2 * radius + 1 + (blockDim.y - 1)) + threadIdx.x] = 0.0f;
                }
            }
            __syncthreads();
        }

        y = threadIdx.y;

        z += gridDim.z;
    }

    

    // float *s_data_dst = s_data_GY_flt + 2 * radius + height;

    // if (y == 0) 
    // {
    //     for (int i = radius; i--;)
    //     {
    //         *(s_data_GY_flt + i) = *(s_data_GY_flt + radius + height + i) = *(s_data_GY_flt + 2 * radius + 2 * height + i) = 0.0f;
    //     }
    // }

    // for (int z = 0; z < depth; ++z)
    // {
    //     //inline size_t index = x + y * width + z * width * height;
    //     while(x < width && y < height)
    //     {
    //         *(s_data_src + y) = data[x + y * width + z * width * height];
    //         y += blockDim.y;
    //     }

    //     y = threadIdx.y;
    //     __syncthreads();

    //     for (int n = n_iter; n--;)
    //     {
    //         while(x < width && y < height)
    //         {
    //             *(s_data_dst + y) = *(s_data_src + y);
    //             for (int i = radius; i--;)
    //             {
    //                 *(s_data_dst + y) += *(s_data_src + y + (i + 1)) + *(s_data_src + y - (i + 1));
    //             }
    //             *(s_data_dst + y) /= 2 * radius + 1;
    //             y += blockDim.y;
    //         }

    //         y = threadIdx.y;
    //         float *tmp = s_data_src;
    //         s_data_src = s_data_dst;
    //         s_data_dst = tmp;
    //         __syncthreads();
    //     }

    //     while(x < width && y < height)
    //     {
    //         data[x + y * width + z * width * height] = *(s_data_src + y);
    //         y += blockDim.y;
    //     }
    //     y = threadIdx.y;
    // }
}

__global__ void g_filter_gauss_Y_flt_new(float *data, const size_t width, const size_t height, const size_t depth, const size_t radius, const size_t n_iter)
{
    const int localX = threadIdx.x % 8;
    
    size_t x = blockIdx.x * 8 + localX;
    const size_t y0 = blockIdx.y * (height / (gridDim.y)) + threadIdx.y * 4 + threadIdx.x / 8;
    size_t y = y0;
    size_t z = blockIdx.z;

    const size_t yPage = 4 * blockDim.y;

    extern __shared__ float s_data_GY_flt_new[];
    float *s_data_src = s_data_GY_flt_new + radius;

    if (threadIdx.x < 9 && threadIdx.y == 0)
    {
        for (int i = radius; i--;)
        {
            *(s_data_GY_flt_new + i) = *(s_data_GY_flt_new + height * (threadIdx.x + 1) + radius * (threadIdx.x + 1) + i) = 0.0f;
        }
    }

    while (z < depth)
    {
        while (x < width && y < height)
        {
            *(s_data_src + localX * (height + radius) + y) = data[x + y * width + z * width * height];
            y += yPage;
        }
        y = y0;
        __syncthreads();

        float *s_data_dst = s_data_src + height * 8 + radius * 8;

        for (int i = 0; i < min(width - blockIdx.x * 8, (size_t)8); i++)
        {
            //if (z == 0 && threadIdx.x == 0 && threadIdx.y == 0) {printf("hi %lu\n", width - blockIdx.x * 8);}

            float *src = s_data_src + i * (height + radius);
            for (int j = n_iter; j--;)
            {
                unsigned int index = threadIdx.y * blockDim.x + threadIdx.x;
                while (index < height)
                {
                    float tmp = src[index];
                    for (int k = radius; k--;)
                    {
                        tmp += src[index + (k + 1)] + src[index - (k + 1)];
                    }
                    s_data_dst[index] = tmp / (2 * radius + 1);
                    index += blockDim.x * blockDim.y;
                }
                float *tmpPtr = src;
                src = s_data_dst;
                s_data_dst = tmpPtr;
                __syncthreads();
            }

            // if (IS_ODD (n_iter))
            // {
            //     unsigned int index = threadIdx.y * blockDim.x + threadIdx.x;
            //     while (index < height)
            //     {
            //         s_data_dst[index] = src[index];
            //         index += blockDim.x * blockDim.y;
            //     }
            //     float *tmpPtr = src;
            //     src = s_data_dst;
            //     s_data_dst = tmpPtr;
            //     __syncthreads();
            // }
        }

        // if (true)
        // {
        //     while (x < width && y < height)
        //     {
        //         data[x + y * width + z * width * height] = *(s_data_src + localX * (height + radius) + y);
        //         y += yPage;
        //     }
        //     y = y0;
        // }

        if (IS_EVEN(n_iter))
        {
            while (x < width && y < height)
            {
                data[x + y * width + z * width * height] = *(s_data_src + localX * (height + radius) + y);
                y += yPage;
            }
            y = y0;
        }
        else
        {
            while (x < width && y < height)
            {
                data[x + y * width + z * width * height] = *(s_data_src + ((localX - 1 + 9) % 9) * (height + radius) + y);
                y += yPage;
            }
            y = y0;
        }
        z += gridDim.z;
        __syncthreads();
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
        //size_t index = x + y * width + z * width * height;
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

__global__ void g_filter_boxcar_Z_flt_new(float *data, const size_t width, const size_t height, const size_t depth, const size_t radius)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    const int filter_size = (2 * radius + 1);
    float value = 0.0f;
    const float invers = 1.0f / filter_size;
    extern __shared__ float s_data_BZ_flt_new[];
    float *s_data_start = s_data_BZ_flt_new + (threadIdx.x + threadIdx.y * blockDim.x) * filter_size;
    int ptr = 0;    

    float in0;
    float in1;
    float in2;
    float in3;
    float in4;
    float in5;
    float in6;
    float in7;

    float out0;
    float out1;
    float out2;
    float out3;
    float out4;
    float out5;
    float out6;
    float out7;

    float locvar;

    int preFetCnt = -1;

    if (x < width && y < height)
    {
        for (int z = depth; z--;)
        {
            preFetCnt = ++preFetCnt % 8;
            if (preFetCnt == 0)
            {
                if ( z > 6)
                {
                    in0 = data[x + y * width + z * width * height];
                    in1 = data[x + y * width + (z - 1) * width * height];
                    in2 = data[x + y * width + (z - 2) * width * height];
                    in3 = data[x + y * width + (z - 3) * width * height];
                    in4 = data[x + y * width + (z - 4) * width * height];
                    in5 = data[x + y * width + (z - 5) * width * height];
                    in6 = data[x + y * width + (z - 6) * width * height];
                    in7 = data[x + y * width + (z - 7) * width * height];
                }
                else if ( z > 5)
                {
                    in0 = data[x + y * width + z * width * height];
                    in1 = data[x + y * width + (z - 1) * width * height];
                    in2 = data[x + y * width + (z - 2) * width * height];
                    in3 = data[x + y * width + (z - 3) * width * height];
                    in4 = data[x + y * width + (z - 4) * width * height];
                    in5 = data[x + y * width + (z - 5) * width * height];
                    in6 = data[x + y * width + (z - 6) * width * height];
                }
                else if ( z > 4)
                {
                    in0 = data[x + y * width + z * width * height];
                    in1 = data[x + y * width + (z - 1) * width * height];
                    in2 = data[x + y * width + (z - 2) * width * height];
                    in3 = data[x + y * width + (z - 3) * width * height];
                    in4 = data[x + y * width + (z - 4) * width * height];
                    in5 = data[x + y * width + (z - 5) * width * height];
                }
                else if ( z > 3)
                {
                    in0 = data[x + y * width + z * width * height];
                    in1 = data[x + y * width + (z - 1) * width * height];
                    in2 = data[x + y * width + (z - 2) * width * height];
                    in3 = data[x + y * width + (z - 3) * width * height];
                    in4 = data[x + y * width + (z - 4) * width * height];
                }
                else if ( z > 2)
                {
                    in0 = data[x + y * width + z * width * height];
                    in1 = data[x + y * width + (z - 1) * width * height];
                    in2 = data[x + y * width + (z - 2) * width * height];
                    in3 = data[x + y * width + (z - 3) * width * height];
                }
                else if (z > 1)
                {
                    in0 = data[x + y * width + z * width * height];
                    in1 = data[x + y * width + (z - 1) * width * height];
                    in2 = data[x + y * width + (z - 2) * width * height];
                }
                else if (z > 0)
                {
                    in0 = data[x + y * width + z * width * height];
                    in1 = data[x + y * width + (z - 1) * width * height];
                }
                else
                {
                    in0 = data[x + y * width + z * width * height];
                }
                
            }

            if (z < depth - filter_size)
            {
                //value -= data[x + y * width + (z + filter_size) * width * height];
                value -= s_data_start[ptr];
            }

            switch (preFetCnt) 
            {
                case 0: locvar = in0; break;
                case 1: locvar = in1; break;
                case 2: locvar = in2; break;
                case 3: locvar = in3; break;
                case 4: locvar = in4; break;
                case 5: locvar = in5; break;
                case 6: locvar = in6; break;
                case 7: locvar = in7; break;
            }

            value += s_data_start[ptr++] = locvar;
            ptr = ptr % filter_size;

            switch (preFetCnt) 
            {
                case 0: out0 = value * invers; break;
                case 1: out1 = value * invers; break;
                case 2: out2 = value * invers; break;
                case 3: out3 = value * invers; break;
                case 4: out4 = value * invers; break;
                case 5: out5 = value * invers; break;
                case 6: out6 = value * invers; break;
                case 7: out7 = value * invers; break;
            }

            if (preFetCnt == 7)
            {
                if (z < depth - radius - 7)
                {
                    data[x + y * width + (z + radius) * width * height] = out7;
                    data[x + y * width + (z + 1 + radius) * width * height] = out6;
                    data[x + y * width + (z + 2 + radius) * width * height] = out5;
                    data[x + y * width + (z + 3 + radius) * width * height] = out4;
                    data[x + y * width + (z + 4 + radius) * width * height] = out3;
                    data[x + y * width + (z + 5 + radius) * width * height] = out2;
                    data[x + y * width + (z + 6 + radius) * width * height] = out1;
                    data[x + y * width + (z + 7 + radius) * width * height] = out0;
                }
                else if (z < depth - radius - 6)
                {
                    data[x + y * width + (z + radius) * width * height] = out7;
                    data[x + y * width + (z + 1 + radius) * width * height] = out6;
                    data[x + y * width + (z + 2 + radius) * width * height] = out5;
                    data[x + y * width + (z + 3 + radius) * width * height] = out4;
                    data[x + y * width + (z + 4 + radius) * width * height] = out3;
                    data[x + y * width + (z + 5 + radius) * width * height] = out2;
                    data[x + y * width + (z + 6 + radius) * width * height] = out1;
                }
                else if (z < depth - radius - 5)
                {
                    data[x + y * width + (z + radius) * width * height] = out7;
                    data[x + y * width + (z + 1 + radius) * width * height] = out6;
                    data[x + y * width + (z + 2 + radius) * width * height] = out5;
                    data[x + y * width + (z + 3 + radius) * width * height] = out4;
                    data[x + y * width + (z + 4 + radius) * width * height] = out3;
                    data[x + y * width + (z + 5 + radius) * width * height] = out2;
                }
                else if (z < depth - radius - 4)
                {
                    data[x + y * width + (z + radius) * width * height] = out7;
                    data[x + y * width + (z + 1 + radius) * width * height] = out6;
                    data[x + y * width + (z + 2 + radius) * width * height] = out5;
                    data[x + y * width + (z + 3 + radius) * width * height] = out4;
                    data[x + y * width + (z + 4 + radius) * width * height] = out3;
                }
                else if (z < depth - radius - 3)
                {
                    data[x + y * width + (z + radius) * width * height] = out7;
                    data[x + y * width + (z + 1 + radius) * width * height] = out6;
                    data[x + y * width + (z + 2 + radius) * width * height] = out5;
                    data[x + y * width + (z + 3 + radius) * width * height] = out4;
                }
                else if (z < depth - radius - 2)
                {
                    data[x + y * width + (z + radius) * width * height] = out7;
                    data[x + y * width + (z + 1 + radius) * width * height] = out6;
                    data[x + y * width + (z + 2 + radius) * width * height] = out5;
                }
                else if (z < depth - radius - 1)
                {
                    data[x + y * width + (z + radius) * width * height] = out7;
                    data[x + y * width + (z + 1 + radius) * width * height] = out6;
                }
                else if (z < depth - radius)
                {
                    data[x + y * width + (z + radius) * width * height] = out7;
                }
                
            }

            // if (z < depth - radius)
            // {
            //     data[x + y * width + (z + radius) * width * height] = value * invers;
            // }
        }

        if (preFetCnt == 6)
        {
            data[x + y * width + (radius) * width * height] = out6;
            data[x + y * width + (1 + radius) * width * height] = out5;
            data[x + y * width + (2 + radius) * width * height] = out4;
            data[x + y * width + (3 + radius) * width * height] = out3;
            data[x + y * width + (4 + radius) * width * height] = out2;
            data[x + y * width + (5 + radius) * width * height] = out1;
            data[x + y * width + (6 + radius) * width * height] = out0;
        }
        else if (preFetCnt == 5)
        {
            data[x + y * width + (radius) * width * height] = out5;
            data[x + y * width + (1 + radius) * width * height] = out4;
            data[x + y * width + (2 + radius) * width * height] = out3;
            data[x + y * width + (3 + radius) * width * height] = out2;
            data[x + y * width + (4 + radius) * width * height] = out1;
            data[x + y * width + (5 + radius) * width * height] = out0;
        }
        else if (preFetCnt == 4)
        {
            data[x + y * width + (radius) * width * height] = out4;
            data[x + y * width + (1 + radius) * width * height] = out3;
            data[x + y * width + (2 + radius) * width * height] = out2;
            data[x + y * width + (3 + radius) * width * height] = out1;
            data[x + y * width + (4 + radius) * width * height] = out0;
        }
        else if (preFetCnt == 3)
        {
            data[x + y * width + (radius) * width * height] = out3;
            data[x + y * width + (1 + radius) * width * height] = out2;
            data[x + y * width + (2 + radius) * width * height] = out1;
            data[x + y * width + (3 + radius) * width * height] = out0;
        }
        else if (preFetCnt == 2)
        {
            data[x + y * width + (radius) * width * height] = out2;
            data[x + y * width + (1 + radius) * width * height] = out1;
            data[x + y * width + (2 + radius) * width * height] = out0;
        }
        else if (preFetCnt == 1)
        {
            data[x + y * width + (radius) * width * height] = out1;
            data[x + y * width + (1 + radius) * width * height] = out0;
        }
        else if (preFetCnt == 0)
        {
            data[x + y * width + (radius) * width * height] = out0;
        }

        for (int z = radius; z--;)
        {
            //value -= data[x + y * width + (z + radius + 1) * width * height];
            value -= s_data_start[ptr++];
            ptr = ptr % filter_size;
            data[x + y * width + z * width * height] = value * invers;
        }
    }
}

__global__ void g_filter_XYZ_flt(float *data, const size_t width, const size_t height, const size_t depth, const size_t copy_depth, const size_t radius_g, const size_t radius_b, const size_t n_iter)
{
    size_t x = blockIdx.x * gridDim.x + threadIdx.x;
    size_t y = blockIdx.y * gridDim.y + threadIdx.y;
    size_t z = 0;
    size_t local_index = threadIdx.y * blockDim.x + threadIdx.x;
    size_t index = y * width + x;
    size_t warpID = local_index / 32;

    extern __shared__ float s_data_XYZ_flt[];
    float *s_data_src = s_data_XYZ_flt;
    float *s_data_dst = s_data_src + blockDim.x * blockDim.y * copy_depth;

    if (x < width && y < height) 
    {
        for (int i = copy_depth; i--;){ s_data_dst[local_index + i * blockDim.x * blockDim.y] = s_data_src[local_index + i * blockDim.x * blockDim.y] = data[index + i * width * height];}
    }

    x = blockIdx.x * gridDim.x + threadIdx.y;
    y = blockIdx.y * gridDim.y + threadIdx.x;
    local_index = threadIdx.x * blockDim.x + threadIdx.y;

    // Handle X direction
    while(z < copy_depth)
    {
        for (int k = n_iter; k--;)
        {
            if (warpID < radius_g)
            {
                for (int i = warpID + 1; --i;){ s_data_dst[local_index + z * blockDim.x * blockDim.y] += s_data_src[local_index + z * blockDim.x * blockDim.y - i];}    // left side of value
                for (int i = radius_g + 1; --i;) {s_data_dst[local_index + z * blockDim.x * blockDim.y] += s_data_src[local_index + z * blockDim.x * blockDim.y + i];}  // right side of value
            }
            else if (warpID > blockDim.x - radius_g - 1)
            {
                for (int i = blockDim.x - warpID; --i;){ s_data_dst[local_index + z * blockDim.x * blockDim.y] += s_data_src[local_index + z * blockDim.x * blockDim.y + i];}   // right side of value
                for (int i = radius_g + 1; --i;) {s_data_dst[local_index + z * blockDim.x * blockDim.y] += s_data_src[local_index + z * blockDim.x * blockDim.y - i];}          // left side of value
            }
            else 
            {
                for (int i = radius_g + 1; --i;) {s_data_dst[local_index + z * blockDim.x * blockDim.y] += s_data_src[local_index + z * blockDim.x * blockDim.y + i] + s_data_src[local_index + z * blockDim.x * blockDim.y - i];}
            }
            s_data_dst[local_index + z * blockDim.x * blockDim.y] /= 2 * radius_g + 1;
        }
    }

    // Handle X direction
    for (int i = n_iter; i--;)
    {

    }


    // Handle Y Direction

    // Handle Z Direction
    
}

__global__ void g_Mask8(float *data_box, char *maskData8, const size_t width, const size_t height, const size_t depth, const double threshold, float *rms_smooth, const int8_t value)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {return;}

    size_t index = x + y * width;

    while (index < width * height * depth)
    {
        //if (fabs(data_box[index]) > threshold * (*rms_smooth)) {maskData8[index] = (char)1;}
        maskData8[index] |= !(__float_as_int(fabs(data_box[index]) - threshold * (*rms_smooth)) >> 31);
        index += width * height;
    }
}

__global__ void g_Mask1(float *data_box, char *maskData1, const size_t width, const size_t height, const size_t depth, const double threshold, float *rms_smooth)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t x1 = x / 8;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    u_int16_t z = 0;

    const size_t index = x + y * width;
    const size_t index1 = x1 + y * ((width + 7) / 8);

    uint8_t result = 0;

    if (x < width && y < height)
    {
        while (z < depth)
        {

            //result = (!(__float_as_int(fabs(data_box[index + z * width * height]) - threshold * (*rms_smooth)) >> 31)) << (7 - (threadIdx.x % 8));
            result = (fabs(data_box[index + z * width * height]) > threshold * (*rms_smooth)) << (7 - (threadIdx.x % 8));

            //if (IS_NAN(data_box[index + z * width * height])) printf("NaN result: %i\n", result);

            //__syncwarp();

            for (int offset = 1; offset < 8; offset *= 2)
            {
                result |= __shfl_down_sync(0xffffffff, result, offset);
            }

            //__syncwarp();

            if (threadIdx.x % 8 == 0) {maskData1[index1 + z * ((width + 7) / 8) * height] |= result;}
            result = 0;
            //index += width * height;
            //index1 += ((width + 7) / 8) * height;
            z += blockDim.z;
        }
    }
}

__global__ void g_find_max_min(const float *data_box, const size_t size, const size_t cadence, float *min_out, float *max_out)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t num_threads = gridDim.x * blockDim.x;
    const float *ptr = data_box + size + index * cadence;

    float local_min = INFINITY;
    float local_max = -INFINITY;

    while (ptr > data_box + num_threads * cadence)
    {
        ptr -= num_threads * cadence;
        float value = *ptr;
        local_min = min(value, local_min);
        local_max = max(value, local_max);
    }

    for (int offset = 32; offset /= 2;)
    {
        local_min = min(local_min, __shfl_down_sync(0xffffffff, local_min, offset));
        local_max = max(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    if (threadIdx.x % 32 == 0)
    {
        atomicMin(min_out, local_min);
        atomicMax(max_out, local_max);
    }
}

__global__ void g_FlagSources(float *mask32, uint32_t *bbCounter, uint32_t *bbPtr, const size_t width, const size_t height, const size_t depth, const size_t radius_z)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = 0;

    if (x >= width ||  y >= height) { return; }

    size_t index = x + y * width;
    size_t maxIndex = width * height * depth;
    float data;
    uint32_t startIndex;
    uint32_t endIndex;
    uint32_t emptyCounter = 0;
    uint32_t lable;
    bool inSource = false;

    while (index < maxIndex)
    {
        data = mask32[index];

        if (data < 0 && !inSource)
        {
            lable = atomicAdd(bbCounter, 1) + 1;
            mask32[index] = lable;
            startIndex = index;
            endIndex = index;
            emptyCounter = 0;
            inSource = true;
        }
        else if (data < 0)
        {
            mask32[index] = lable;
            endIndex = index;
            emptyCounter = 0;
        }
        else if (++emptyCounter >= radius_z && inSource)
        {
            bbPtr[(lable - 1) * 3] = startIndex;
            bbPtr[(lable - 1) * 3 + 1] = endIndex;
            bbPtr[(lable - 1) * 3 + 2] = (u_int32_t)0;
            inSource = false;
            emptyCounter = 0;
        }
        else
        {
            ++emptyCounter;
        }
        index += width * height;
        __syncthreads();
    }

    if (inSource)
    {
        bbPtr[(lable - 1) * 3] = startIndex;
        bbPtr[(lable - 1) * 3 + 1] = endIndex;
        bbPtr[(lable - 1) * 3 + 2] = (u_int32_t)0;
        inSource = false;
        emptyCounter = 0;
    }
}

__global__ void g_MergeSourcesFirstPass(float *mask32, uint32_t *bbPtr, const size_t width, const size_t height, const size_t depth, const size_t radius_x, const size_t radius_y, const size_t radius_z)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = 0;

	const size_t radius_x_squ   = radius_x * radius_x;
	const size_t radius_y_squ   = radius_y * radius_y;
	const size_t radius_z_squ   = radius_z * radius_z;
	const size_t radius_xy_squ  = radius_x_squ * radius_y_squ;
	const size_t radius_xz_squ  = radius_x_squ * radius_z_squ;
	const size_t radius_yz_squ  = radius_y_squ * radius_z_squ;
	const size_t radius_xyz_squ = radius_x_squ * radius_yz_squ;

    if (x >= width ||  y >= height) { return; }

    size_t index;

    uint32_t lable;
    uint32_t mergeLable;

    while (z < depth)
    {
        index = x + y * width + z * width * height;
        lable = mask32[index];

        if (lable > 0)
        {
            mergeLable = bbPtr[(lable - 1) * 3 + 2] > 0 ? bbPtr[(lable - 1) * 3 + 2] : lable;
            const size_t x1 = (x > radius_x) ? (x - radius_x) : 0;
            const size_t y1 = (y > radius_y) ? (y - radius_y) : 0;
            const size_t z1 = (z > radius_z) ? (z - radius_z) : 0;
            const size_t x2 = (x + radius_x + 1 < width) ? (x + radius_x) : (width - 1);
            const size_t y2 = (y + radius_y + 1 < height) ? (y + radius_y) : (height - 1);
            const size_t z2 = (z + radius_z + 1 < depth) ? (z + radius_z) : (depth - 1);

            // Loop over entire bounding box
            for(size_t zz = z1; zz <= z2; ++zz)
            {
                const size_t dz_squ = zz > z ? (zz - z) * (zz - z) * radius_xy_squ : (z - zz) * (z - zz) * radius_xy_squ;
                
                for(size_t yy = y1; yy <= y2; ++yy)
                {
                    const size_t dy_squ = yy > y ? (yy - y) * (yy - y) * radius_xz_squ : (y - yy) * (y - yy) * radius_xz_squ;
                    
                    for(size_t xx = x1; xx <= x2; ++xx)
                    {
                        const size_t dx_squ = xx > x ? (xx - x) * (xx - x) * radius_yz_squ : (x - xx) * (x - xx) * radius_yz_squ;
                        
                        // Check merging radius, assuming ellipsoid (with dx^2 / rx^2 + dy^2 / ry^2 + dz^2 / rz^2 = 1)
                        if(dx_squ + dy_squ + dz_squ > radius_xyz_squ) continue;

                        uint32_t otherLable = mask32[xx + yy * width + zz * width * height];

                        if (otherLable > 0 && otherLable < mergeLable)
                        {
                            mergeLable = otherLable;
                        }
                    }
                }
            }

            bbPtr[(lable - 1) * 3 + 2] = mergeLable;
            //__syncthreads();
            uint32_t oldMergeLable = mergeLable;

            for(size_t zz = z1; zz <= z2; ++zz)
            {
                const size_t dz_squ = zz > z ? (zz - z) * (zz - z) * radius_xy_squ : (z - zz) * (z - zz) * radius_xy_squ;
                
                for(size_t yy = y1; yy <= y2; ++yy)
                {
                    const size_t dy_squ = yy > y ? (yy - y) * (yy - y) * radius_xz_squ : (y - yy) * (y - yy) * radius_xz_squ;
                    
                    for(size_t xx = x1; xx <= x2; ++xx)
                    {
                        const size_t dx_squ = xx > x ? (xx - x) * (xx - x) * radius_yz_squ : (x - xx) * (x - xx) * radius_yz_squ;
                        
                        // Check merging radius, assuming ellipsoid (with dx^2 / rx^2 + dy^2 / ry^2 + dz^2 / rz^2 = 1)
                        if(dx_squ + dy_squ + dz_squ > radius_xyz_squ) continue;

                        uint32_t otherLable = mask32[xx + yy * width + zz * width * height];
                        uint32_t otherMergeLable = otherLable ? bbPtr[(otherLable - 1) * 3 + 2] : 0;

                        if (otherLable > 0 && otherMergeLable < mergeLable)
                        {
                            mergeLable = otherMergeLable;
                        }
                    }
                }
            }

            if (mergeLable < oldMergeLable)
            {
                bbPtr[(lable - 1) * 3 + 2] = mergeLable;
                atomicMin(bbPtr + ((oldMergeLable - 1) * 3 + 2), mergeLable);
            }

            //__syncthreads();
        }
        z++;
        __syncthreads();
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

__global__ void g_DataCube_copy_mask_1_to_8(char *mask8, char *mask1, size_t width, size_t height, size_t depth)
{
    size_t x1 = blockIdx.x * blockDim.x + threadIdx.x;
    size_t x8 = x1 * 8;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = 0;

    size_t index1 = x1 + y * ((width + 7) / 8);
    size_t index8 = x8 + y * width;

    if (x8 < width && y < height)
    {
        while (z < depth)
        {
            if (x8 < width - 7)
            {
                mask8[index8 + 0] = (mask1[index1] & (1 << 7)) >> 7;
                mask8[index8 + 1] = (mask1[index1] & (1 << 6)) >> 6;
                mask8[index8 + 2] = (mask1[index1] & (1 << 5)) >> 5;
                mask8[index8 + 3] = (mask1[index1] & (1 << 4)) >> 4;
                mask8[index8 + 4] = (mask1[index1] & (1 << 3)) >> 3;
                mask8[index8 + 5] = (mask1[index1] & (1 << 2)) >> 2;
                mask8[index8 + 6] = (mask1[index1] & (1 << 1)) >> 1;
                mask8[index8 + 7] = (mask1[index1] & (1 << 0)) >> 0;
            }
            else
            {
                for (int i = 0; x8 + i < width; i++)
                {
                    mask8[index8 + i] = (mask1[index1] & (1 << (7 - i))) >> (7 - i);
                }
            }

            index1 += ((width + 7) / 8) * height;
            index8 += width * height;
            z += 1;
        }
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

__global__ void g_std_dev_val_flt(float *data, float *data_dst_duo, const size_t size, const float value, const size_t cadence, const int range, const int scaleNoise = 0)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_count = blockDim.x * gridDim.x;

    if (scaleNoise)
    {
        index = threadIdx.x;
        thread_count = blockDim.x;
        data += blockIdx.x * size;
    }

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
            //if (*(s_data_sdf_start + counter) < 0.0f) {atomicAdd(data_dst_duo + 2, 1);}
            *s_data_sdf_start += *(s_data_sdf_start + counter);
            *(s_data_sdf_start + 1) += *(s_data_sdf_start + 1 + counter);
        }
        counter *= 2;
        __syncthreads();
    }

    if (scaleNoise)
    {
        float rms = sqrt(*s_data_sdf / *(s_data_sdf + 1));
        float *ptr = data + size + index * cadence;
        while (ptr > ptr2)
        {
            ptr -= cadence * thread_count;
            *ptr /= rms;
        }
        return;
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(data_dst_duo, (*s_data_sdf_start));
        atomicAdd(data_dst_duo + 1, (*(s_data_sdf_start + 1)));
    }
}

__global__ void g_std_dev_val_flt_final_step(float *data_duo)
{
    *data_duo = sqrt(*data_duo / *(data_duo + 1));
}

__global__ void g_mad_val_flt(float *data, float *data_dst_arr, unsigned int *counter, const size_t size, const size_t max_size, const float value, const size_t cadence, const int range)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_count = blockDim.x * gridDim.x;

    extern __shared__ float s_data_madf_counter[];
    float *s_data_madf_start = s_data_madf_counter + 1;
    *s_data_madf_counter = 0.0f;

    __shared__ unsigned int count;
    count = 0;

    const float *ptr = data + size + index * cadence;
    const float *ptr2 = data + cadence * thread_count - 1;

    bool test = true;
    __shared__ bool globalTest;
    int offset = -1;

    globalTest = true;

    while (globalTest)
    {
        ptr -= cadence * thread_count;

        if (ptr < data) {test = false;}

        if(test && ((range == 0 && IS_NOT_NAN(*ptr)) || (range < 0 && *ptr < 0.0) || (range > 0 && *ptr > 0.0)))
		{
            offset = atomicAdd(&count, 1);
			s_data_madf_start[offset] = fabs(*ptr - value);
		}

        __syncthreads();

        if (threadIdx.x == 0)
        {
            count = atomicAdd(counter, count);
        }

        __syncthreads();

        if (offset >= 0 && count + offset < max_size)
        {
            data_dst_arr[count + offset] = s_data_madf_start[offset];
        }
        offset = -1;

        if (threadIdx.x == blockDim.x - 1)
        {
            count = 0.0f;
            globalTest = test;
        }

        __syncthreads();
    }
}

__global__ void g_mad_val_flt_final_step(float *data, unsigned int *sizePtr)
{
    // unsigned int size = *sizePtr;
    // const size_t index = threadIdx.x;
    // size_t leftIdx;
    // size_t rightIdx;
    // __shared__ bool cnt;
    // cnt = true;

    // float *left;
    // float *right;
    // bool even = true;

    // while (cnt)
    // {
    //     if (even)
    //     {
    //         left = data + index * 2;
    //         right = left + 1;

    //         leftIdx = index * 2;
    //         rightIdx = leftIdx + 1;

    //         while (rightIdx < size)
    //         {
    //             if (data[leftIdx] > data[rightIdx])
    //             {
    //                 float tmp = data[leftIdx];
    //                 data[leftIdx] = data[rightIdx];
    //                 data[rightIdx] = tmp;
    //                 cnt = true;
    //             }
    //             left += blockDim.x * 2;
    //             right = left + 1;

    //             leftIdx += blockDim.x * 2;
    //             rightIdx = leftIdx + 1;
    //         }
    //         even = false;
    //     }
    //     else
    //     {
    //         cnt = false;
    //         left = data + 1 + index * 2;
    //         right = left + 1;

    //         leftIdx = 1 + index * 2;
    //         rightIdx = leftIdx + 1;

    //         while (rightIdx < size)
    //         {
    //             if (data[leftIdx] > data[rightIdx])
    //             {
    //                 float tmp = data[leftIdx];
    //                 data[leftIdx] = data[rightIdx];
    //                 data[rightIdx] = tmp;
    //                 cnt = true;
    //             }
    //             left += blockDim.x * 2;
    //             right = left + 1;

    //             leftIdx += blockDim.x * 2;
    //             rightIdx = leftIdx + 1;
    //         }
    //         even = true;
    //     }
    //     __syncthreads();
    // }

    // if (threadIdx.x == 0)
    // {
    //     *data = *(data + size / 2);
    // }

    __shared__ unsigned int size;
    __shared__ unsigned int smaller_than_pivot;
    
    size = *sizePtr;
    smaller_than_pivot = 0;
    unsigned int index_to_select = size / 2;
    unsigned int num_threads = blockDim.x;
    int overflow = size % num_threads;
        
    size_t index = threadIdx.x;
    __shared__ bool test;
    test = false;

    float *l = data + index;
	float *m = index < overflow ? data + size - overflow + index : data + size - overflow - num_threads + index;
	__shared__ float *ptr;
    ptr = data + size / 2;
    float *i;
    float *j;
	
    while(true)
    {
        //test = false;
        if(l < m)
        {
            float value = *ptr;
            i = l;
            j = m;
            
            do
            {
                while(*i < value && i < j) i+=num_threads;
                while(value < *j && j > i) j-=num_threads;

                if (*i == *j) {i+=num_threads;}
                
                else if(i < j)
                {
                    //printf("swap: %f and %f\n", *i, *j);
                    float tmp = *i;
                    *i = *j;
                    *j = tmp;
                    //i+=blockDim.x;
                    //j-=blockDim.x;
                    //test = true;
                }
            } while(i < j);
            
            //if(j < ptr) atomicSub(&size, ((i - l)/num_threads)); l = i;
            //if(ptr < i) atomicSub(&size, ((m - j)/num_threads)); m = j;

            if ((j - l)/num_threads > 0)
            {
                atomicAdd(&smaller_than_pivot, (j - l)/num_threads);
            }            
        }
        __syncthreads();

        if (index == 0) printf("Smaller: %lu\n", smaller_than_pivot);

        if (smaller_than_pivot == index_to_select) break;

        else if (smaller_than_pivot > index_to_select)
        {
             m = j - 1;
             size = smaller_than_pivot;
        }
        else 
        {
            l = i;
            size -= smaller_than_pivot;
            index_to_select -= smaller_than_pivot;
        }
        if (l < m) ptr = i;
        __syncthreads();
        smaller_than_pivot = 0;
        __syncthreads();
    }

    __syncthreads();	

    if (index == 0)
    {
        *data = *ptr;
    }
}

__global__ void g_mad_val_hist_flt(float *data, const size_t size, unsigned int *bins, unsigned int *total_count, const float value, const size_t cadence, const int range, const unsigned int precision, const float *min_flt, const float *max_flt)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_count = blockDim.x * gridDim.x;

    extern __shared__ unsigned int s_data_madf_hist[];
    
    if (threadIdx.x < precision)
    {
        int i = threadIdx.x;
        while (i < precision)
        {
            s_data_madf_hist[i] = 0;
            i += blockDim.x;
        }
    }

    const float *ptr = data + size + index * cadence;
    const float *ptr2 = data + cadence * thread_count - 1;
    const float my_max = range == 0 ? max(fabs(*min_flt - value), fabs(*max_flt - value)) : (range < 0 ? max(fabs(*min_flt - value), fabs(0 - value)) : max(fabs(*max_flt - value), fabs(0 - value)));
    const float bucket_size = my_max / precision;

    while (ptr > ptr2)
    {
        ptr -= cadence * thread_count;

        if((range == 0 && IS_NOT_NAN(*ptr)) || (range < 0 && *ptr < 0.0) || (range > 0 && *ptr > 0.0))
		{
            atomicAdd(s_data_madf_hist + min((unsigned int)(fabs(*ptr - value) / bucket_size), precision - 1), 1);
		}
    }

    __syncthreads();

    if (threadIdx.x < precision)
    {
        int i = threadIdx.x;
        while (i < precision)
        {
            atomicAdd(bins + i, s_data_madf_hist[i]);
            atomicAdd(total_count, s_data_madf_hist[i]);
            i += blockDim.x;
        }
    }
}

__global__ void g_mad_val_hist_flt_cpy_nth_bin(float *data, const size_t size, float *data_box, unsigned int *bins, unsigned int *total_count, unsigned int *counter, const float value, const size_t cadence, const int range, const unsigned int precision, const float *min_flt, const float *max_flt)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_count = blockDim.x * gridDim.x;
    extern __shared__ float s_data_madf_cpy_bin[];
    __shared__ bool globalTest;
    __shared__ unsigned int count;
    globalTest = true;
    count = 0;

    const float *ptr = data + size + index * cadence;
    const float *ptr2 = data + cadence * thread_count - 1;
    const float my_max = range == 0 ? max(fabs(*min_flt - value), fabs(*max_flt - value)) : (range < 0 ? max(fabs(*min_flt - value), fabs(0 - value)) : max(fabs(*max_flt - value), fabs(0 - value)));
    const float bucket_size = my_max / precision;

    int selected_bin = 0;
    for (int j = *bins; j < *total_count / 2 + 1 && selected_bin < precision - 1; j += bins[++selected_bin]) {}

    bool test = true;
    int offset = -1;

    __syncthreads();

    while (globalTest)
    {
        ptr -= cadence * thread_count;

        if (ptr < data) {test = false;}

        if(test && (unsigned int)(fabs(*ptr - value) / bucket_size) == selected_bin && ((range == 0 && IS_NOT_NAN(*ptr)) || (range < 0 && *ptr < 0.0) || (range > 0 && *ptr > 0.0)))
		{
            offset = atomicAdd(&count, 1);
			s_data_madf_cpy_bin[offset] = fabs(*ptr - value);
		}

        __syncthreads();

        if (threadIdx.x == 0)
        {
            count = atomicAdd(counter, count);
        }

        __syncthreads();

        if (offset >= 0)
        {
            data_box[count + offset] = s_data_madf_cpy_bin[offset];
        }
        offset = -1;

        __syncthreads();

        if (threadIdx.x == blockDim.x - 1)
        {
            count = 0.0f;
            globalTest = test;
        }

        __syncthreads();
    }
}

__global__ void g_mad_val_hist_flt_final_step(float *data, const unsigned int *sizePtr, unsigned int *total_count, unsigned int *bins)
{
    int i = 0;
    int n = *total_count / 2;
    while (n >= 0)
    {
        n -= bins[i++];
    }
    n += bins[--i];

    const unsigned int size = *sizePtr;
    float *l = data;
	float *m = data + size - 1;
	float *ptr = data + n;

    int h = 0;
    int o = 0;
    for (int k = 0; k < size; k++)
    {
        if (data[k] == 0.0) h++;
        else o++;
    }

	while(l < m)
	{
		float value = *ptr;
		float *i = l;
		float *j = m;
		
		do
		{
			while(*i < value) ++i;
			while(value < *j) --j;
			
			if(i <= j)
			{
				float tmp = *i;
				*i = *j;
				*j = tmp;
				++i;
				--j;
			}
		} while(i <= j);
		
		if(j < ptr) l = i;
		if(ptr < i) m = j;
	}
	
	*data = MAD_TO_STD * *ptr;
}

__global__ void g_mad_val_hist_flt_scale_noise(float *data, const size_t size, const float value, const size_t cadence, const int range, const unsigned int precision, const unsigned int max_val_for_eval)
{
    int index = threadIdx.x;
    int thread_count = blockDim.x;

    data += size * blockIdx.x;

    extern __shared__ uint s_data_madf_hist_scaleN[];

    const float *ptr = data + size + index * cadence;
    const float *ptr2 = data + cadence * thread_count - 1;

    uint abs_count;
    uint count_before_n = 0;
    bool first_pass = true;

    __shared__ float min_flt;
    __shared__ float max_flt;
    __shared__ double rms;
    __shared__ uint total_count;
    __shared__ uint offset;

    if (threadIdx.x == 0)
    {
        min_flt = INFINITY;
        max_flt = -INFINITY;
        offset = 0;
    }

    __syncthreads();

    float min_thread = INFINITY;
    float max_thread = -INFINITY;

    while (ptr > ptr2)
    {
        ptr -= cadence * thread_count;

        if (*ptr < min_thread) min_thread = *ptr;
        if (*ptr > max_thread) max_thread = *ptr;
    }

    for (int offs = 16; offs > 0; offs /= 2)
    {
        min_thread = min(min_thread, __shfl_down_sync(0xffffffff, min_thread, offset));
        max_thread = max(max_thread, __shfl_down_sync(0xffffffff, max_thread, offset));
    }

    if (threadIdx.x % 32 == 0)
    {
        atomicMin(&min_flt, min_thread);
        atomicMax(&max_flt, max_thread);
    }
    atomicMin(&min_flt, min_thread);
    atomicMax(&max_flt, max_thread);

    __syncthreads();

    float my_max = range == 0 ? max(fabs(min_flt - value), fabs(max_flt - value)) : (range < 0 ? max(fabs(min_flt - value), fabs(0 - value)) : max(fabs(max_flt - value), fabs(0 - value)));
    //my_max += __FLT_EPSILON__;
    float my_min = 0;
    float bucket_size = my_max / precision;

    uint16_t selected_bin;
    uint count;

    do
    {
        ptr = data + size + index * cadence;
        count = 0;
        if (threadIdx.x == 0)
        {
            memset(s_data_madf_hist_scaleN, 0, precision * sizeof(uint));
            total_count = 0;
        }

        __syncthreads();

        while (ptr > ptr2)
        {
            ptr -= cadence * thread_count;

            float val = fabs(*ptr - value);
            if(((range == 0 && IS_NOT_NAN(*ptr)) || (range < 0 && *ptr < 0.0) || (range > 0 && *ptr > 0.0)) && val >= my_min && val < my_max)
            {
                atomicAdd(s_data_madf_hist_scaleN + min((uint)((val - my_min) / bucket_size), precision - 1), 1);
                ++count;
            }
        }

        for (int offs = 16; offs > 0; offs /= 2)
        {
            count += __shfl_down_sync(0xffffffff, count, offs);
        }

        if (threadIdx.x % 32 == 0)
        {
            atomicAdd(&total_count, count);
        }

        __syncthreads();

        if (threadIdx.x == 0 && first_pass)
        {
            first_pass = false;
            abs_count = total_count;
        }

        selected_bin = 0;
        for (int j = *s_data_madf_hist_scaleN; j < total_count / 2 + 1 && selected_bin < precision - 1; j += s_data_madf_hist_scaleN[++selected_bin])
        {
            if (threadIdx.x == 0)
            {
                count_before_n += s_data_madf_hist_scaleN[selected_bin];
            }
        }

        my_min += bucket_size * selected_bin;
        my_max = my_min + bucket_size;
        //my_max += __FLT_EPSILON__;
        bucket_size = (my_max - my_min) / precision;
    }
    while (s_data_madf_hist_scaleN[selected_bin] > max_val_for_eval);

    float *nth_bin = (float*)(s_data_madf_hist_scaleN + precision);

    ptr = data + size + index * cadence;
    while (ptr > ptr2)
    {
        ptr -= cadence * thread_count;

        float val = fabs(*ptr - value);
        uint hit = 0;
        if(((range == 0 && IS_NOT_NAN(*ptr)) || (range < 0 && *ptr < 0.0) || (range > 0 && *ptr > 0.0)) && val >= my_min && val < my_max)
        {
            hit = 1 << threadIdx.x % 32;
        }

        __syncwarp();

        for (int offs = 16; offs > 0; offs /= 2)
        {
            hit |= __shfl_up_sync(0xffffffff, hit, offs);
        }

        int local_offset;

        if (threadIdx.x % 32 == 31 && hit)
        {
            local_offset = atomicAdd(&offset, __popc(hit));
        }

        __syncwarp();

        local_offset = __shfl_sync(0xffffffff, local_offset, 31);

        if (hit & (1 << threadIdx.x % 32))
        {
            nth_bin[local_offset + __popc(hit) - 1] = val;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {

        int i = 0;
        int n = abs_count / 2;
        n -= count_before_n;

        for (int a = 0; a < s_data_madf_hist_scaleN[selected_bin]; a++) if (blockIdx.x == 0) printf("Abs: %u, Min: %f, Max: %f, value: %.20e\n", abs_count, my_min, my_max, nth_bin[a]);

        const unsigned int size = s_data_madf_hist_scaleN[selected_bin];
        float *l = nth_bin;
        float *m = nth_bin + size - 1;
        float *ptr = nth_bin + n;

        while(l < m)
        {
            float value = *ptr;
            float *i = l;
            float *j = m;
            
            do
            {
                while(*i < value) ++i;
                while(value < *j) --j;
                
                if(i <= j)
                {
                    float tmp = *i;
                    *i = *j;
                    *j = tmp;
                    ++i;
                    --j;
                }
            } while(i <= j);
            
            if(j < ptr) l = i;
            if(ptr < i) m = j;
        }

        rms = MAD_TO_STD * (IS_ODD(abs_count) ? *ptr : (ptr != nth_bin && s_data_madf_hist_scaleN[selected_bin] > 1 ? (*ptr + *(ptr-1)) / 2 : (*ptr + my_min) / 2));

        if (blockIdx.x == 0) printf("rms: %.20e\n", rms);
    }

    __syncthreads();

    float *rms_ptr = data + size + index * cadence;
    while (rms_ptr > ptr2)
    {
        rms_ptr -= cadence * thread_count;

        *rms_ptr /= rms;
    }
}

__global__ void g_create_histogram_flt(const float *data, const size_t size, const int range, unsigned int *histogram, const size_t n_bins, const float *data_min, const float *data_max, const size_t cadence)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_threads = gridDim.x * blockDim.x;
    extern __shared__ unsigned int s_hist_flt[];

    if (threadIdx.x < n_bins) s_hist_flt[threadIdx.x] = 0;

    float local_min = *data_min;
    float local_max = *data_max;

    if(local_max < local_min)
	{
		if (blockIdx.x == 0 && threadIdx.x == 0) printf("Maximum(%f) is not greater than minimum(%f).\n", local_max, local_min);
		return;
	}
	
	// Determine data range
	if(range < 0)
	{
		if(local_min >= 0.0)
		{
			if (blockIdx.x == 0 && threadIdx.x == 0) printf("Minimum is not less than zero.\n");
			return;
		}
		local_max = 0.0;
	}
	else if(range > 0)
	{
		if(local_max <= 0.0)
		{
			if (blockIdx.x == 0 && threadIdx.x == 0) printf("Maximum is not greater than zero.\n");
			return;
		}
		local_min = 0.0;
	}
	else
	{
		const float limit = fabs(local_min) < fabs(local_max) ? fabs(local_min) : fabs(local_max);
		local_min = -limit;
		local_max = limit;
	}

	// Basic setup
	const float slope = (float)(n_bins - 1) / (local_max - local_min);
	const float *ptr  = data + size + cadence * index;
	const float tmp   = 0.5 - slope * local_min;  // The 0.5 is needed for correct rounding later on.

    __syncthreads();
	
	// Generate histogram
	while(ptr > data + cadence * num_threads)
	{
        ptr -= cadence * num_threads;
		if(*ptr >= local_min && *ptr <= local_max)
		{
			size_t bin = (size_t)(slope * *ptr + tmp);
			atomicAdd(s_hist_flt + bin, 1);
		}
	}

    __syncthreads();

    if (threadIdx.x < n_bins) {atomicAdd(histogram + threadIdx.x, s_hist_flt[threadIdx.x]);}
}

__global__ void g_scale_max_min_with_second_moment_flt(const unsigned int *histogram, const size_t n_bins, const int range, float *data_min, float *data_max)
{
    // Calculate second moment
	float mom0 = 0.0;
	float mom1 = 0.0;
	float mom2 = 0.0;

    const float inv_optimal_mom2 = 5.0 / n_bins;  // Require standard deviation to cover 1/5th of the histogram for optimal results
	
    // calculation split between threads of warp
	for(size_t i = threadIdx.x; i < n_bins; i += blockDim.x)
	{
        // if (threadIdx.x == 0) printf("I: %lu Hist: %u\n", i, histogram[i]);
		mom0 += histogram[i];
		mom1 += histogram[i] * (size_t)i;
	}

    for (int offset = 16; offset > 0; offset /= 2)
    {
        mom0 += __shfl_down_sync(0xffffffff, mom0, offset);
        mom1 += __shfl_down_sync(0xffffffff, mom1, offset);
    }

    mom0 = __shfl_sync(0xffffffff, mom0, 0);
    mom1 = __shfl_sync(0xffffffff, mom1, 0);

	mom1 /= mom0;
	
	for(size_t i = threadIdx.x; i < n_bins; i += blockDim.x) 
    {
        mom2 += histogram[i] * (mom1 - i) * (mom1 - i);
    }

    for (int offset = 16; offset > 0; offset /= 2)
    {
        mom2 += __shfl_down_sync(0xffffffff, mom2, offset);
    }

	mom2 = sqrt(mom2 / mom0);
	
    if (threadIdx.x == 0)
    {
        // Ensure that 2nd moment is equal to optimal_mom2 for optimal fitting
        if(range < 0) 
        {
            *data_min *= mom2 * inv_optimal_mom2;
            *data_max = 0.0;
        }
        else if(range > 0) 
        {
            *data_max *= mom2 * inv_optimal_mom2;
            *data_min = 0.0;
        }
        else
        {
            const float limit = fabs(*data_min) < fabs(*data_max) ? fabs(*data_min) : fabs(*data_max);
            *data_min = -limit * mom2 * inv_optimal_mom2;
            *data_max = limit * mom2 * inv_optimal_mom2;
        }
    }
}

// expected to launch with only one warp and one block
__global__ void g_calc_sigma_flt(const unsigned int *histogram, const size_t n_bins, const int range, const float *data_min, const float *data_max, float *sigma_out)
{
    // Fit Gaussian function using linear regression
	// (excluding first and last point in case of edge effects)
	float mean_x = 0.0;
	float mean_y = 0.0;
	size_t counter = 0;
    
    size_t origin = n_bins / 2;
	if(range < 0) origin = n_bins - 1;
	else if(range > 0) origin = 0;
	
	for(size_t i = threadIdx.x + 1; i < n_bins - 1; i += blockDim.x)
	{
		if(histogram[i])
		{
			long int ii = i - origin;
			mean_x += (float)(ii * ii);
			mean_y += log((float)(histogram[i]));
            //printf("Hist at %lu: %u\n", i, histogram[i]);
			++counter;
		}
	}

    for (int offset = 32; offset /= 2;)
    {
        mean_x += __shfl_down_sync(0xffffffff, mean_x, offset);
        mean_y += __shfl_down_sync(0xffffffff, mean_y, offset);
        counter += __shfl_down_sync(0xffffffff, counter, offset);
    }

    mean_x = __shfl_sync(0xffffffff, mean_x, 0);
    mean_y = __shfl_sync(0xffffffff, mean_y, 0);
    counter = __shfl_sync(0xffffffff, counter, 0);
	
	mean_x /= counter;
	mean_y /= counter;

	float upper_sum = 0.0;
	float lower_sum = 0.0;
	
	for(size_t i = threadIdx.x + 1; i < n_bins - 1; i += blockDim.x)
	{
		if(histogram[i])
		{
			long int ii = i - origin;
			const float x = (float)(ii * ii);
			const float y = log((float)(histogram[i]));
			upper_sum += (x - mean_x) * (y - mean_y);
			lower_sum += (x - mean_x) * (x - mean_x);
		}
	}

    for (int offset = 32; offset /= 2;)
    {
        upper_sum += __shfl_down_sync(0xffffffff, upper_sum, offset);
        lower_sum += __shfl_down_sync(0xffffffff, lower_sum, offset);
    }
	
	// Determine standard deviation from slope
    if (threadIdx.x == 0)
    {
        *sigma_out = sqrt(-0.5 * lower_sum / upper_sum) * (*data_max - *data_min) / (n_bins - 1);
    }
}

// expected to launch with only one warp and one block
__global__ void g_gaufit_flt(const float *data, const size_t size, unsigned int *histogram, float *sigma_out, const size_t cadence, const int range, const float *min, const float *max)
{
    if (threadIdx.x == 0) printf("Starting Gaussfit\n");
	// Determine maximum and minimum
	float data_max = *max;
	float data_min = *min;
	
	if(data_min >= 0.0 || data_max <= 0.0)
	{
		if (threadIdx.x == 0) printf("Maximum is not greater than minimum.");
		return;
	}
	
	// Determine data range
	if(range < 0)
	{
		if(data_min >= 0.0)
		{
			if (threadIdx.x == 0) printf("Minimum is not less than zero.");
			return;
		}
		data_max = 0.0;
	}
	else if(range > 0)
	{
		if(data_max <= 0.0)
		{
			if (threadIdx.x == 0) printf("Maximum is not greater than zero.");
			return;
		}
		data_min = 0.0;
	}
	else
	{
		const float limit = fabs(data_min) < fabs(data_max) ? fabs(data_min) : fabs(data_max);
		data_min = -limit;
		data_max = limit;
	}

    if (threadIdx.x == 0) printf("DataMin: %f\nDataMax: %f\n", data_min, data_max);
	
	// Create initial histogram
	const size_t n_bins = 101;
	size_t origin = n_bins / 2;
	if(range < 0) origin = n_bins - 1;
	else if(range > 0) origin = 0;
	const float inv_optimal_mom2 = 5.0 / n_bins;  // Require standard deviation to cover 1/5th of the histogram for optimal results

    if (threadIdx.x == 0) printf("Before hist\n");

    if (threadIdx.x == 0)
    {
        cudaStream_t s;
        cudaEvent_t t;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        //cudaEventCreate(&t);
        g_create_histogram_flt<<<1024, 1024, n_bins * sizeof(unsigned int), s>>>(data, size, range, histogram, n_bins, &data_min, &data_max, cadence);
        cudaEventRecord(t, s);
        cudaStreamWaitEvent(s, t);
    }

    __syncthreads();
	
	// Calculate second moment
	float mom0 = 0.0;
	float mom1 = 0.0;
	float mom2 = 0.0;
	
    // calculation split between threads of warp
	for(size_t i = threadIdx.x; i < n_bins; i += blockDim.x)
	{
        //if (threadIdx.x == 0) printf("I: %lu Mom0: %f   Mom1: %f\n", i, mom0, mom1);
		mom0 += histogram[i];
		mom1 += histogram[i] * (float)i;
	}

    if (threadIdx.x == 0) printf("Mom0: %f\nMom1: %f\n", mom0, mom1);

    __syncwarp();

    for (int offset = 32; offset /= 2;)
    {
        mom0 += __shfl_down_sync(0xffffffff, mom0, offset);
        mom1 += __shfl_down_sync(0xffffffff, mom1, offset);
    }

	mom1 /= mom0;

    if (threadIdx.x == 0) printf("Mom0: %f\nMom1: %f\n", mom0, mom1);
	
	for(size_t i = threadIdx.x; i < n_bins; i += blockDim.x) 
    {
        mom2 += histogram[i] * (mom1 - i) * (mom1 - i);
    }

    __syncwarp();

    for (int offset = 32; offset /= 2;)
    {
        mom2 += __shfl_down_sync(0xffffffff, mom2, offset);
    }

	mom2 = sqrt(mom2 / mom0);
	
	// Ensure that 2nd moment is equal to optimal_mom2 for optimal fitting
	if(range < 0) data_min *= mom2 * inv_optimal_mom2;
	else if(range > 0) data_max *= mom2 * inv_optimal_mom2;
	else
	{
		data_min *= mom2 * inv_optimal_mom2;
		data_max *= mom2 * inv_optimal_mom2;
	}
	
	// Regenerate histogram with new scaling
	if (threadIdx.x == 0)
    {
        cudaStream_t s;
        cudaEvent_t t;
        memset(histogram, 0, n_bins * sizeof(unsigned int));
        //cudaEventCreate(&t);
        g_create_histogram_flt<<<1024, 1024, n_bins * sizeof(unsigned int), s>>>(data, size, range, histogram, n_bins, &data_min, &data_max, cadence);
        cudaEventRecord(t, s);
        cudaStreamWaitEvent(NULL, t);
    }

    __syncthreads();
	
	// Fit Gaussian function using linear regression
	// (excluding first and last point in case of edge effects)
	float mean_x = 0.0;
	float mean_y = 0.0;
	size_t counter = 0;
	
	for(size_t i = threadIdx.x + 1; i < n_bins - 1; i += blockDim.x)
	{
		if(histogram[i])
		{
			long int ii = i - origin;
			mean_x += (float)(ii * ii);
			mean_y += log((float)(histogram[i]));
			++counter;
		}
	}

    if (threadIdx.x == 0) printf("mean_x: %f\nmean_y: %f\n", mean_x, mean_y);

    __syncwarp();

    for (int offset = 32; offset /= 2;)
    {
        mean_x += __shfl_down_sync(0xffffffff, mean_x, offset);
        mean_y += __shfl_down_sync(0xffffffff, mean_y, offset);
        counter += __shfl_down_sync(0xffffffff, counter, offset);
    }

    if (threadIdx.x == 0) printf("mean_x: %f\nmean_y: %f\n", mean_x, mean_y);
	
	mean_x /= counter;
	mean_y /= counter;

	float upper_sum = 0.0;
	float lower_sum = 0.0;
	
	for(size_t i = threadIdx.x + 1; i < n_bins - 1; i += blockDim.x)
	{
		if(histogram[i])
		{
			long int ii = i - origin;
			const float x = (float)(ii * ii);
			const float y = log((float)(histogram[i]));
			upper_sum += (x - mean_x) * (y - mean_y);
			lower_sum += (x - mean_x) * (x - mean_x);
		}
	}

    __syncwarp();

    for (int offset = 32; offset /= 2;)
    {
        upper_sum += __shfl_down_sync(0xffffffff, upper_sum, offset);
        lower_sum += __shfl_down_sync(0xffffffff, lower_sum, offset);
    }

    if (threadIdx.x == 0) printf("upper_sum: %f\nlower_sum: %f\n", upper_sum, lower_sum);
	
	// Determine standard deviation from slope
    if (threadIdx.x == 0)
    {
        const float sigma = sqrt(-0.5 * lower_sum / upper_sum) * (data_max - data_min) / (n_bins - 1);
        if (threadIdx.x == 0) printf("sigma: %.3e\n", sigma);
    }
}

__global__ void g_DataCube_transpose_inplace_flt(float *data, const size_t width, const size_t height, const size_t depth)
{
    const size_t x0 = blockIdx.x * 32 + 8 * (threadIdx.y % 4);
    const size_t y0 = blockIdx.y * 32 + 8 * (threadIdx.y / 4);

    const size_t localX = threadIdx.x % 4;
    const size_t localY = threadIdx.x / 4;

    const size_t xt = x0 + localX;
    const size_t yt = y0 + localY;

    const size_t xb = y0 + localX;
    const size_t yb = x0 + localY;

    size_t z = 0;

    float topVal1;
    float topVal2;

    float botVal1;
    float botVal2;

    if (xt < width && yt < height && xt + 4 < width && xb < width && yb < height && xb + 4 < width)
    {
        if (x0 >= y0)
        {
            while (z < depth)
            {
                topVal1 = data[xt + (4 * (threadIdx.x / 16)) + yt * width + z * width * height];
                topVal2 = data[xt + (4 * (1 - (threadIdx.x / 16))) + yt * width + z * width * height];

                botVal1 = data[xb + (4 * (threadIdx.x / 16)) + yb * width + z * width * height];
                botVal2 = data[xb + (4 * (1 - (threadIdx.x / 16))) + yb * width + z * width * height];

                topVal1 = __shfl_sync(0xffffffff, topVal1, localX * 4 + localY % 4, 16);
                botVal1 = __shfl_sync(0xffffffff, botVal1, localX * 4 + localY % 4, 16);

                topVal2 = __shfl_sync(0xffffffff, topVal2, 16 * (1 - (threadIdx.x / 16)) + localX * 4 + localY % 4);
                botVal2 = __shfl_sync(0xffffffff, botVal2, 16 * (1 - (threadIdx.x / 16)) + localX * 4 + localY % 4);

                data[xt + (4 * (threadIdx.x / 16)) + yt * width + z * width * height] = botVal1;
                data[xt + (4 * (1 - (threadIdx.x / 16))) + yt * width + z * width * height] = botVal2;

                data[xb + (4 * (threadIdx.x / 16)) + yb * width + z * width * height] = topVal1;
                data[xb + (4 * (1 - (threadIdx.x / 16))) + yb * width + z * width * height] = topVal2;

                z+= gridDim.z;
            }
            
        }
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

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
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

__global__ void g_bin_data(float *data, const size_t size, unsigned int *bins, const unsigned int precision, const float min_flt, const float max_flt, const int range, const size_t cadence, unsigned int *total_count, bool flag)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_count = blockDim.x * gridDim.x;

    u_int8_t warpIdx = threadIdx.x / 32;
    u_int8_t laneIdx = threadIdx.x % 32;    
    extern __shared__ unsigned int s_data_bin_data[];
    unsigned int *s_bins = s_data_bin_data + precision * warpIdx; 
    
    if (laneIdx < precision)
    {
        int i = laneIdx;
        while (i < precision)
        {
            s_bins[i] = 0;
            i+=32;
        }
    }

    const float *ptr = data + size + cadence * index;
    const float *ptr2 = data + cadence * thread_count - 1;
    const float my_max = range == 0 ? max(fabs(min_flt), fabs(max_flt)) : (range < 0 ? max(fabs(min_flt), 0.0f) : max(fabs(max_flt), 0.0f));
    const float bucket_size = flag ? (max_flt - min_flt) / precision : my_max / precision;


    while (ptr > ptr2)
    {
        ptr -= cadence * thread_count;

        if((range == 0 && IS_NOT_NAN(*ptr)) || (range < 0 && *ptr < 0.0) || (range > 0 && *ptr > 0.0))
		{
            if (flag && fabs(*ptr) >= min_flt && fabs(*ptr) <= max_flt)
            {
                atomicAdd(s_bins + min((unsigned int)((fabs(*ptr) - min_flt) / bucket_size), precision - 1), 1);
            }
            else if (!flag)
            {
                atomicAdd(s_bins + min((unsigned int)(fabs(*ptr) / bucket_size), precision - 1), 1);
            }
		}
    }

    __syncthreads();

    size_t localCount = 0;
    if (laneIdx < precision)
    {
        int i = laneIdx;
        while (i < precision)
        {
            localCount += s_bins[i];
            i+=32;
        }
    }

    u_int8_t offset = 2;
    while (offset <= blockDim.x / 32)
    {
        if (warpIdx % offset == 0)
        {
            if (laneIdx < precision)
            {
                int i = laneIdx;
                while (i < precision)
                {
                    localCount += s_data_bin_data[i + precision * (warpIdx + offset / 2)];
                    s_bins[i] += s_data_bin_data[i + precision * (warpIdx + offset / 2)];
                    i+=32;
                }
            }
        }
        __syncthreads();
        offset *= 2;
    }

    if (warpIdx == 0)
    {
        for (int offset = 16; offset > 0; offset /= 2)
        {
            localCount += __shfl_down_sync(0xffffffff, localCount, offset);
        }
    }

    if (warpIdx == 0 && laneIdx < precision)
    {
        int i = laneIdx;
        while (i < precision)
        {
            atomicAdd(bins + i, s_bins[i]);
            i+=32;
        }
    }

    if(warpIdx == 0 && laneIdx == 0)
    {
        atomicAdd(total_count, localCount);
    }
}

__global__ void g_cpy_bin(float *data, const size_t size, float *target_arr, unsigned int *arr_ptr, const float min_flt, const float max_flt, const int range, const size_t cadence)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_count = blockDim.x * gridDim.x;

    u_int8_t warpIdx = threadIdx.x / 32;
    u_int8_t laneIdx = threadIdx.x % 32;
    
    bool warpTest = true;
    bool test = true;

    float candidate;
    u_int8_t flag = 0;

    const float *ptr = data + size + cadence * index;
    const float *ptr2 = data + cadence * thread_count - 1;

    while (warpTest)
    {
        flag = 0;
        ptr -= cadence * thread_count;

        if (ptr < data) {test = false;}

        if(test && ((range == 0 && IS_NOT_NAN(*ptr)) || (range < 0 && *ptr < 0.0) || (range > 0 && *ptr > 0.0)))
		{
            if ((fabs(*ptr) >= min_flt && fabs(*ptr) <= max_flt))
            {
			    candidate = fabs(*ptr);
                flag = 1;
            }
            
		}

        __syncwarp();

        int bitmask = __ballot_sync(0xffffffff, flag);

        unsigned int offset = __popc(bitmask << (32 - laneIdx));
        unsigned int g_offset = 0;

        if (laneIdx == 31 && bitmask)
        {
            g_offset = atomicAdd(arr_ptr, offset + flag);
        }

        __syncwarp();

        g_offset = __shfl_sync(0xffffffff, g_offset, 31);

        if (flag) {target_arr[g_offset + offset] = candidate;}

        __syncwarp();

        warpTest = __shfl_sync(0xffffffff, test, 31);
    }
}

__global__ void g_median_final(float *data, const unsigned int *sizePtr, unsigned int n)
{
    int i = 0;

    const unsigned int size = *sizePtr;
    float *l = data;
	float *m = data + size - 1;
	float *ptr = data + n;

    int h = 0;
    int o = 0;
    for (int k = 0; k < size; k++)
    {
        if (data[k] == 0.0) h++;
        else o++;
    }

	while(l < m)
	{
		float value = *ptr;
		float *i = l;
		float *j = m;
		
		do
		{
			while(*i < value) ++i;
			while(value < *j) --j;
			
			if(i <= j)
			{
				float tmp = *i;
				*i = *j;
				*j = tmp;
				++i;
				--j;
			}
		} while(i <= j);
		
		if(j < ptr) l = i;
		if(ptr < i) m = j;
	}
	
	*data = MAD_TO_STD * *ptr;
}