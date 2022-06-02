#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "my_helper/utils.h"
#include <cuda.h>

__global__ void testKernel(int *out, const int *in, size_t N, int addValue, int cycles) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        volatile int value = in[i];
        for (int j = 0; j < cycles; j++) {
            value += addValue;
        }
        out[i] = value;
    }
}
int main() {

    int size = 1 << 24, batch_size = 1 << 20;
    int batch_num = size / batch_size;
    int blocksize = 512;   // initial block size

    dim3 block (blocksize, 1);
    dim3 grid  ((batch_size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *h_idata;
    CHECK(cudaHostAlloc((void **) &h_idata, bytes, cudaHostAllocDefault));
    int *h_odata;
    CHECK(cudaHostAlloc((void **) &h_odata, bytes, cudaHostAllocDefault));

    // initialize the array
    for (int i = 0; i < size; i++)
    {
        // mask off high 2 bytes to force max number to 255
        h_idata[i] = (int)( rand() & 0xFF );
    }

    int *d_idata = NULL;
    int *d_odata = NULL;
    CHECK(cudaMalloc((void **) &d_idata, bytes));
    CHECK(cudaMalloc((void **) &d_odata, bytes));

    cudaStream_t *streams = (cudaStream_t *)malloc(batch_num * sizeof(cudaStream_t));
    for (int i = 0; i < batch_num; i++) {
        CHECK(cudaStreamCreate(&(streams[i])));
    }

    // set up max connectioin
    const char* iname = "CUDA_DEVICE_MAX_CONNECTIONS";
    setenv (iname, "16", 1);

    // event for time used
    cudaEvent_t start, stop, *syncEvents = (cudaEvent_t *)malloc(2 * sizeof(cudaEvent_t));
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventCreate(&(syncEvents[0])));
    CHECK(cudaEventCreate(&(syncEvents[1])));
    float time_used_label = 0.0, elapsed_time;
    cudaEventRecord(start);
    for (int i = 0; i < batch_num; i++) {
        CHECK(cudaMemcpyAsync(d_idata + i * batch_size, h_idata + i * batch_size, batch_size * sizeof(int), cudaMemcpyHostToDevice, streams[i]));
        testKernel<<<grid, block, 0, streams[i]>>>(d_odata + i * batch_size, d_idata + i * batch_size, batch_size, 5, 50000);
        CHECK(cudaMemcpyAsync(h_odata + i * batch_size, d_odata + i * batch_size, batch_size * sizeof(int), cudaMemcpyDeviceToHost, streams[i]));
    }

//    for (int i = 0; i < batch_num; i++) {
//        testKernel<<<grid, block, 0, streams[i]>>>(d_odata + i * batch_size, d_idata + i * batch_size, batch_size, 5, 1000);
//    }
//
//    for (int i = 0; i < batch_num; i++) {
//        CHECK(cudaMemcpyAsync(h_odata + i * batch_size, d_odata + i * batch_size, batch_size * sizeof(int), cudaMemcpyDeviceToHost, streams[i]));
//    }

    cudaEventRecord(stop);
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    time_used_label += elapsed_time;
    std::cout << "Time cost:" << time_used_label << "ms." << std::endl;

    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    CHECK(cudaFreeHost(h_idata));
    CHECK(cudaFreeHost(h_odata));

    for (int i = 0; i < batch_num; i++) {
        CHECK(cudaStreamDestroy(streams[i]));
    }
    free(streams);

    return 0;


}

