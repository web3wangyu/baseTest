//
// Created by ps on 2022/4/6.
//

#ifndef CUDA_LEARNING_UTILS_H
#define CUDA_LEARNING_UTILS_H
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include <stdexcept>

#define cudaCheckError(expr) {                                                               \
    cudaError e;                                                                             \
    if ((e = expr) != cudaSuccess) {                                                         \
        const char* error_str = cudaGetErrorString(e);                                       \
        std::cout << "Cuda failure:" << error_str;                                           \
        throw std::runtime_error(error_str);                                                    \
    }                                                                                        \
}

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

#endif //CUDA_LEARNING_UTILS_H
