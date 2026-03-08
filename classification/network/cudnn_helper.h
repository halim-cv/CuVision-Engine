#ifndef CUDNN_HELPER_H
#define CUDNN_HELPER_H

#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <cstdlib>

using namespace std;

#define CUDA_CHECK(status)                                                    \
    if (status != cudaSuccess) {                                               \
        cerr << "CUDA Error at line " << __LINE__ << ": "                 \
                  << cudaGetErrorString(status) << endl;                  \
        exit(EXIT_FAILURE);                                               \
    }

#define CUDNN_CHECK(status)                                                   \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
        cerr << "cuDNN Error at line " << __LINE__ << ": "                \
                  << cudnnGetErrorString(status) << endl;                 \
        exit(EXIT_FAILURE);                                               \
    }

#endif // CUDNN_HELPER_H
