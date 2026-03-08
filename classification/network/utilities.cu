#include "cudnn_helper.h"
#include <iostream>

using namespace std;

// GPU Utility Functions
void printDeviceInformation() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        cout << "No CUDA compatible devices found." << endl;
        return;
    }

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        cout << "Device " << i << ": " << prop.name << endl;
        cout << "  Compute Capability: " << prop.major << "." << prop.minor << endl;
        cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << endl;
    }
}
