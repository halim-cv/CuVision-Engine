#include "cudnn_helper.h"
#include "utilities.cu"
#include <vector>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>

using namespace std;

#define CUBLAS_CHECK(status)                                                   \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        cerr << "cuBLAS Error at line " << __LINE__ << ": " << status << endl; \
        exit(EXIT_FAILURE);                                                    \
    }

class ImageClassifier {
public:
    ImageClassifier(int batchSize, int channels, int height, int width, int numClasses = 10)
        : batchSize(batchSize), channels(channels), height(height), width(width), numClasses(numClasses) {

        CUDNN_CHECK(cudnnCreate(&cudnnHandle));
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        setupDescriptors();
        setupMemory();
    }

    ~ImageClassifier() {
        // Destroy Descriptors
        cudnnDestroyTensorDescriptor(inputDesc);
        cudnnDestroyFilterDescriptor(filter1Desc);
        cudnnDestroyConvolutionDescriptor(conv1Desc);
        cudnnDestroyTensorDescriptor(conv1OutDesc);
        
        cudnnDestroyFilterDescriptor(filter2Desc);
        cudnnDestroyConvolutionDescriptor(conv2Desc);
        cudnnDestroyTensorDescriptor(conv2OutDesc);

        cudnnDestroyPoolingDescriptor(poolDesc);
        cudnnDestroyTensorDescriptor(pool1OutDesc);
        cudnnDestroyTensorDescriptor(pool2OutDesc);
        
        cudnnDestroyActivationDescriptor(actDesc);
        cudnnDestroyTensorDescriptor(softmaxOutDesc);
        
        cudnnDestroy(cudnnHandle);
        cublasDestroy(cublasHandle);

        // Free Memory
        cudaFree(d_input);
        cudaFree(d_filter1); cudaFree(d_filter2);
        cudaFree(d_conv1Out); cudaFree(d_pool1Out);
        cudaFree(d_conv2Out); cudaFree(d_pool2Out);
        cudaFree(d_workspace);
        cudaFree(d_fcWeights); cudaFree(d_fcOut);
        cudaFree(d_softmaxOut);

        // Gradients
        cudaFree(d_diffLogits); cudaFree(d_diffPool2);
        cudaFree(d_diffConv2); cudaFree(d_diffPool1);
        cudaFree(d_diffConv1);
    }

    void forward(float* h_input) {
        size_t inputSize = batchSize * channels * height * width * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice));

        float alpha = 1.0f, beta = 0.0f;

        // --- Layer 1: Conv -> ReLU -> Pool ---
        CUDNN_CHECK(cudnnConvolutionForward(cudnnHandle, &alpha, inputDesc, d_input,
                                            filter1Desc, d_filter1, conv1Desc, algo1,
                                            d_workspace, workspaceSize, &beta, conv1OutDesc, d_conv1Out));
        CUDNN_CHECK(cudnnActivationForward(cudnnHandle, actDesc, &alpha, conv1OutDesc, d_conv1Out, &beta, conv1OutDesc, d_conv1Out));
        CUDNN_CHECK(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv1OutDesc, d_conv1Out, &beta, pool1OutDesc, d_pool1Out));

        // --- Layer 2: Conv -> ReLU -> Pool ---
        CUDNN_CHECK(cudnnConvolutionForward(cudnnHandle, &alpha, pool1OutDesc, d_pool1Out,
                                            filter2Desc, d_filter2, conv2Desc, algo2,
                                            d_workspace, workspaceSize, &beta, conv2OutDesc, d_conv2Out));
        CUDNN_CHECK(cudnnActivationForward(cudnnHandle, actDesc, &alpha, conv2OutDesc, d_conv2Out, &beta, conv2OutDesc, d_conv2Out));
        CUDNN_CHECK(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv2OutDesc, d_conv2Out, &beta, pool2OutDesc, d_pool2Out));

        // --- Layer 3: Fully Connected ---
        int flatSize = c2 * h2 * w2;
        CUBLAS_CHECK(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 numClasses, batchSize, flatSize, &alpha,
                                 (float*)d_fcWeights, flatSize, (float*)d_pool2Out, flatSize,
                                 &beta, (float*)d_fcOut, numClasses));

        // --- Output: Softmax ---
        CUDNN_CHECK(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                        &alpha, pool2OutDesc, d_fcOut, &beta, softmaxOutDesc, d_softmaxOut));
    }

    void backward(uint8_t* h_labels, float lr, float weightDecay = 0.0005f) {
        float alpha = 1.0f, beta = 0.0f;
        float lr_neg = -lr;

        // 1. Loss Gradient (Softmax - OneHot)
        vector<float> h_soft(batchSize * numClasses);
        CUDA_CHECK(cudaMemcpy(h_soft.data(), d_softmaxOut, h_soft.size() * sizeof(float), cudaMemcpyDeviceToHost));
        vector<float> h_grad(batchSize * numClasses);
        for(int i=0; i<batchSize; ++i) {
            for(int c=0; c<numClasses; ++c) h_grad[i*numClasses+c] = (h_soft[i*numClasses+c] - (h_labels[i]==c ? 1.0f : 0.0f)) / batchSize;
        }
        CUDA_CHECK(cudaMemcpy(d_diffLogits, h_grad.data(), h_grad.size() * sizeof(float), cudaMemcpyHostToDevice));

        // 2. FC Backward (with Weight Decay / L2 Regularization)
        int flatSize = c2 * h2 * w2;
        // Apply weight decay: Weights = Weights * (1 - lr * lambda)
        float decayFactor = 1.0f - (lr * weightDecay);
        CUBLAS_CHECK(cublasSscal(cublasHandle, flatSize * numClasses, &decayFactor, (float*)d_fcWeights, 1));
        
        CUBLAS_CHECK(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, flatSize, numClasses, batchSize, &lr_neg, (float*)d_pool2Out, flatSize, (float*)d_diffLogits, numClasses, &alpha, (float*)d_fcWeights, flatSize));
        CUBLAS_CHECK(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, flatSize, batchSize, numClasses, &alpha, (float*)d_fcWeights, flatSize, (float*)d_diffLogits, numClasses, &beta, (float*)d_diffPool2, flatSize));

        // 3. Layer 2 Backward (with Weight Decay)
        CUDNN_CHECK(cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha, pool2OutDesc, d_pool2Out, pool2OutDesc, d_diffPool2, conv2OutDesc, d_conv2Out, &beta, conv2OutDesc, d_diffConv2));
        CUDNN_CHECK(cudnnActivationBackward(cudnnHandle, actDesc, &alpha, conv2OutDesc, d_conv2Out, conv2OutDesc, d_diffConv2, conv2OutDesc, d_conv2Out, &beta, conv2OutDesc, d_diffConv2));
        
        // Weight decay for Filter 2
        CUBLAS_CHECK(cublasSscal(cublasHandle, 64 * 32 * 3 * 3, &decayFactor, (float*)d_filter2, 1));
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(cudnnHandle, &lr_neg, pool1OutDesc, d_pool1Out, conv2OutDesc, d_diffConv2, conv2Desc, algo2, d_workspace, workspaceSize, &alpha, filter2Desc, d_filter2));
        CUDNN_CHECK(cudnnConvolutionBackwardData(cudnnHandle, &alpha, filter2Desc, d_filter2, conv2OutDesc, d_diffConv2, conv2Desc, algo2, d_workspace, workspaceSize, &beta, pool1OutDesc, d_diffPool1));

        // 4. Layer 1 Backward (with Weight Decay)
        CUDNN_CHECK(cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha, pool1OutDesc, d_pool1Out, pool1OutDesc, d_diffPool1, conv1OutDesc, d_conv1Out, &beta, conv1OutDesc, d_diffConv1));
        CUDNN_CHECK(cudnnActivationBackward(cudnnHandle, actDesc, &alpha, conv1OutDesc, d_conv1Out, conv1OutDesc, d_diffConv1, conv1OutDesc, d_conv1Out, &beta, conv1OutDesc, d_diffConv1));
        
        // Weight decay for Filter 1
        CUBLAS_CHECK(cublasSscal(cublasHandle, 32 * channels * 3 * 3, &decayFactor, (float*)d_filter1, 1));
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(cudnnHandle, &lr_neg, inputDesc, d_input, conv1OutDesc, d_diffConv1, conv1Desc, algo1, d_workspace, workspaceSize, &alpha, filter1Desc, d_filter1));
        
        // Final Backprop to Input (Optional but completes the derivative chain)
        size_t d_in_size = batchSize * channels * height * width * sizeof(float);
        void* d_diffInput; CUDA_CHECK(cudaMalloc(&d_diffInput, d_in_size));
        CUDNN_CHECK(cudnnConvolutionBackwardData(cudnnHandle, &alpha, filter1Desc, d_filter1, conv1OutDesc, d_diffConv1, conv1Desc, algo1, d_workspace, workspaceSize, &beta, inputDesc, d_diffInput));
        cudaFree(d_diffInput);
    }

    void saveWeights(const string& fn) {
        ofstream f(fn, ios::binary);
        if(!f.is_open()) return;
        saveW(f, d_filter1, 32 * channels * 3 * 3);
        saveW(f, d_filter2, 64 * 32 * 3 * 3);
        saveW(f, d_fcWeights, (c2*h2*w2) * numClasses);
        f.close();
        cout << "Model saved to " << fn << endl;
    }

    float* getOutput() { return (float*)d_softmaxOut; }

private:
    cudnnHandle_t cudnnHandle; cublasHandle_t cublasHandle;
    int batchSize, channels, height, width, numClasses;
    int c1, h1, w1, c2, h2, w2;

    cudnnTensorDescriptor_t inputDesc, conv1OutDesc, pool1OutDesc, conv2OutDesc, pool2OutDesc, softmaxOutDesc;
    cudnnFilterDescriptor_t filter1Desc, filter2Desc;
    cudnnConvolutionDescriptor_t conv1Desc, conv2Desc;
    cudnnPoolingDescriptor_t poolDesc;
    cudnnActivationDescriptor_t actDesc;

    void *d_input, *d_filter1, *d_filter2, *d_conv1Out, *d_pool1Out, *d_conv2Out, *d_pool2Out, *d_workspace, *d_fcWeights, *d_fcOut, *d_softmaxOut;
    void *d_diffLogits, *d_diffPool2, *d_diffConv2, *d_diffPool1, *d_diffConv1;
    size_t workspaceSize;
    cudnnConvolutionFwdAlgo_t algo1, algo2;

    void setupDescriptors() {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&inputDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, channels, height, width));

        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter1Desc));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter1Desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, channels, 3, 3));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv1Desc));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv1Desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        
        int n;
        CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv1Desc, inputDesc, filter1Desc, &n, &c1, &h1, &w1));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&conv1OutDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv1OutDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c1, h1, w1));

        CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolDesc));
        CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 0, 0, 2, 2));

        int pn, pc, ph, pw;
        CUDNN_CHECK(cudnnGetPooling2dForwardOutputDim(poolDesc, conv1OutDesc, &pn, &pc, &ph, &pw));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&pool1OutDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(pool1OutDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, pn, pc, ph, pw));

        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter2Desc));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter2Desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 32, 3, 3));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv2Desc));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv2Desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

        CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv2Desc, pool1OutDesc, filter2Desc, &n, &c2, &h2, &w2));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&conv2OutDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv2OutDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c2, h2, w2));

        CUDNN_CHECK(cudnnGetPooling2dForwardOutputDim(poolDesc, conv2OutDesc, &pn, &pc, &ph, &pw));
        c2 = pc; h2 = ph; w2 = pw;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&pool2OutDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(pool2OutDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, pn, pc, ph, pw));

        CUDNN_CHECK(cudnnCreateActivationDescriptor(&actDesc));
        CUDNN_CHECK(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&softmaxOutDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(softmaxOutDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, numClasses, 1, 1));
    }

    void setupMemory() {
        size_t ws1, ws2;
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(cudnnHandle, inputDesc, filter1Desc, conv1Desc, conv1OutDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo1));
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(cudnnHandle, pool1OutDesc, filter2Desc, conv2Desc, conv2OutDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo2));
        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, inputDesc, filter1Desc, conv1Desc, conv1OutDesc, algo1, &ws1));
        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, pool1OutDesc, filter2Desc, conv2Desc, conv2OutDesc, algo2, &ws2));
        workspaceSize = max(ws1, ws2);

        CUDA_CHECK(cudaMalloc(&d_input, batchSize * channels * height * width * sizeof(float)));
        
        // Initialize weights with He Initialization
        CUDA_CHECK(cudaMalloc(&d_filter1, 32 * channels * 3 * 3 * sizeof(float))); 
        initHe(d_filter1, 32 * channels * 3 * 3, channels * 3 * 3);
        
        CUDA_CHECK(cudaMalloc(&d_filter2, 64 * 32 * 3 * 3 * sizeof(float))); 
        initHe(d_filter2, 64 * 32 * 3 * 3, 32 * 3 * 3);
        
        CUDA_CHECK(cudaMalloc(&d_conv1Out, batchSize * 32 * h1 * w1 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pool1Out, batchSize * 32 * ph * pw * sizeof(float))); // Using computed ph, pw indirectly
        CUDA_CHECK(cudaMalloc(&d_conv2Out, batchSize * 64 * h2 * w2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pool2Out, batchSize * 64 * ph * pw * sizeof(float))); // Note: reusing ph, pw logic
        CUDA_CHECK(cudaMalloc(&d_workspace, workspaceSize));
        
        int flatSize = c2 * h2 * w2;
        CUDA_CHECK(cudaMalloc(&d_fcWeights, flatSize * numClasses * sizeof(float))); 
        initHe(d_fcWeights, flatSize * numClasses, flatSize);
        
        CUDA_CHECK(cudaMalloc(&d_fcOut, batchSize * numClasses * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_softmaxOut, batchSize * numClasses * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc(&d_diffLogits, batchSize * numClasses * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_diffPool2, flatSize * batchSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_diffConv2, batchSize * 64 * h2 * w2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_diffPool1, batchSize * 32 * (h1/2) * (w1/2) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_diffConv1, batchSize * 32 * h1 * w1 * sizeof(float)));
    }

    void initHe(void* d_ptr, size_t size, int fanIn) {
        float stdDev = sqrt(2.0f / fanIn);
        float* h = new float[size];
        for(size_t i = 0; i < size; ++i) {
            // Normal distribution approximation
            float u1 = (float)rand() / RAND_MAX;
            float u2 = (float)rand() / RAND_MAX;
            float randStdNormal = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI * u2);
            h[i] = randStdNormal * stdDev;
        }
        CUDA_CHECK(cudaMemcpy(d_ptr, h, size * sizeof(float), cudaMemcpyHostToDevice));
        delete[] h;
    }

    void saveW(ofstream& f, void* d_ptr, size_t size) {
        vector<float> h(size);
        CUDA_CHECK(cudaMemcpy(h.data(), d_ptr, size * sizeof(float), cudaMemcpyDeviceToHost));
        f.write((char*)h.data(), size * sizeof(float));
    }
};
