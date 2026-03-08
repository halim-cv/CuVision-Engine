# CuVision-Engine

High-Performance Native Computer Vision for the Edge.

CuVision-Engine is a low-latency Computer Vision framework developed in C++ and CUDA. By leveraging cuDNN and cuBLAS directly, it achieves maximum hardware utilization on NVIDIA GPUs, making it ideal for real-time applications where high-level frameworks like PyTorch or TensorFlow introduce unnecessary overhead.

## Core Capabilities

- **Native CUDA/cuDNN Optimization**: Direct manipulation of memory and descriptors for ultra-fast inference.
- **Edge Deployment Ready**: Designed with a minimal footprint for systems like NVIDIA Jetson Nano/Xavier.
- **Robust Training Pipeline**: Includes state-of-the-art CNN optimizations like Batch Normalization, Dropout, and Momentum SGD.
- **Modular Architecture**: Logical separation between data processing, neural network computation, and hardware utilities.

## Technical Specifications

| Feature | Implementation | Benefit |
| :--- | :--- | :--- |
| **Architecture** | Deep CNN (Conv -> BN -> ReLU -> Pool) | High feature extraction capacity with normalized signals. |
| **Normalization** | Batch Normalization (Spatial) | Mitigates internal covariate shift, speeding up convergence. |
| **Regularization** | Dropout (50%) & L2 Weight Decay | Prevents co-adaptation and overfitting on training data. |
| **Initialization** | He (Kaiming) Normal | Prevents vanishing/exploding gradients in ReLU layers. |
| **Optimization** | Custom Momentum SGD Kernel | Accelerated gradient descent directly on the GPU. |
| **Learning Rate** | Scheduled LR Decay | Smooth convergence and high final precision. |
| **Hardware** | Synchronous CUDA Streams | Guaranteed deterministic behavior for safety-critical tasks. |

## Project Structure

- **classification/**: Deep CNN implementation for category recognition.
  - **network/**: Native CUDA kernels, benchmarking timers, and cuDNN descriptors.
  - **dataset/**: Preprocessing scripts and high-speed binary data loaders.
- **segmentation/**: Future module for pixel-wise classification.
- **detection/**: Future module for real-time object tracking.

## Getting Started

### Hardware Requirements
- NVIDIA GPU (Pascal architecture or newer recommended).
- CUDA Toolkit 11.0+.
- cuDNN 8.x library.

### Build Instructions
Navigate to a module directory and execute the platform-specific build script:

```powershell
cd classification
.\compile.ps1
```

### Execution Pipeline

1. **Prepare Data**:
   Download and format the dataset into optimized binary blobs:
   ```bash
   python dataset/prepare_dataset.py
   ```

2. **Train & Infer**:
   Execute the compiled binary to start the training process:
   ```powershell
   .\dnn_classifier.exe
   ```

## Development Roadmap

- [x] Deep CNN Core with He Initialization.
- [x] Batch Normalization & Dropout Regularization.
- [x] Custom Momentum-Accelerated SGD Kernel.
- [x] L2 Regularization & Scheduled LR Decay.
- [x] High-precision GPU Benchmark Timers.
- [x] Accelerated Data Augmentation (CUDA Kernels).
- [ ] Object Detection (Anchor-based kernels).
- [ ] TensorRT Integration for production deployment.

## Reference Papers

The optimizations implemented in this engine are based on foundational deep learning research:

1. **Batch Normalization**: [*Ioffe and Szegedy, 2015*] "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" ([arXiv:1502.03167](https://arxiv.org/abs/1502.03167))
2. **Dropout Regularization**: [*Srivastava et al., 2014*] "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" ([JMLR](https://jmlr.org/papers/v15/srivastava14a.html))
3. **He (Kaiming) Initialization**: [*He et al., 2015*] "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" ([arXiv:1502.01852](https://arxiv.org/abs/1502.01852))
4. **Momentum SGD**: [*Sutskever et al., 2013*] "On the importance of initialization and momentum in deep learning" ([ICML](http://proceedings.mlr.press/v28/sutskever13.html))

## License
Distributed under the MIT License. See `LICENSE` for more information.
