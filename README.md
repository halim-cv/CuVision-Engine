# CuVision-Engine

High-Performance Native Computer Vision for the Edge.

CuVision-Engine is a low-latency Computer Vision framework developed in C++ and CUDA. By leveraging cuDNN and cuBLAS directly, it achieves maximum hardware utilization on NVIDIA GPUs, making it ideal for real-time applications where high frameworks like PyTorch or TensorFlow introduce unnecessary overhead.

## Core Capabilities

- **Native CUDA/cuDNN Optimization**: Direct manipulation of memory and descriptors for ultra-fast inference.
- **Edge Deployment Ready**: Designed with a minimal footprint for systems like NVIDIA Jetson Nano/Xavier.
- **Robust Training Pipeline**: Includes advanced features like He Initialization and weight decay for stable learning.
- **Modular Architecture**: Logical separation between data processing, neural network logic, and hardware utilities.

## Technical Specifications

| Feature | Implementation | Benefit |
| :--- | :--- | :--- |
| **Architecture** | 3-Layer Deep CNN (2 Conv + 1 FC) | High feature extraction capacity with minimal FLOPs. |
| **Initialization** | He (Kaiming) Normal | Prevents vanishing/exploding gradients in ReLU layers. |
| **Regularization** | L2 Weight Decay (Lambda = 0.0005) | Reduces overfitting by penalizing large weights. |
| **Optimization** | SGD with Learning Rate Decay | Smooth convergence and high final precision. |
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
- [x] L2 Regularization & Scheduled LR Decay.
- [x] High-precision GPU Benchmark Timers.
- [ ] Accelerated Data Augmentation (CUDA Kernels).
- [ ] Object Detection (Anchor-based kernels).
- [ ] TensorRT Integration for production deployment.

## License
Distributed under the MIT License. See `LICENSE` for more information.
