# CuVision-Engine

### High-Performance Native Computer Vision for the Edge.

CuVision-Engine is a high-performance, low-latency Computer Vision framework written entirely in native C++/CUDA. Engineered for maximum throughput on NVIDIA hardware, it provides optimized implementations for Classification, Segmentation, and Object Detection by leveraging cuDNN and cuBLAS directly.

## Key Features
- Native CUDA/cuDNN: Bypasses heavy deep learning frameworks for maximum hardware utilization.
- Optimized for Edge: Designed for real-time inference on NVIDIA Jetson, mobile GPUs, and production workstations.
- Deep CNN Architecture: Includes He Initialization, L2 Regularization, and SGD with Learning Rate Decay.
- Modular Design: Clean separation between data pipelines, network architectures, and utilities.

## Project Structure
- classification/: Core CNN implementations, training loops, and benchmarking.
  - network/: CUDA kernels and cuDNN descriptors.
  - dataset/: High-performance binary data loaders and preprocessing scripts.
- segmentation/: (Coming Soon) Pixel-wise classification prototypes.
- detection/: (Coming Soon) Anchor-based object detection.

## Getting Started

### Prerequisites
- NVIDIA GPU with CUDA Compute Capability 6.0+
- CUDA Toolkit 11.0+
- cuDNN 8.0+
- C++ Compiler (MSVC on Windows, GCC on Linux)

### Compilation
Navigate to the module directory (e.g., classification) and run the build script:
powershell
cd classification
.\compile.ps1


### Training
1. Prepare the dataset:
bash
python dataset/prepare_dataset.py

2. Run the classifier:
powershell
.\dnn_classifier.exe


## Roadmap
- [x] Deep CNN Core: 3-layer architecture with He Initialization.
- [ ] Data Augmentation: Real-time CUDA-accelerated flips, rotations, and color jittering.
- [ ] TensorRT Integration: 10x-20x faster inference for edge deployment.
- [ ] YOLO Implementation: Real-time object detection kernels.

## License
MIT License - See the [LICENSE](LICENSE) file for details.
