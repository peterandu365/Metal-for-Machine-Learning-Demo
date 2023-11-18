# PyTorch Model Inference on CPU and MPS

This repository contains a Jupyter Notebook demonstrating the usage of PyTorch for running a deep learning model on both a CPU and an Apple MPS (Metal Performance Shaders) device. It's a practical example for understanding the performance differences between CPU and GPU (MPS) in model inference.

## Overview

The notebook included in this repository (`model_inference_cpu_mps.ipynb`) contains two main segments:

1. **CPU Inference**: Running a ResNet50 model using a CPU device.
2. **MPS Inference**: Running the same ResNet50 model on an MPS device, falling back to CPU if MPS is not available.

This is particularly useful for those who are interested in comparing the performance and efficiency of model inference on different hardware setups.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch
- torchvision
- Jupyter Notebook

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/peterandu365/Metal-for-Machine-Learning-Demo
cd Metal-for-Machine-Learning-Demo
```

Install the required packages:

```bash
pip install torch torchvision jupyter
```

Run Jupyter Notebook:

```bash
jupyter notebook
```

Open the `model_inference_cpu_mps.ipynb` file in the Jupyter Notebook interface.

### Usage

The notebook is divided into two sections for CPU and MPS inference. Simply run each cell in the notebook to see how the model performs on each device. Make sure to observe the printed time taken for inference on each device for comparison.

## Contributing

Contributions to enhance the functionality or efficiency of the notebook are welcome. Please feel free to fork the repository and open a pull request with your improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch Team for the fantastic deep learning framework.
- Apple for providing the MPS technology.

