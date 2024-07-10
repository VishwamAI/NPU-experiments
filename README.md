# NPU-experiments

## Overview
This repository contains experiments and performance measurements for Neural Processing Units (NPUs). The goal is to develop advanced NPUs that can assist in various fields such as quantum physics, robotics, and software engineering.

## Repository Structure
- `npu_test_suite.py`: Script for testing and measuring the performance of ONNX models on NPUs.
- `inspect_tensor_shapes.py`: Script for inspecting the tensor shapes of ONNX models.
- `convert_to_onnx.py`: Script for converting models to ONNX format.
- `requirements.txt`: List of dependencies required for running the scripts.

## Setup
To set up the environment and install the necessary dependencies, run the following command:
```bash
pip install -r requirements.txt
```

## Running the NPU Test Suite
To measure the performance of an ONNX model using the `npu_test_suite.py` script, use the following command:
```bash
python npu_test_suite.py <model_path> --iterations <number_of_iterations>
```
- `<model_path>`: Path to the ONNX model file.
- `<number_of_iterations>`: (Optional) Number of iterations to run for performance measurement. Default is 10.

## Inspecting Tensor Shapes
To inspect the tensor shapes of an ONNX model using the `inspect_tensor_shapes.py` script, use the following command:
```bash
python inspect_tensor_shapes.py <model_path>
```
- `<model_path>`: Path to the ONNX model file.

## Converting Models to ONNX
To convert a model to ONNX format using the `convert_to_onnx.py` script, follow the instructions provided in the script.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any questions or inquiries, please contact the project maintainers.
