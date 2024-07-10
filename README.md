# NPU-experiments

## Overview
This repository contains experiments and performance measurements for Neural Processing Units (NPUs). The goal is to develop advanced NPUs that can assist in various fields such as quantum physics, robotics, and software engineering.

## Repository Structure
- `src/`: Contains source code files.
  - `npu_test_suite.py`: Script for testing and measuring the performance of ONNX models on NPUs.
  - `inspect_tensor_shapes.py`: Script for inspecting the tensor shapes of ONNX models.
  - `convert_to_onnx.py`: Script for converting models to ONNX format.
- `docs/`: Contains documentation files.
- `tests/`: Contains test scripts and data.
- `examples/`: Contains example usage scripts and models.
- `requirements.txt`: List of dependencies required for running the scripts.
- `data/`: Contains dataset files for input data generation.

## Setup
To set up the environment and install the necessary dependencies, run the following command:
```bash
pip install -r requirements.txt
```
The `requirements.txt` file includes the following dependencies:
- onnx
- onnxruntime
- numpy
- optimum
- psutil
- pydot
- graphviz
- pandas
- scikit-learn

## Running the NPU Test Suite
To measure the performance of an ONNX model using the `npu_test_suite.py` script, use the following command:
```bash
python src/npu_test_suite.py <model_path> --iterations <number_of_iterations> [--use_dataset] [--dataset_name <dataset_name>]
```
- `<model_path>`: Path to the ONNX model file.
- `<number_of_iterations>`: (Optional) Number of iterations to run for performance measurement. Default is 10.
- `--use_dataset`: (Optional) Flag to indicate whether to use a dataset for input data generation.
- `--dataset_name`: (Optional) Name of the dataset to use if `--use_dataset` is specified. Currently supported: 'iris'.

### Example
To run the NPU Test Suite using the Iris dataset for input data generation, use the following command:
```bash
python src/npu_test_suite.py <model_path> --iterations 10 --use_dataset --dataset_name iris
```

## Dataset Functionality
The `npu_test_suite.py` script now supports using real datasets for input data generation. This feature allows for more realistic performance testing of ONNX models on NPUs.

### Supported Datasets
- `iris`: The Iris dataset is a classic dataset used for machine learning and statistical analysis. It contains 150 samples of iris flowers, each with four features: sepal length, sepal width, petal length, and petal width.

### Adding New Datasets
To add a new dataset, follow these steps:
1. Add the dataset file to the `data/` directory.
2. Update the `generate_input_data` function in `npu_test_suite.py` to include a case for the new dataset.
3. Ensure that the dataset is properly loaded and preprocessed within the `generate_input_data` function.

## Inspecting Tensor Shapes
To inspect the tensor shapes of an ONNX model using the `inspect_tensor_shapes.py` script, use the following command:
```bash
python src/inspect_tensor_shapes.py <model_path>
```
- `<model_path>`: Path to the ONNX model file.

## Converting Models to ONNX
To convert a model to ONNX format using the `convert_to_onnx.py` script, follow the instructions provided in the script.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any questions or inquiries, please contact the project maintainers.
