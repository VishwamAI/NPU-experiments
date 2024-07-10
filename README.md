# NPU-experiments

## Overview
This repository contains experiments and development for advanced Neural Processing Units (NPUs) designed to assist in various advanced fields such as quantum physics, robotics, and software engineering. The goal is to create NPUs that are more advanced than Microsoft's NPUs, with a focus on building a comprehensive model from the ground up.

## Project Structure
- `npu_test_suite.py`: A script for testing and measuring the performance of the NPU models.
- `.gitattributes`: Configuration file for Git LFS to track large files such as `.onnx` models.
- `README.md`: This file, providing an overview of the project.
- `requirements.txt`: A file listing all the Python dependencies required for the project (to be created).
- `LICENSE`: Licensing information for the project (to be created).

## Getting Started
To get started with the NPU experiments, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/VishwamAI/NPU-experiments.git
   cd NPU-experiments
   ```

2. **Install dependencies:**
   Ensure you have Python 3.10.12 installed. Then, install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the NPU test suite:**
   The `npu_test_suite.py` script can be used to test and measure the performance of the NPU models. Use the following command to run the test suite:
   ```bash
   python npu_test_suite.py --model_path path/to/gpt2.onnx --iterations 10
   ```

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code follows the project's coding standards and includes appropriate tests.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details (to be created).

## Acknowledgments
Special thanks to the contributors and the community for their support and contributions to this project.
