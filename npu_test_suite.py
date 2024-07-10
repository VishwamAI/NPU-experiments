import onnxruntime as ort
import numpy as np
import psutil
import time
import traceback
import argparse

def generate_input_data(model):
    """
    Generate random input data for the given ONNX model.

    Args:
        model (onnxruntime.InferenceSession): The ONNX model session.

    Returns:
        dict: A dictionary containing the generated input data.
    """
    input_data = {}
    for input_info in model.get_inputs():
        input_name = input_info.name
        shape = input_info.shape
        # Replace dynamic dimensions (None or -1) with a default size of 1
        shape = [dim if isinstance(dim, int) else 1 for dim in shape]
        dtype = np.float32 if input_info.type == 'tensor(float)' else np.int32
        input_data[input_name] = np.random.rand(*shape).astype(dtype)
    return input_data

def measure_performance(model_path, iterations=10):
    """
    Measure the performance of the given ONNX model.

    Args:
        model_path (str): The path to the ONNX model file.
        iterations (int): The number of iterations to run for performance measurement.

    Returns:
        tuple: A tuple containing the average inference time and memory usage.
    """
    try:
        session = ort.InferenceSession(model_path)
        input_data = generate_input_data(session)
        io_binding = session.io_binding()
        for name, data in input_data.items():
            io_binding.bind_input(name, 'cpu', data.dtype, data.shape, data.ctypes.data)

        start_time = time.time()
        for _ in range(iterations):
            session.run_with_iobinding(io_binding)
        end_time = time.time()

        duration = end_time - start_time
        avg_time = duration / iterations
        memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)

        return avg_time, memory_usage
    except Exception as e:
        print("Error during performance measurement:")
        print(traceback.format_exc())
        return None, None

def main():
    """
    Main function to measure the performance of the ONNX model.
    """
    parser = argparse.ArgumentParser(description="Measure the performance of an ONNX model.")
    parser.add_argument("model_path", type=str, help="The path to the ONNX model file.")
    parser.add_argument("--iterations", type=int, default=10, help="The number of iterations to run for performance measurement.")
    args = parser.parse_args()

    avg_time, memory_usage = measure_performance(args.model_path, args.iterations)
    if avg_time is not None:
        print(f"Average inference time: {avg_time:.4f} seconds")
        print(f"Memory usage: {memory_usage:.2f} MB")
    else:
        print("Performance measurement failed.")

if __name__ == "__main__":
    main()
