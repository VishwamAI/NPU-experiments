import onnxruntime as ort
import numpy as np
import psutil
import time
import traceback
import argparse

def generate_input_data(model, sequence_length=768):
    """
    Generate random input data for the given ONNX model.

    Args:
        model (onnxruntime.InferenceSession): The ONNX model session.
        sequence_length (int): The sequence length to use for inputs with dynamic dimensions.

    Returns:
        dict: A dictionary containing the generated input data.
    """
    input_data = {}
    for input_info in model.get_inputs():
        input_name = input_info.name
        shape = input_info.shape
        # Replace dynamic dimensions (None or -1) with a default size of 1
        shape = [dim if isinstance(dim, int) else 1 for dim in shape]
        # Set a specific sequence length for inputs with dynamic dimensions
        if len(shape) > 1 and shape[1] == 1:
            shape[1] = sequence_length  # Use the provided sequence length
        dtype = np.float32 if input_info.type == 'tensor(float)' else np.int64 if input_info.type == 'tensor(int64)' else np.int32
        input_data[input_name] = np.random.rand(*shape).astype(dtype)
    return input_data

def measure_performance(model_path, iterations=10, sequence_length=768):
    """
    Measure the performance of the given ONNX model.

    Args:
        model_path (str): The path to the ONNX model file.
        iterations (int): The number of iterations to run for performance measurement.
        sequence_length (int): The sequence length to use for inputs with dynamic dimensions.

    Returns:
        tuple: A tuple containing the average inference time and memory usage.
    """
    try:
        session = ort.InferenceSession(model_path)
        input_data = generate_input_data(session, sequence_length)
        io_binding = session.io_binding()
        for name, data in input_data.items():
            print(f"Binding input: {name}, dtype: {data.dtype}, shape: {data.shape}, buffer_ptr: {data.ctypes.data}")
            io_binding.bind_input(name, 'cpu', 0, data.dtype, data.shape, data.ctypes.data)

        # Bind outputs
        for output_info in session.get_outputs():
            output_name = output_info.name
            output_shape = output_info.shape
            # Replace dynamic dimensions (None or -1) with a default size of 1
            output_shape = [dim if isinstance(dim, int) else 1 for dim in output_shape]
            # Set a specific sequence length for outputs with dynamic dimensions
            if len(output_shape) > 1 and output_shape[1] == 1:
                output_shape[1] = sequence_length  # Use the provided sequence length
            output_dtype = np.float32 if output_info.type == 'tensor(float)' else np.int32
            output_buffer = np.empty(output_shape, dtype=output_dtype)
            io_binding.bind_output(output_name, 'cpu', 0, output_buffer.dtype, output_buffer.shape, output_buffer.ctypes.data)

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
    parser.add_argument("--sequence_length", type=int, default=768, help="The sequence length to use for inputs with dynamic dimensions.")
    args = parser.parse_args()

    avg_time, memory_usage = measure_performance(args.model_path, args.iterations, args.sequence_length)
    if avg_time is not None:
        print(f"Average inference time: {avg_time:.4f} seconds")
        print(f"Memory usage: {memory_usage:.2f} MB")
    else:
        print("Performance measurement failed.")

if __name__ == "__main__":
    main()
