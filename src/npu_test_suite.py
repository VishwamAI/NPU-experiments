import onnxruntime as ort
import numpy as np
import psutil
import time
import traceback
import argparse
import pandas as pd
from sklearn.datasets import load_iris
from transformers import AutoTokenizer  # Add this import for handling tokenization

# Dictionary to handle model-specific sequence lengths
MODEL_SEQUENCE_LENGTHS = {
    'gpt2': 128,
    'bert': 512,
    'roberta': 512,
    't5': 512,
    'distilbert': 512,
    'albert': 512,
    'xlm': 512,
    'xlnet': 512,
    'bart': 512,
    'electra': 512,
    'longformer': 4096,
    'reformer': 4096,
    'bigbird': 4096,
    'gpt3': 2048,
    'megatron': 2048,
    'turing-nlg': 2048,
    'switch-transformer': 2048,
    'vit': 224,
    'deit': 224,
    'swin': 224,
    'resnet': 224,
    'efficientnet': 224,
    'mobilenet': 224,
    'densenet': 224,
    'inception': 299,
    'nasnet': 331,
    'xception': 299,
    'yolo': 416,
    'ssd': 300,
    'faster-rcnn': 800,
    'mask-rcnn': 800,
    'retinanet': 800,
    'unet': 512,
    'deeplabv3': 512,
    'segformer': 512,
    'dpt': 512,
    'gluoncv': 512,
    'pytorchcv': 512,
    'open-mmlab': 512,
    'detectron2': 800,
    'mmdetection': 800,
    'openpose': 368,
    'alphapose': 368,
    'hrnet': 256,
    'simplepose': 256,
    'pose-resnet': 256,
    'pose-hg': 256,
    'pose-hrnet': 256,
    'pose-dla': 256,  # General category for all pose-dla models
}

# Global constant for the default sequence length
DEFAULT_SEQUENCE_LENGTH = 768

def generate_random_input_data(model, model_name='gpt2'):
    """
    Generate random input data for the given ONNX model.

    Args:
        model (onnxruntime.InferenceSession): The ONNX model session.
        model_name (str): The name of the model to use for determining the sequence length.

    Returns:
        dict: A dictionary containing the generated random input data.
    """
    input_data = {}
    sequence_length = MODEL_SEQUENCE_LENGTHS.get(model_name, DEFAULT_SEQUENCE_LENGTH)
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

def generate_input_data(model, model_name='gpt2', use_dataset=False, dataset_name=None):
    """
    Generate input data for the given ONNX model.

    Args:
        model (onnxruntime.InferenceSession): The ONNX model session.
        model_name (str): The name of the model to use for determining the sequence length.
        use_dataset (bool): Whether to use a dataset for input data generation.
        dataset_name (str): The name of the dataset to use if use_dataset is True.

    Returns:
        dict: A dictionary containing the generated input data.
    """
    input_data = {}
    if use_dataset and dataset_name:
        if dataset_name == 'iris':
            data = load_iris()
            X = data.data
            for input_info in model.get_inputs():
                input_name = input_info.name
                shape = input_info.shape
                # Replace dynamic dimensions (None or -1) with the shape of the dataset
                shape = [dim if isinstance(dim, int) else X.shape[i] for i, dim in enumerate(shape)]
                dtype = np.float32 if input_info.type == 'tensor(float)' else np.int64 if input_info.type == 'tensor(int64)' else np.int32
                input_data[input_name] = X.astype(dtype)
        elif model_name in MODEL_SEQUENCE_LENGTHS:
            if 'gpt' in model_name or 'bert' in model_name or 'roberta' in model_name or 't5' in model_name:
                try:
                    # Load the tokenizer for the specified transformer model
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    tokenizer.pad_token = tokenizer.eos_token
                    # Generate a sample input text and tokenize it
                    text = ["This is a sample input text for the model."] * MODEL_SEQUENCE_LENGTHS[model_name]
                    inputs = tokenizer(text, return_tensors='np', padding='max_length', max_length=MODEL_SEQUENCE_LENGTHS[model_name], truncation=True)
                    for input_name, input_values in inputs.items():
                        input_data[input_name] = input_values
                except Exception as e:
                    print(f"Error loading tokenizer for model {model_name}: {e}")
                    input_data = generate_random_input_data(model, model_name)
            elif 'resnet' in model_name or 'efficientnet' in model_name or 'mobilenet' in model_name:
                # Generate random image data for image models
                for input_info in model.get_inputs():
                    input_name = input_info.name
                    shape = input_info.shape
                    shape = [dim if isinstance(dim, int) else 224 for dim in shape]  # Default to 224x224 for image models
                    dtype = np.float32 if input_info.type == 'tensor(float)' else np.int32
                    input_data[input_name] = np.random.rand(*shape).astype(dtype)
            else:
                input_data = generate_random_input_data(model, model_name)
        else:
            input_data = generate_random_input_data(model, model_name)
    else:
        input_data = generate_random_input_data(model, model_name)
    return input_data

def measure_performance(model_path: str, iterations: int = 5, model_name: str = 'gpt2', use_dataset: bool = False, dataset_name: str = None, memory_threshold: int = 4000, verbose: bool = True) -> tuple[float, float, int, float, float]:
    """
    Measure the performance of the given ONNX model.

    Args:
        model_path (str): The path to the ONNX model file.
        iterations (int): The number of iterations to run for performance measurement.
        model_name (str): The name of the model to use for determining the sequence length.
        use_dataset (bool): Whether to use a dataset for input data generation.
        dataset_name (str): The name of the dataset to use if use_dataset is True.
        memory_threshold (int): The memory usage threshold in MB. If exceeded, the number of iterations will be reduced.
        verbose (bool): Whether to print detailed output during iterations.

    Returns:
        tuple: A tuple containing the average inference time, memory usage, actual number of iterations performed, average latency, and throughput.
    """
    try:
        session = ort.InferenceSession(model_path)
        input_data = generate_input_data(session, model_name, use_dataset, dataset_name)
        io_binding = session.io_binding()
        for name, data in input_data.items():
            if verbose:
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
                try:
                    output_shape[1] = MODEL_SEQUENCE_LENGTHS[model_name]  # Use the provided sequence length
                except KeyError:
                    if verbose:
                        print(f"Model name '{model_name}' not found in MODEL_SEQUENCE_LENGTHS. Using default sequence length of {DEFAULT_SEQUENCE_LENGTH}.")
                    output_shape[1] = DEFAULT_SEQUENCE_LENGTH  # Default sequence length
            output_dtype = np.float32 if output_info.type == 'tensor(float)' else np.int32
            output_buffer = np.empty(output_shape, dtype=output_dtype)
            io_binding.bind_output(output_name, 'cpu', 0, output_buffer.dtype, output_buffer.shape, output_buffer.ctypes.data)

        start_time = time.time()
        latencies = []
        for i in range(iterations):
            iter_start_time = time.time()
            session.run_with_iobinding(io_binding)
            iter_end_time = time.time()
            latencies.append(iter_end_time - iter_start_time)
            current_memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)
            if verbose:
                print(f"Iteration {i+1}/{iterations}, Current memory usage: {current_memory_usage:.2f} MB")
            if current_memory_usage > memory_threshold:  # Use the memory_threshold parameter
                if verbose:
                    print(f"Memory usage exceeded {memory_threshold} MB. Reducing the number of iterations.")
                iterations = i + 1
                break
        end_time = time.time()

        duration = end_time - start_time
        avg_time = duration / iterations  # Use the actual number of iterations performed
        memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)
        avg_latency = sum(latencies) / len(latencies)
        throughput = iterations / duration

        return avg_time, memory_usage, iterations, avg_latency, throughput  # Return the actual number of iterations performed
    except Exception as e:
        print("Error during performance measurement:")
        print(traceback.format_exc())
        return None, None, None, None, None

def main():
    """
    Main function to measure the performance of the ONNX model.
    """
    parser = argparse.ArgumentParser(description="Measure the performance of an ONNX model.")
    parser.add_argument("model_path", type=str, help="The path to the ONNX model file.")
    parser.add_argument("--iterations", type=int, default=5, help="The number of iterations to run for performance measurement.")
    parser.add_argument("--model_name", type=str, default='gpt2', help="The name of the model to use for determining the sequence length.")
    parser.add_argument("--use_dataset", action="store_true", help="Whether to use a dataset for input data generation.")
    parser.add_argument("--dataset_name", type=str, help="The name of the dataset to use if use_dataset is True.")
    parser.add_argument("--memory_threshold", type=int, default=4000, help="The memory usage threshold in MB. If exceeded, the number of iterations will be reduced.")
    parser.add_argument("--verbose", action="store_true", help="Whether to print detailed output during iterations.")
    args = parser.parse_args()

    avg_time, memory_usage, actual_iterations, avg_latency, throughput = measure_performance(args.model_path, args.iterations, args.model_name, args.use_dataset, args.dataset_name, args.memory_threshold, args.verbose)
    if avg_time is not None:
        print(f"Average inference time: {avg_time:.4f} seconds")
        print(f"Memory usage: {memory_usage:.2f} MB")
        print(f"Actual iterations performed: {actual_iterations}")
        print(f"Average latency: {avg_latency:.4f} seconds")
        print(f"Throughput: {throughput:.2f} iterations/second")
    else:
        print("Performance measurement failed.")

if __name__ == "__main__":
    main()
