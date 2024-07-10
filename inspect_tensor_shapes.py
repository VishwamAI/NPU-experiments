import onnx

def inspect_tensor_shapes(model_path):
    model = onnx.load(model_path)
    graph = model.graph

    print("Inputs:")
    for input_tensor in graph.input:
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"Name: {input_tensor.name}, Shape: {shape}")

    print("\nOutputs:")
    for output_tensor in graph.output:
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"Name: {output_tensor.name}, Shape: {shape}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python inspect_tensor_shapes.py <model_path>")
    else:
        model_path = sys.argv[1]
        inspect_tensor_shapes(model_path)
