import torch
from transformers import GPT2Model, GPT2Tokenizer

def convert_gpt2_to_onnx(output_path):
    """
    Convert the GPT-2 model to ONNX format and save it to the specified output path.

    Args:
        output_path (str): The path to save the converted ONNX model.
    """
    # Load pre-trained GPT-2 model and tokenizer
    model_name = "gpt2"
    model = GPT2Model.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Set the model to evaluation mode
    model.eval()

    # Generate dummy input data
    input_text = "Hello, how are you?"
    inputs = tokenizer(input_text, return_tensors="pt")

    # Define input and output names
    input_names = ["input_ids", "attention_mask"]
    output_names = ["last_hidden_state"]

    # Modify the model to not use past_key_values during export
    def forward_no_past_key_values(self, input_ids, attention_mask=None, *args, **kwargs):
        if 'past_key_values' in kwargs:
            kwargs['past_key_values'] = None
        return self.forward(input_ids=input_ids, attention_mask=attention_mask, *args, **kwargs)

    model.forward = forward_no_past_key_values.__get__(model, GPT2Model)

    # Export the model to ONNX format
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"},
                      "attention_mask": {0: "batch_size", 1: "sequence_length"},
                      "last_hidden_state": {0: "batch_size", 1: "sequence_length"}},
        opset_version=14,
        do_constant_folding=True
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert GPT-2 model to ONNX format.")
    parser.add_argument("output_path", type=str, help="The path to save the converted ONNX model.")
    args = parser.parse_args()

    convert_gpt2_to_onnx(args.output_path)
