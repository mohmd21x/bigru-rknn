import argparse
from onnxmltools.utils.float16_converter import convert_float_to_float16
from onnxmltools.utils import load_model, save_model

node_block_list = ['Sin_689', 'MatMul_694', 'MatMul_698', 'Clip_699', 'Clip_700', 'Sub_702', 'Sub_704']

def main():
    # Set up an argument parser
    parser = argparse.ArgumentParser(description='Convert ONNX model from Float32 to Float16.')
    parser.add_argument('--input_model', type=str, required=True, help='Path to the input ONNX model file.')
    parser.add_argument('--output_model', type=str, required=True, help='Path for saving the converted ONNX model file.')
    
    # Parse arguments
    args = parser.parse_args()

    # Load the model
    print(f"Loading model from {args.input_model}")
    onnx_model = load_model(args.input_model)

    # Convert model from Float32 to Float16
    print("Converting model...")
    new_onnx_model = convert_float_to_float16(onnx_model, min_positive_val=1e-7, max_finite_val=1e4, keep_io_types=True, node_block_list=node_block_list)

    # Save the converted model
    print(f"Saving converted model to {args.output_model}")
    save_model(new_onnx_model, args.output_model)

    print("Conversion complete.")

if __name__ == "__main__":
    main()

