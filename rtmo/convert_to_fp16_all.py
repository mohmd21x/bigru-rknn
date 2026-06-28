import os
import subprocess

def convert_onnx_to_fp16_corrected(input_dir):
    # List all files in the input directory
    files = os.listdir(input_dir)
    
    # Filter out files with the .onnx extension
    onnx_files = [file for file in files if file.endswith('.onnx')]
    
    # Iterate over each ONNX file to convert it to FP16
    for onnx_file in onnx_files:
        # Split the file name to insert '.fp16' before the last underscore
        parts = onnx_file.rsplit('_', 1)
        if len(parts) == 2:
            # If there is at least one underscore, insert '.fp16' before the last part
            output_model_name = f"{parts[0]}.fp16_{parts[1]}"
        else:
            # If there's no underscore, just replace '.onnx' with '.fp16.onnx'
            output_model_name = onnx_file.replace('.onnx', '.fp16.onnx')
        
        # Construct the command to run the conversion script
        command = [
            'python3', 'convert_to_fp16.py',
            '--input_model', os.path.join(input_dir, onnx_file),
            '--output_model', os.path.join(input_dir, output_model_name)
        ]
        
        # Execute the command
        subprocess.run(command, check=True)

if __name__ == '__main__':
    # Assuming the current directory is the input directory
    input_dir = os.getcwd()
    convert_onnx_to_fp16_corrected(input_dir)
