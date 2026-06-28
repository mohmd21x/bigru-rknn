#!/bin/sh

# Loop through all .onnx files in the current directory
for model in *.onnx; do
    echo "Sanitizing model: $model"
    # Perform sanitization using polygraphy
    polygraphy surgeon sanitize "$model" --fold-constants --output "$model"
    echo "Sanitization complete for: $model"
done

echo "All models have been sanitized."
