for f in $(ls -S *.onnx | tac); 
do 
echo Process "$f"
python3 symbolic_shape_infer.py --input "$f" --output "$f" --auto_merge; 
done

