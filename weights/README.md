# Model weights

Place pose and fall-classifier weights here.

## YOLOv8 pose (default backend)

Expected filename: `yolo8n.pt` (or `yolov8n-pose.pt`)

This must be a **pose** model checkpoint (e.g. download `yolov8n-pose.pt` from Ultralytics
and rename/copy it to `yolo8n.pt`).

```bash
# Example download (from repo root)
mkdir -p weights
wget -O weights/yolo8n.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt
```

## RTMO-S pose (alternative backend)

For Python testing, use the MMPOSE ONNX models under `rtmo/` (not this folder):

| File | Description |
|------|-------------|
| `rtmo/rtmo-s.onnx` | RTMO-S with in-graph NMS + keypoint decode (default for `--pose-backend rtmo`) |
| `rtmo/rtmo-t.onnx` | Smaller/faster variant |

```bash
# Standalone RTMO smoke test
python test/rtmo_s_oonx_infer.py /path/to/image.jpg --onnx rtmo/rtmo-s.onnx
```

The older `weights/rtmo_s_no_nms.onnx` + `rtmo_s_dcc_decoder_params.yml` pair is for the
embedded RKNN path in `fall_cpp`, not the Python test backend.

## Fall classifier (BiGRU)

Checkpoints live under `checkpoints/` (not in this folder). ONNX/RKNN exports may also be
placed here for deployment (`bigru_hierarchical.onnx`, etc.).
