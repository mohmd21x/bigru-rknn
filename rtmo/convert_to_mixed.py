import numpy as np
import onnx
from onnxconverter_common import auto_mixed_precision_model_path
import argparse
from rtmo_gpu import RTMO_GPU, draw_skeleton
import cv2

PROVIDERS=[('TensorrtExecutionProvider', {'trt_fp16_enable':True,}), 'CUDAExecutionProvider', 'CPUExecutionProvider']

def detect_model_input_size(model_path):
    model = onnx.load(model_path)
    for input_tensor in model.graph.input:
        # Assuming the input node is named 'input'
        if input_tensor.name == 'input':
            tensor_shape = input_tensor.type.tensor_type.shape
            # Extract the dimensions: (batch_size, channels, height, width)
            dims = [dim.dim_value for dim in tensor_shape.dim]
            # Replace dynamic batch size (-1 or 0) with 1
            if dims[0] < 1:
                dims[0] = 1
            return tuple(dims[2:4])  # Return (height, width)
    raise ValueError("Input node 'input' not found in the model")

def load_and_preprocess_image(image_path, preprocesss=None):

    image = cv2.imread(image_path)

    if preprocesss is not None:
        image  = preprocesss(image)

    return image

def compare_result(res1, res2):
    keypoints1, scores1 = res1
    keypoints2, scores2 = res2

    from termcolor import colored

    for j, (d1, d2) in enumerate(zip(keypoints1, keypoints2)):
        print(f'Detection {j}: ')
        for i, (j1, j2) in enumerate(zip(d1, d2)):
            (x1, y1), (x2, y2) = j1, j2
            s1, s2 = scores1[j][i], scores2[j][i]
            print(f"Joint-{i:2d}:")
            print(f'\tOriginal  ({colored("x", "blue")},{colored("y","green")},{colored("score", "red")}) = ({colored("{:4.1f}".format(x1),"blue")}, {colored("{:4.1f}".format(y1),"green")}, {colored("{:5.4f}".format(s1),"red")})')
            print(f'\tConverted ({colored("x", "blue")},{colored("y","green")},{colored("score", "red")}) = ({colored("{:4.1f}".format(x2),"blue")}, {colored("{:4.1f}".format(y2),"green")}, {colored("{:5.4f}".format(s2),"red")})')

def validate_pose(res1, res2, postprocess=None):

    if postprocess is not None:
        res1 = postprocess(res1)
        res2 = postprocess(res2)

    compare_result(res1, res2)

    for r1, r2 in zip(res1, res2):
        if not np.allclose(r1, r2, rtol=args.rtol, atol=args.atol):
            return False
    return True

def infer_on_image(onnx_model, model_input_size, test_image_path):
    body = RTMO_GPU(onnx_model=onnx_model, 
        model_input_size=model_input_size, 
        is_yolo_nas_pose=args.yolo_nas_pose)

    frame = cv2.imread(test_image_path)
    img_show = frame.copy()
    keypoints, scores = body(img_show)

    img_show = draw_skeleton(img_show,
                            keypoints,
                            scores,
                            kpt_thr=0.3,
                            line_width=2)
    img_show = cv2.resize(img_show, (788, 525))
    cv2.imshow(f'{args.target_model_path}', img_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(args):
    model_input_size = detect_model_input_size(args.source_model_path)

    body = RTMO_GPU(onnx_model=args.source_model_path,
                    model_input_size=model_input_size,
                    is_yolo_nas_pose=args.yolo_nas_pose)
    
    def preprocess(image, body, is_yolo_nas_pose):

        img, _ = body.preprocess(image)

        # build input to (1, 3, H, W)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32 if not is_yolo_nas_pose else np.uint8)
        img = img[None, :, :, :]
        return img
    
    image = load_and_preprocess_image(args.test_image_path, lambda img: preprocess(img, body, args.yolo_nas_pose))

    input_feed = {'input': image}

    auto_mixed_precision_model_path.auto_convert_mixed_precision_model_path(source_model_path=args.source_model_path, 
                                    input_feed=input_feed, 
                                    target_model_path=args.target_model_path,
                                    customized_validate_func=lambda res1,res2:validate_pose(res1, res2, body.postprocess), 
                                    rtol=args.rtol, atol=args.atol,
                                    provider=PROVIDERS, 
                                    keep_io_types=True,
                                    verbose=True)

    infer_on_image(args.target_model_path, model_input_size, args.test_image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an ONNX model to mixed precision format.")
    parser.add_argument("source_model_path", type=str, help="Path to the source ONNX model.")
    parser.add_argument("target_model_path", type=str, help="Path where the mixed precision model will be saved.")
    parser.add_argument("test_image_path", type=str, help="Path to a test image for validating the model conversion.")
    parser.add_argument('--rtol', type=float, default=0.01, help=' the relative tolerance to do validation')
    parser.add_argument('--atol', type=float, default=0.001, help=' the absolute tolerance to do validation')
    parser.add_argument('--yolo_nas_pose', action='store_true', help='Use YOLO NAS Pose (flat format only) instead of RTMO Model')

    args = parser.parse_args()
    
    main(args)
