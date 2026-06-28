#!/usr/bin/python3

import time
import cv2
from pathlib import Path
import argparse
import os
from rtmo_gpu import RTMO_GPU_Batch, draw_skeleton, resize_to_fit_screen, draw_bbox

if __name__ == "__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process the path to a video file folder.')
    parser.add_argument('path', type=str, help='Path to the folder containing video files (required)')
    parser.add_argument('model_path', type=str, help='Path to a RTMO ONNX (or engine) model file (required)')
    parser.add_argument('--yolo_nas_pose', action='store_true', help='Use YOLO NAS Pose (flat format only) instead of RTMO Model')
    parser.add_argument('--batch_size', type=int, default=1, help='Path to a RTMO ONNX input batch size')

    # Parse the command-line arguments
    args = parser.parse_args()

    model = args.model_path # 'rtmo-s_8xb32-600e_body7-640x640.onnx'

    body = RTMO_GPU_Batch(model=model, is_yolo_nas_pose=args.yolo_nas_pose, batch_size=args.batch_size)

    for mp4_path in Path(args.path).glob('*'):
    
        # Now, use the best.url, which is the direct video link for streaming
        cap = cv2.VideoCapture(filename=os.path.abspath(mp4_path))
        frame_idx = 0
        s = time.time()
        while cap.isOpened():
            success, frame = cap.read()
            frame_idx += 1

            if not success:
                break

            frame_out, bboxes, bboxes_scores, keypoints, scores = body(frame)

            if keypoints is not None:
                if frame_idx % args.batch_size == 0 and frame_idx:
                    current_time = time.time()
                    det_time = current_time - s
                    fps = round(args.batch_size / det_time, 1)
                    print(f'det: {fps} FPS')
                    s = current_time

                img_show = frame_out.copy()
                
                # if you want to use black background instead of original image,
                # img_show = np.zeros(img_show.shape, dtype=np.uint8)

                img_show = draw_skeleton(img_show,
                                        keypoints,
                                        scores,
                                        kpt_thr=0.3,
                                        line_width=2)
                img_show = draw_bbox(img_show, bboxes, bboxes_scores)
                img_show = resize_to_fit_screen(img_show, 720, 480)
                cv2.putText(img_show, f'{fps:.1f}', (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow(f'{model}', img_show)
                cv2.waitKey(10)
