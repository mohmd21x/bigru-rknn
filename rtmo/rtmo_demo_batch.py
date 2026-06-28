#!/usr/bin/python3

import time
import cv2
from pathlib import Path
import argparse
from rtmo_gpu import RTMO_GPU_Batch, draw_skeleton, resize_to_fit_screen, draw_bbox # Ensure to import RTMO_GPU_Batch

def process_video(video_path, body_estimator, batch_size=4):
    cap = cv2.VideoCapture(video_path)

    batch_frames = []
    frame_idxs = []

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        batch_frames.append(frame)
        frame_idxs.append(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Process the batch when it's full
        if len(batch_frames) == batch_size:
            s = time.time()
            batch_bboxes, batch_bboxes_scores, batch_keypoints, batch_scores = body_estimator.__batch_call__(batch_frames)
            det_time = time.time() - s
            fps = round(batch_size / det_time, 1)
            print(f'Batch det: {fps} FPS')

            for i, keypoints in enumerate(batch_keypoints):
                scores = batch_scores[i]
                frame = batch_frames[i]
                bboxes = batch_bboxes[i]
                bboxes_scores = batch_bboxes_scores[i]
                img_show = frame.copy()
                img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.3, line_width=2)
                img_show = draw_bbox(img_show, bboxes, bboxes_scores)
                img_show = resize_to_fit_screen(img_show, 720, 480)
                cv2.putText(img_show, f'{fps:.1f}', (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow(f'{video_path}', img_show)
                cv2.waitKey(10)

            # Clear the batch
            batch_frames = []

    # Process remaining frames if any
    if batch_frames:
        # Padding
        while len(batch_frames) < batch_size:
            # Option 1: Add a black frame
            # black_frame = np.zeros_like(batch_frames[0])
            # batch_frames.append(black_frame)

            # Option 2: Duplicate the last frame
            batch_frames.append(batch_frames[-1])
        batch_bboxes, batch_bboxes_scores, batch_keypoints, batch_scores = body_estimator.__batch_call__(batch_frames)
        for i, keypoints in enumerate(batch_keypoints):
            scores = batch_scores[i]
            frame = batch_frames[i]
            bboxes = batch_bboxes[i]
            bboxes_scores = batch_bboxes_scores[i]
            img_show = frame.copy()
            img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.3, line_width=2)
            img_show = draw_bbox(img_show, bboxes, bboxes_scores)
            img_show = resize_to_fit_screen(img_show, 720, 480)
            cv2.imshow(f'{video_path}', img_show)
            #cv2.waitKey(10)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process the path to a video file folder.')
    parser.add_argument('path', type=str, help='Path to the folder containing video files (required)')
    parser.add_argument('model_path', type=str, help='Path to a RTMO ONNX model file (required)')
    parser.add_argument('batch_size', type=int, help='Path to a RTMO ONNX input batch size (required)')

    # Parse the command-line arguments
    args = parser.parse_args()

    onnx_model = args.model_path  # Example: 'rtmo-s_8xb32-600e_body7-640x640.onnx'

    # Instantiate the RTMO_GPU_Batch instead of RTMO_GPU
    body_estimator = RTMO_GPU_Batch(model=onnx_model)

    for mp4_path in Path(args.path).glob('*'):
        process_video(str(mp4_path), body_estimator, args.batch_size)
