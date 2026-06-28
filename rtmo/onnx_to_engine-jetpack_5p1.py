#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
This script demonstrates how to use the Calibrator API provided by Polygraphy
to calibrate a TensorRT engine to run in INT8 precision.
"""
import numpy as np
from polygraphy.backend.trt import Calibrator, CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, TrtRunner, save_engine, load_plugins, Profile
from polygraphy.logger import G_LOGGER
from termcolor import cprint
load_plugins(plugins=['libmmdeploy_tensorrt_ops.so'])
import cv2
import argparse

G_LOGGER.severity = G_LOGGER.EXTRA_VERBOSE
PREVIEW_CALIBRATOR_OUTPUT = True
    
def calib_data_from_video(batch_size=1):

    # image preproc3ssing taken from rtmlib
    def preprocess(img: np.ndarray):
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        if len(img.shape) == 3:
            padded_img = np.ones(
                (MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3),
                dtype=np.uint8) * 114
        else:
            padded_img = np.ones(MODEL_INPUT_SIZE, dtype=np.uint8) * 114

        ratio = min(MODEL_INPUT_SIZE[0] / img.shape[0],
                    MODEL_INPUT_SIZE[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
        padded_img[:padded_shape[0], :padded_shape[1]] = resized_img

        return padded_img, ratio

    cap = cv2.VideoCapture(filename=VIDEO_PATH)
    imgs = []
    while cap.isOpened():
        
        success, frame = cap.read()
        if success:
            img, ratio = preprocess(frame) # pad & resize
            img = img.transpose(2, 0, 1) # transpose to 1,3,416,416
            img = np.ascontiguousarray(img, dtype=np.float32) # to f32
            img = img[None, :, :, :] # add batch dim
            
            imgs.append(img)
            if len(imgs) == batch_size:
                batch_img = np.vstack(imgs)
                yield {"input": batch_img}
                imgs = []
                # cprint(f'batch_img.shape = {batch_img.shape}', 'yellow')
        else:
            break
            
    cap.release()

def main(onnx_path, engine_path, batch_size):

    # We can provide a path or file-like object if we want to cache calibration data.
    # This lets us avoid running calibration the next time we build the engine.
    #
    # TIP: You can use this calibrator with TensorRT APIs directly (e.g. config.int8_calibrator).
    # You don't have to use it with Polygraphy loaders if you don't want to.
    if batch_size < 1: # dynamic batch size

        profiles = [
            # The low-latency case. For best performance, min == opt == max.
            Profile().add("input", 
                        min=(1, 3, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1]), 
                        opt=(4, 3, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1]), 
                        max=(9, 3, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1])),
        ]
    
    else: # fixed
        profiles = [
            # The low-latency case. For best performance, min == opt == max.
            Profile().add("input", 
                        min=(batch_size, 3, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1]), 
                        opt=(batch_size, 3, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1]), 
                        max=(batch_size, 3, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1])),
        ]

    opt_batch_size = profiles[0]['input'].opt[0]
    calibrator = Calibrator(data_loader=calib_data_from_video(opt_batch_size))

    # We must enable int8 mode in addition to providing the calibrator.
    build_engine = EngineFromNetwork(
        NetworkFromOnnxPath(f"{onnx_path}"), config=CreateConfig(
                                                                use_dla=False,
                                                                tf32=True, 
                                                                fp16=True, 
                                                                int8=True, 
                                                                precision_constraints="prefer",
                                                                sparse_weights=True,
                                                                calibrator=calibrator,
                                                                profiles=profiles,
                                                                max_workspace_size = 2 * 1024 * 1024 * 1024,
                                                                allow_gpu_fallback=True,
                                                                )
    )

    # When we activate our runner, it will calibrate and build the engine. If we want to
    # see the logging output from TensorRT, we can temporarily increase logging verbosity:
    save_engine(build_engine, f'{engine_path}')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process a video file.")
    parser.add_argument("video_path", type=str, help="The path to the video file used to calibrate int8 engine")
    parser.add_argument("onnx_path", type=str, help="The path to the input ONNX model file")
    parser.add_argument("engine_path", type=str, help="The path to the exported TensorRT Engine model file")
    parser.add_argument("--batch_size", type=int, default=-1, help="Input batch size (not specified if dynamic)")
    args = parser.parse_args()
    VIDEO_PATH = args.video_path
    MODEL_INPUT_SIZE=(416,416) if 'rtmo-t' in args.onnx_path else (640,640)
    
    if PREVIEW_CALIBRATOR_OUTPUT:
        cprint('You are previwing video used to calibrate TensorRT int8 engine model ...', 'yellow')
        for output_dict in calib_data_from_video(): 
            if output_dict:
                image = output_dict['input'] # get frame
                image_to_show = image.squeeze(0).transpose(1, 2, 0) / 255.0 # to-uint8 transpose remove batch dim
                cv2.imshow(VIDEO_PATH,image_to_show)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop if 'q' is pressed
                    break
        cv2.destroyAllWindows()  # Close all OpenCV windows
            
    main(args.onnx_path, args.engine_path, args.batch_size)
