import argparse
import os
import time
from datetime import datetime

import cv2
import torch
import yaml
import numpy as np
from PIL import Image

from models.video_depth import onlineVideoDepthAnything
from OnlineVideoDepthAnything.utils.loading_utils import colorize_pred, save_depth_video_mp4
from custom_utils.stream_handler import FrameStatus, InputStreamHandler
from ptc_depth import PTCDepth

def main():
    # Initialize predictor (single-GPU streaming)
    viz_results = True
    save_video_toggle = True
    use_ptc = True
    stream_type = "webcam" # ["yarp", "video", "webcam"]
    video_path = "assets/example_videos/Cars_and_Gasstation.mp4"
    output_folder = "outputs"
    webcam_index = 0
    yarp_port = "/sam3/rgbImage:i"
    oVDA_config_path = './configs/oVDA_c16.yaml'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load config
    with open(oVDA_config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Load model
    model = onlineVideoDepthAnything(**config['net']).to(device)
    ckpt = torch.load(config['pretrained_path'], map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()

    # Initialize input source
    src = InputStreamHandler(
        kind=stream_type,
        video_path=video_path,
        webcam_index=webcam_index,
        # fps_request=5,
        yarp_port_name=yarp_port,
    )
    print(f"Opening source: {stream_type}")
    src.open()
    stream_buffer = src.read()
    h, w, c = stream_buffer.frame.shape
    if use_ptc:
        ptc_pipeline = PTCDepth(H=h, W=w, fx=388.90, fy=388.666, cx=318.32, cy=251.50)
    peak_memory = 0
    frame_idx = 0

    frame_timestamps = []  # To compute output fps
    video_frames = []  # Buffer of frames for final video save

    stop_processing = False

    prev_time = time.time()
    try:
        while stop_processing is not True:
            # Read frame (RGB)
            stream_buffer = src.read()
            if stream_buffer.status == FrameStatus.NO_FRAME:
                # YARP: no new frame yet; try again.
                continue
            if stream_buffer.status == FrameStatus.EOS:
                # End of stream for video/webcam or closed YARP port.
                break
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            frame_rgb = stream_buffer.frame
            frame_gbr = frame_rgb[:, :, ::-1]
            depth = model.infer_depth_streaming(frame_rgb, device=device, preprocess_device="cpu").squeeze(0)
            if use_ptc:
                dummy_baseline = np.float64(0.1)
                normalized_depth = (depth - depth.min()) / (depth.max() - depth.min())
                result = ptc_pipeline(frame_rgb, normalized_depth, dummy_baseline)
                if True in np.isnan(result['depth']):
                    print("nan in ptc using original")
                else:
                    print("using ptc")
                    depth = result['depth']
            depth = depth.astype(np.uint8)
            pred_color = colorize_pred(depth, vmin=0, vmax=15, add_colorbar=True)
            if viz_results:
                # Display FPS
                cv2.putText(pred_color,f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0, 255, 0),2)
                cv2.imshow(
                    "Livestream", cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)
                )
                # cv2.imshow("Livestream", pred_color)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_processing = True
            # Update running statistics
            frame_idx += 1
            frame_timestamps.append(time.time())
            if save_video_toggle:
                video_frames.append(cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))
            current_peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3  # GB
            peak_memory = max(peak_memory, current_peak_memory)
            print(
                f"Processed frame {frame_idx}. "
                f"Current peak memory: {current_peak_memory:.2f} GB, "
                f"Overall peak memory: {peak_memory:.2f} GB.",
                end="\r",
            )
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, stopping processing gracefully...")

    finally:
        # Source cleanup
        src.close()

        if save_video_toggle:
            if len(frame_timestamps) >= 2:
                elapsed = frame_timestamps[-1] - frame_timestamps[0]
                # Use average FPS over the whole run
                effective_fps = (len(frame_timestamps) - 1) / elapsed
            else:
                effective_fps = 30.0

            output_dir = f"{output_folder}/{stream_type}.mp4"
            save_depth_video_mp4(
                video=np.array(video_frames),
                path=output_dir,
                fps=effective_fps,
            )
            print(
                f"\nSaved video to {output_dir} at {effective_fps:.2f} FPS."
            )

        # Close any OpenCV windows
        if viz_results:
            cv2.destroyAllWindows()

        print(f"Processed {frame_idx} frames.")
        print(f"Peak GPU memory usage: {peak_memory:.2f} GB.")
if __name__ == "__main__":
    main()