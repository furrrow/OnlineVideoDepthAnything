import argparse
import os
import time
from datetime import datetime

import cv2
import torch
import yaml
import numpy as np
from PIL import Image

from models.utils.preprocessing import VideoPreprocessor
from models.video_depth import onlineVideoDepthAnything
from src.utils.loading_utils import load_video_as_numpy, save_video_mp4, save_predictions_tiff, save_side_by_side, \
    colorize_pred, save_depth_video_mp4

from stream_handler import FrameStatus, InputStreamHandler

@torch.no_grad()
def predict_depth(model:onlineVideoDepthAnything, frame:np.ndarray, device, preprocess_device='cpu', input_size=518, fp32=False,
                  print_process_res=False, output_raw=False, return_process_res=False) -> np.ndarray:
    """
    heavily borrowing from onlineVideoDepthAnything.infer_video_depth()
    :param model: nn.Module
            onlineVideoDepthAnything model, initialized outside
    :param frame: torch.Tensor | np.ndarray
            Input data of shape (Batch, Time, Channels, H, W) or (Time, Channels, H, W)
    :param device: str
            torch device string of type: 'cuda:0' defining the cuda device to run on or 'cpu'
    :param preprocess_device: str, default='cpu'
            torch device string defining the preprocess device. Can be a different one.
    :param input_size: int, default=518
            Defining the rought resolution for processing. The exact Resolution will be automatically calculated.
    :param fp32: bool, default=False
            Defining if the model is run in fp32 or fp16 (if False). Since fp32 is only marginally better, we recommend to use fp16.
    :param print_process_res: bool, default=False
            Prints out the resolution the preprocessing has resized the original input to.
    :param output_raw: bool, default=False
            Returns the original prediction of oVDA. Will be in the resolution of the preprocessed input video. If set to False,
            the depth prediction is resized to the original input video size
    :param return_process_res: bool, default=False
            Returns predictions and the process resolution. This is used for keeping track for downstream processing.

    :return:out_depth : np.ndarray
            predicted depth of shape (B, T, H, W)
    """
    # Preprocessing of frames
    frame_height, frame_width = frame.shape[:2]
    ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
    if ratio > 1.78:  # VDA recommend to process video with ratio smaller than 16:9 due to memory limitation
        input_size = int(input_size * 1.777 / ratio)
        input_size = round(input_size / 14) * 14

    pre = VideoPreprocessor(input_size=input_size, device=preprocess_device, ensure_multiple_of=14,
                            keep_aspect_ratio=True, resize_method='lower_bound')
    prepared_frame = pre.preprocess(frame)
    b, t, c, h, w = prepared_frame.size()

    # Handle multiple videos
    if output_raw:
        out_depth = np.zeros((t, h, w))
    else:
        out_depth = np.zeros((t, frame_height, frame_width))

    print_resize_warining = False
    cache_size = 0

    mask_indices = torch.tensor(list(range(1, model.cache_size))).to(device)
    input_position = torch.tensor([0, 1]).to(device)

    input_cache = model.setup_cache(h, w, device)
    if not fp32:
        for key in input_cache:
            input_cache[key] = input_cache[key].half()
            model.half()
            prepared_frame = prepared_frame.half()

    # Predict depths
    depths = []
    times = []
    with torch.no_grad():
        # aiming for prepared_frame: [1, 1, 3, 518, 924]
        input_frame = prepared_frame.to(device)
        depth_pred, output_cache = model.forward(
            input_frame,
            input_cache=input_cache,
            mask_indices=mask_indices,
            input_position=input_position,
        )
        depth_pred = depth_pred.squeeze(1).unflatten(0, (1, 1))
        depth_pred = depth_pred.squeeze(dim=0)
        depths.append(depth_pred.cpu())

        # Update Cache
        if cache_size == model.cache_size - 1:
            for key in input_cache:
                input_cache[key] = torch.cat([input_cache[key][:, :, :, 1:cache_size, :], output_cache[key],
                                              torch.zeros_like(output_cache[key])], dim=3)
        else:
            for key in input_cache:
                input_cache[key][:, :, :, cache_size, :] = output_cache[key][:, :, :, 0, :]

        cache_size += 1
        cache_size = min(cache_size, model.cache_size - 1)

        remaining = list(range(cache_size, model.cache_size))
        padding = [model.cache_size - 1] * (cache_size - 1)

        mask_indices = torch.tensor(remaining + padding).to(device)
        input_position = torch.tensor([cache_size, cache_size]).to(device)

        # The depths are in the process resolution.
        depths = torch.stack(depths, dim=1).float().numpy()
        if output_raw:
            out_depth[0] = depths[0]
        else:
            if depths[0][0].shape != (frame_height, frame_width):
                h, w = depths[0][0].shape
                gh, gw = (frame_height, frame_width)

            out_depth[0] = \
            torch.nn.functional.interpolate(torch.from_numpy(depths), size=(frame_height, frame_width), mode='bilinear',
                          align_corners=True)[0, :, :, :].numpy()

    if not return_process_res:
        return out_depth
    else:
        return out_depth, (h, w)


def main():
    # Initialize predictor (single-GPU streaming)
    viz_results = True
    save_video_toggle = True
    stream_type = "webcam" # ["yarp", "video", "webcam"]
    video_path = None
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
        yarp_port_name=yarp_port,
    )
    print(f"Opening source: {stream_type}")
    src.open()

    peak_memory = 0
    frame_idx = 0

    frame_timestamps = []  # To compute output fps
    video_frames = []  # Buffer of frames for final video save

    stop_processing = False



    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            while stop_processing is not True:
                # Read frame (RGB)
                stream_buffer = src.read()
                if stream_buffer.status == FrameStatus.NO_FRAME:
                    # YARP: no new frame yet; try again.
                    continue
                if stream_buffer.status == FrameStatus.EOS:
                    # End of stream for video/webcam or closed YARP port.
                    break
                frame_rgb = stream_buffer.frame
                pred_depth = predict_depth(model, frame_rgb, device)
                pred_color = colorize_pred(pred_depth[0], vmin=0, vmax=20)
                if viz_results:
                    cv2.imshow(
                        "Livestream", cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)
                    )
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

            output_dir = "./outputs/webcam.mp4"
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