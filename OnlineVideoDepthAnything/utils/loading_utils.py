import cv2
from tqdm import tqdm

cv2.setNumThreads(1)
import numpy as np
import glob
import os
from natsort import natsorted
import imageio.v3 as iio
import tifffile
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from OnlineVideoDepthAnything.models.utils.preprocessing import VideoPreprocessor

def save_side_by_side(original, pred_color, out_path, fps=20):
    """
    original: [T,H,W,3] float32
    pred_color: [T,H,W,3] uint8
    """
    h, w = original.shape[1], original.shape[2]
    combined_w = w * 2

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (combined_w, h)
    )

    for orig, pred in zip(original, pred_color):
        orig_uint8 = (orig * 255).astype(np.uint8)[..., ::-1]  # RGB → BGR
        combined = np.concatenate([orig_uint8, pred], axis=1)
        writer.write(combined)

    writer.release()


def colorize_pred(pred, vmin=None, vmax=None, add_colorbar=False):
    if pred.ndim == 4 and pred.shape[-1] == 1:
        pred = pred[..., 0]

    single_frame = pred.ndim == 2

    if single_frame:
        pred = pred[None, ...]

    vmin = float(pred.min()) if vmin is None else float(vmin)
    vmax = float(pred.max()) if vmax is None else float(vmax)

    cmap = matplotlib.colormaps["Spectral"]
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    frames = []

    for frame in pred:
        rgb = cmap(norm(frame))[..., :3] * 255
        bgr = rgb.astype(np.uint8)[..., ::-1]
        frames.append(bgr)

    video = np.stack(frames)

    # Add colorbar
    if add_colorbar:
        fig, ax = plt.subplots(figsize=(1.0, 5))
        fig.subplots_adjust(left=0.2, right=0.7)

        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = fig.colorbar(sm, cax=ax)
        cbar.set_label("Depth")

        fig.canvas.draw()
        colorbar = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)

        colorbar = colorbar[..., ::-1]  # RGB → BGR

        colorbar = cv2.resize(
            colorbar,
            (colorbar.shape[1], video.shape[1])
        )

        video = np.concatenate(
            [video, np.broadcast_to(
                colorbar[None, ...],
                (video.shape[0], *colorbar.shape)
            )],
            axis=2
        )

    if single_frame:
        return video[0]

    return video

def load_video_as_numpy(path):
    """
    Loads Video
    -----------------
    Loads a video as a numpy array of type float 32 and in range 0 to 1

    Parameters
    ----------
    path : str
        Path to the video you want to load

    Returns
    -------
    result : np.ndarray
        Numpy array of shape [Time, Height, Width, Channels]
    """
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()
    return np.stack(frames)

def load_frames_as_numpy(image_list: str):
    """
    Loads Frames from a Folder
    -----------------
    Loads a bunch of frames as a numpy array of type float 32 and in range 0 to 1
    Supported Formats: jpg, jpeg, png, webp, bmp, tiff

    Parameters
    ----------
    image_list : str
        Path to the folder containing the frames you want to load

    Returns
    -------
    result : np.ndarray
        Numpy array of shape [Time, Height, Width, Channels]
    """
    if isinstance(image_list, str):
        img_str = image_list
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp', '*.tiff']
        image_list = []
        for ext in image_extensions:
            image_list.extend(glob.glob(os.path.join(img_str, ext)))
    
    image_list = natsorted(image_list)

    # Load first frame to get ref values
    first_frame = iio.imread(image_list)
    h, w, c = first_frame.shape

    frames = np.zeros((len(image_list), h, w, c), dtype=np.float32)

    for i, image_path in enumerate(image_list):
        image = iio.imread(image_path)
        frames[i] = image.astype(np.float32) / 255.0

    return frames

def save_video_mp4(video: np.ndarray, path: str, fps=20):
    vmin = float(video.min())
    vmax = float(video.max())

    h, w = video.shape[1], video.shape[2]

    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    # colormap = cm.get_cmap('Spectral')
    colormap = matplotlib.colormaps["Spectral"]

    for frame in video:
        # Normalize globally
        norm = (np.clip(frame, vmin, vmax) - vmin) / (vmax - vmin + 1e-8)

        # Apply matplotlib colormap → RGBA in [0..1]
        colored = colormap(norm)[..., :3]   # drop alpha channel

        # Convert to uint8 + BGR for OpenCV
        colored_bgr = (colored * 255).astype(np.uint8)[..., ::-1]

        writer.write(colored_bgr)

    writer.release()

def save_depth_video_mp4(video: np.ndarray, path: str, fps=20):
    h, w = video.shape[1], video.shape[2]

    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    for frame in tqdm(video):
        writer.write(frame)

    writer.release()

def save_predictions_tiff(pred: np.ndarray, path: str):
    tifffile.imwrite(path, pred.astype(np.float32))
