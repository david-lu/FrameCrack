import cv2
import numpy as np
from typing import Optional


def video_to_numpy(video_path: str, max_frames: Optional[int] = None) -> np.ndarray:
    """
    Read video file and convert to numpy array.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to read (None for all)
        
    Returns:
        numpy array of shape (num_frames, height, width, channels)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        
        if max_frames and len(frames) >= max_frames:
            break
    
    cap.release()
    return np.array(frames)

def resize_video(video: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    F = video.shape[0]
    out = np.empty((F, new_h, new_w, video.shape[3]), dtype=video.dtype)
    interp = cv2.INTER_CUBIC if (new_h > video.shape[1] or new_w > video.shape[2]) else cv2.INTER_AREA
    for i in range(F):
        out[i] = cv2.resize(video[i], (new_w, new_h), interpolation=interp)  # note: (w, h)
    return out