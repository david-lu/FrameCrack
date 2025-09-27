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
        frames.append(frame)
        
        if max_frames and len(frames) >= max_frames:
            break
    
    cap.release()
    return np.array(frames)