from typing import List
from PIL import Image
import cv2
import os
import random

from typing import List, Tuple

from typing import List, Tuple

def read_video_frames_with_exact_timestamps(video_path: str) -> List[Tuple[float, Image.Image]]:
    """
    Reads a video file and returns a list of tuples, each containing a frame (as a PIL Image)
    and its exact timestamp in seconds.
    
    Parameters:
    - video_path: Path to the video file.
    
    Returns:
    - frames_with_exact_timestamps: A list of tuples, each containing an exact timestamp in seconds
      and a randomly selected frame (as a PIL Image) for that second.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError("Error opening video file")
    
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frames_with_exact_timestamps = []
    current_second = -1
    frames_buffer = []
    frame_indices = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        frame_second = int(frame_number / frame_rate)
        
        if frame_second != current_second:
            if frames_buffer:
                selected_index = random.randint(0, len(frames_buffer) - 1)
                selected_frame_number = frame_indices[selected_index]
                selected_frame = frames_buffer[selected_index]
                
                exact_timestamp = selected_frame_number / frame_rate
                pil_image = Image.fromarray(selected_frame)
                
                frames_with_exact_timestamps.append((exact_timestamp, pil_image))
                frames_buffer.clear()
                frame_indices.clear()
            
            current_second = frame_second
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_buffer.append(frame_rgb)
        frame_indices.append(frame_number)
    
    if frames_buffer:
        selected_index = random.randint(0, len(frames_buffer) - 1)
        selected_frame_number = frame_indices[selected_index]
        selected_frame = frames_buffer[selected_index]
        
        exact_timestamp = selected_frame_number / frame_rate
        pil_image = Image.fromarray(selected_frame)
        frames_with_exact_timestamps.append((exact_timestamp, pil_image))
    
    cap.release()
    
    return frames_with_exact_timestamps




def save_images_to_folder(images: List[Image.Image], folder_path: str):
    """
    Saves a list of PIL Images to the specified folder path. Images are saved in PNG format.

    Parameters:
    - images: List of PIL Image objects to be saved.
    - folder_path: The destination folder path where images will be saved.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    for idx, image in enumerate(images):
        image_path = os.path.join(folder_path, f"image_{idx}.png")
        image.save(image_path)
