from PIL import Image
import cv2
import os
import random
from typing import List, Tuple
from moviepy.editor import VideoFileClip
from moviepy.editor import VideoFileClip
from transformers import pipeline
import torch
import numpy as np

def extract_audio_from_video(video_path, audio_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path, codec='mp3')
    audio_clip.close()
    video_clip.close()

def transcribe(audio_path):
    model = "openai/whisper-medium.en"
    device = 0 if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        chunk_length_s=10,
        device=device,
    )

    out = pipe(audio_path, return_timestamps=True)
    return out['text']

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


from typing import List, Tuple
from PIL import Image
import cv2
import random

def read_video_frames_with_exact_timestamps_interval(video_path: str, interval_seconds: int) -> List[Tuple[float, Image.Image]]:
    """
    Reads a video file and returns a list of tuples, each containing a frame (as a PIL Image)
    and its exact timestamp in seconds, sampling a frame every specified number of seconds.
    
    Parameters:
    - video_path: Path to the video file.
    - interval_seconds: The number of seconds between each frame sample.
    
    Returns:
    - frames_with_exact_timestamps: A list of tuples, each containing an exact timestamp in seconds
      and a randomly selected frame (as a PIL Image) for the specified interval.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError("Error opening video file")
    
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frames_with_exact_timestamps = []
    frames_buffer = []
    frame_indices = []
    last_sampled_second = -interval_seconds  # Initialize to start sampling at 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        frame_second = int(frame_number / frame_rate)
        
        if frame_second >= last_sampled_second + interval_seconds:
            if frames_buffer:
                # Sample and clear the buffer at the beginning of each interval
                selected_index = random.randint(0, len(frames_buffer) - 1)
                selected_frame_number = frame_indices[selected_index]
                selected_frame = frames_buffer[selected_index]
                
                exact_timestamp = selected_frame_number / frame_rate
                pil_image = Image.fromarray(selected_frame)
                
                frames_with_exact_timestamps.append((exact_timestamp, pil_image))
                frames_buffer.clear()
                frame_indices.clear()
                last_sampled_second = frame_second  # Update last sampled second
                
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_buffer.append(frame_rgb)
        frame_indices.append(frame_number)
    
    cap.release()
    
    return frames_with_exact_timestamps

def select_active_frames_and_timestamps_every_second(video_path: str, activity_threshold: float) -> List[Tuple[float, np.ndarray]]:
    """
    Selects frames with significant activity from a video file, based on the average magnitude of optical flow,
    sampling one frame every second. Returns their timestamps and the frames themselves if they exceed the activity threshold.

    Parameters:
    - video_path: Path to the video file.
    - activity_threshold: Threshold for the average magnitude of flow to consider a frame as having significant activity.

    Returns:
    - A list of tuples, each containing the timestamp and the frame (as a numpy array) for frames with activity above the threshold,
      sampled every second.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error opening video file")

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    active_frames_and_timestamps = []
    
    while True:
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if 'prev_gray' in locals():
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                motion_measure = np.mean(magnitude)

                if motion_measure > activity_threshold:
                    timestamp = frame_number / frame_rate
                    active_frames_and_timestamps.append((timestamp, frame))
            
            prev_gray = gray

        next_frame_number = frame_number + (frame_rate - (frame_number % frame_rate))
        cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame_number)

    cap.release()
    
    return active_frames_and_timestamps

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
