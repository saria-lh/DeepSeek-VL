from PIL import Image
import cv2
import os
import numpy as np
from moviepy.editor import VideoFileClip
from transformers import pipeline
import torch

def extract_audio_from_video(video_path: str, audio_path: str) -> None:
    """
    Extracts audio from a video file and saves it as an MP3 file.
    
    Prints a message and returns None if the video has no audio.
    Returns 1 if the audio extraction and save operation is successful.
    """
    with VideoFileClip(video_path) as video_clip:
        if video_clip.audio is None:
            print("The video has no audio.")
            return None
        
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_path, codec='mp3')
        return 1



def transcribe(audio_path: str) -> str:
    """
    Transcribes audio to text using OpenAI's Whisper model.
    """
    model = "openai/whisper-medium.en"
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, else CPU

    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        chunk_length_s=10,
        device=device,
    )
    result = pipe(audio_path, return_timestamps=False)
    return result['text']

def extract_frames_interval_ts(video_path: str, interval_seconds: int = 1) -> list:
    """Extracts frames at specified interval seconds from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error opening video file")

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frames_with_exact_timestamps = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        frame_second = frame_number / frame_rate
        if frame_second % interval_seconds == 0:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames_with_exact_timestamps.append((frame_second, pil_image))

    cap.release()
    return frames_with_exact_timestamps

def extract_active_frames_ts(video_path: str, activity_threshold: float) -> list:
    """Extracts frames based on activity level from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error opening video file")

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    active_frames_and_timestamps = []
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_number % frame_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            motion_measure = np.mean(magnitude)

            if motion_measure > activity_threshold:
                timestamp = frame_number / frame_rate
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                active_frames_and_timestamps.append((timestamp, frame_pil))

            prev_gray = gray

    cap.release()
    return active_frames_and_timestamps

def save_images_to_folder(images: list, folder_path: str) -> None:
    """Saves a list of PIL Images to a folder."""
    os.makedirs(folder_path, exist_ok=True)
    for idx, image in enumerate(images):
        image_path = os.path.join(folder_path, f"image_{idx}.png")
        image.save(image_path)
