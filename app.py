from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
import shutil
import subprocess
import uuid

app = FastAPI()

@app.post("/process-video/")
async def create_upload_file(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    video_path = f"/workspace/DeepSeek-VL/{video_id}.mp4"
    txt_path = video_path.rsplit('.', 1)[0] + ".txt"

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run your processing script here
    subprocess.run(["python3", "inference_1_3b.py", "--video_path", video_path], check=True)

    if os.path.exists(txt_path):
        return FileResponse(txt_path, media_type='application/octet-stream', filename=f"{video_id}.txt")
    else:
        return {"error": "Failed to process video"}

# Additional endpoint to check if the service is running
@app.get("/")
def read_root():
    return {"message": "DeepSeek Video Processing Service is Running"}
