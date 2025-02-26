import cv2
import numpy as np
import time
import threading
import subprocess
import edge_tts
import asyncio
import sounddevice as sd
import scipy.io.wavfile as wav
import requests
import clip
import torch
from PIL import Image
import os
import ssl
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (for frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# (Optional) Disable SSL verification if needed:
ssl._create_default_https_context = ssl._create_unverified_context

# ---------------------- Configuration ----------------------
DETECTION_INTERVAL = 3
MIN_CONFIDENCE = 1
VOICE = "en-US-AriaNeural"
INITIAL_DETECTION_TIME = 5  # Seconds for initial object detection phase
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ---------------------- Load CLIP Model ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# ---------------------- Load Comprehensive Candidate Labels ----------------------
def load_imagenet_labels(label_file="imagenet_classes.txt"):
    with open(label_file, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


imagenet_labels = load_imagenet_labels()
tokenized_labels = clip.tokenize(imagenet_labels).to(device)

# ---------------------- Global Variables ----------------------
audio_active = threading.Event()
detected_objects = set()
initial_detection_done = False


# ---------------------- Text-to-Speech ----------------------
async def async_text_to_speech(text):
    try:
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save("response.mp3")
        return "response.mp3"
    except Exception as e:
        print(f"TTS error: {str(e)}")
        return None


# ---------------------- Azure API Communication ----------------------
def get_azure_reply_output(file_path):
    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "audio/mpeg")}
            response = requests.post(
                "http://chatscopetdai.duckdns.org:8000/voice-assistant/audio-message",
                files=files
            )
            response.raise_for_status()
            return response.content
    except Exception as e:
        print(f"Azure API Error: {str(e)}")
        return None


# ---------------------- Image Processing Endpoint ----------------------
@app.post("/process-frame")
async def process_frame_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(tokenized_labels)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarities = (image_features @ text_features.T).squeeze(0)
        values, indices = similarities.topk(3)
        current_objects = [imagenet_labels[i] for i in indices.tolist()]

        global detected_objects, initial_detection_done
        if not initial_detection_done:
            detected_objects.update(current_objects)

        return {"detected_objects": list(detected_objects), "current_objects": current_objects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------- Audio Processing Endpoint ----------------------
@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded audio
        with open("user_input.webm", "wb") as f:
            content = await file.read()
            f.write(content)

        # Convert to MP3
        subprocess.run(["ffmpeg", "-y", "-i", "user_input.webm", "user_input.mp3"], check=True)

        # Process with Azure
        reply_audio = get_azure_reply_output("user_input.mp3")
        if reply_audio:
            with open("azure_response.mp3", "wb") as f:
                f.write(reply_audio)
            return FileResponse("azure_response.mp3", media_type="audio/mpeg")

        raise HTTPException(status_code=500, detail="Azure API error")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------- Initialization Thread ----------------------
def initialize_detection():
    global initial_detection_done
    time.sleep(INITIAL_DETECTION_TIME)
    initial_detection_done = True
    print("Initial detection phase completed")


@app.on_event("startup")
def startup_event():
    threading.Thread(target=initialize_detection, daemon=True).start()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)