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
from fastapi import FastAPI, File, UploadFile
app = FastAPI()

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

imagenet_labels = load_imagenet_labels()  # Ensure this file exists with ImageNet classes.
tokenized_labels = clip.tokenize(imagenet_labels).to(device)

# ---------------------- Global Variables ----------------------
audio_active = threading.Event()

# ---------------------- Text-to-Speech using edge_tts ----------------------
async def async_text_to_speech(text):
    try:
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save("response.mp3")
        subprocess.run(["afplay", "response.mp3"], check=True)
    except Exception as e:
        print(f"TTS error: {str(e)}")
        # Optionally, add another TTS fallback here if desired.

def text_to_speech(text):
    asyncio.run(async_text_to_speech(text))

# ---------------------- New: Save Text-to-Audio with Custom Filename ----------------------
async def save_text_to_audio_file(text, filename):
    try:
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(filename)
    except Exception as e:
        print(f"TTS error: {str(e)}")

def text_to_audio_file(text, filename):
    asyncio.run(save_text_to_audio_file(text, filename))

# ---------------------- Azure API for Voice Assistant ----------------------
def get_azure_reply_output(file_path):
    """
    Sends the audio file at file_path to your Azure endpoint and returns the audio reply blob.
    """
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "audio/mpeg")}
        response = requests.post(
            "http://chatscopetdai.duckdns.org:8000/voice-assistant/audio-message",
            files=files
        )
        response.raise_for_status()
        return response.content

def azure_get_ai_reply(file_path):
    """
    Wrapper for get_azure_reply_output that saves the reply as an MP3 file and returns its path.
    """
    try:
        reply_audio = get_azure_reply_output(file_path)
        output_file = "azure_response.mp3"
        with open(output_file, "wb") as f:
            f.write(reply_audio)
        return output_file
    except Exception as e:
        print(f"Azure API Error: {str(e)}")
        return None

def generate_and_respond(audio_file_path):
    """
    Sends the recorded audio file to the Azure API, saves the reply as an MP3,
    and plays it back.
    """
    try:
        reply_audio = get_azure_reply_output(audio_file_path)
        with open("azure_response.mp3", "wb") as f:
            f.write(reply_audio)
        print("AI (Azure) responded.")
        subprocess.run(["afplay", "azure_response.mp3"], check=True)
    except Exception as e:
        print(f"Azure API Error: {str(e)}")

# ---------------------- Process Frame with CLIP ----------------------
def process_frame(frame):
    """
    Runs CLIP on the frame to identify candidate objects.
    """
    # Convert the frame (BGR) to a PIL image (RGB)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(tokenized_labels)

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute similarity scores and get top 3 predictions
    similarities = (image_features @ text_features.T).squeeze(0)
    values, indices = similarities.topk(3)
    detected_objects = [imagenet_labels[i] for i in indices.tolist()]
    return detected_objects

# ---------------------- Video Processing Loop ----------------------
def video_processing_loop():
    """
    Captures video from the camera and uses CLIP for initial object detection.
    Then, generates an initial welcome message based on detected objects.
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    detected_objects = set()
    start_time = time.time()
    initial_response_sent = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # During initial detection phase, accumulate detected objects
            if (time.time() - start_time) < INITIAL_DETECTION_TIME:
                objects = process_frame(frame)
                detected_objects.update(objects)
            else:
                if not initial_response_sent:
                    if detected_objects:
                        objects_str = ', '.join(detected_objects)
                        print(objects_str)
                        prompt = (
                            f"You are an AI assistant on a video call. The camera currently sees: {objects_str}. "
                            "Based on this or a good welcoming message for person, please provide an engaging, thoughtful conversation or ask an interesting question to start a conversation."
                        )
                    else:
                        prompt = (
                            "You are an AI assistant on a video call. The camera didn't detect any notable objects. "
                            "Please start the conversation with an engaging question."
                        )

                    # Synthesize the dynamic prompt to an audio file named "prompt.mp3"
                    text_to_audio_file(prompt, "prompt.mp3")
                    # Send the synthesized prompt audio to Azure to get a dynamic AI reply
                    reply_audio_file = azure_get_ai_reply("prompt.mp3")
                    if reply_audio_file:
                        print("Initial AI response:")
                        subprocess.run(["afplay", reply_audio_file], check=True)
                    initial_response_sent = True
                    audio_active.set()

                # Draw mic status on the frame
                status = "Mic: ON" if audio_active.is_set() else "Mic: OFF"
                cv2.putText(frame, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0) if audio_active.is_set() else (0, 0, 255), 2)

            cv2.imshow('AI Video Chat', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                if audio_active.is_set():
                    audio_active.clear()
                else:
                    audio_active.set()
            elif key == ord('h'):
                print("Using Azure voice assistant endpoint for conversation.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

# ---------------------- Voice Conversation Loop ----------------------
def voice_conversation_loop():
    """
    Continuously listens for user audio, converts it to MP3,
    sends it to the Azure endpoint, and plays back the AI's reply.
    """
    fs = 16000  # Sample rate for recording

    while True:
        if audio_active.is_set():
            print("\nListening... (Press 'm' to toggle mic)")
            recording = sd.rec(int(5 * fs), samplerate=fs, channels=1)
            sd.wait()

            # Save recording as WAV
            wav.write("user_input.wav", fs, recording)
            # Convert WAV to MP3 (ensure ffmpeg is installed)
            subprocess.run(["ffmpeg", "-y", "-i", "user_input.wav", "user_input.mp3"], check=True)
            # Send the MP3 file to Azure and play the reply
            generate_and_respond("user_input.mp3")
        else:
            time.sleep(0.1)

# ---------------------- Main ----------------------
if __name__ == "__main__":
    # Start the voice conversation loop in a separate thread
    voice_thread = threading.Thread(target=voice_conversation_loop, daemon=True)
    voice_thread.start()

    # Start video processing in the main thread
    video_processing_loop()
