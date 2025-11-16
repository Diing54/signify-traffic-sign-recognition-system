# app.py (Clean Version - No Bluetooth Fix)
import cv2
import numpy as np
from flask import Flask, Response, url_for
from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.post_processors import pp_od_yolo_ultralytics
from gtts import gTTS
import os # For playing audio
import threading
import time
from collections import deque
import tempfile # To save temporary audio files

# --- Initialize Flask App ---
app = Flask(__name__)

# --- State Variables for TTS Logic ---
# Use a dictionary to store the last announcement time for each class ID
announced_signs_memory = {}
ANNOUNCE_MEMORY_DURATION = 3.0 # Seconds a sign type stays in memory to prevent immediate repeats
last_announcement_time = 0
ANNOUNCE_COOLDOWN = 2.0 # Minimum seconds between ANY announcement

# --- Function to Run gTTS and Play Audio (with cleaning) ---
def announce_sign_thread_gtts(text):
    try:
        # --- Clean up label text ---
        # 1. Remove category prefix (e.g., "regulatory--")
        parts = text.split('--')
        if len(parts) > 1:
            sign_name = parts[1] # Take the part after '--'
        else:
            sign_name = text # Use the whole name if no '--'

        # 2. Replace underscores with spaces
        clean_text = sign_name.replace('_', ' ')
        # --- End cleaning ---
        
        tts = gTTS(text=clean_text, lang='en', slow=False)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_filename = fp.name
            tts.save(temp_filename)
            
        # Play the audio file using mpg123 (quiet mode)
        os.system(f"mpg123 -q {temp_filename}")
        
        # Clean up the temporary file
        os.remove(temp_filename)
        
    except Exception as e:
        print(f"Error during gTTS/playback: {e}")

# --- Model and Camera Setup ---
class TrafficSignDetector(Model):
    # (Initialization code remains the same)
    def __init__(self):
        super().__init__(
            model_file="/home/dgara/packerOut.zip",
            model_type=MODEL_TYPE.CONVERTED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(
            "/home/dgara/labels.txt",
            dtype=str, delimiter="\n"
        )
    def post_process(self, output_tensors):
        return pp_od_yolo_ultralytics(output_tensors)

print("Deploying model to camera... This may take a moment.")
device = AiCamera(frame_rate=16) # Lowered frame rate
model = TrafficSignDetector()
device.deploy(model)
annotator = Annotator()
CONF_THRESHOLD = 0.5
print("✅ Model deployed. Starting camera stream.")

def generate_frames():
    """Generator function to yield camera frames and trigger TTS."""
    global last_announcement_time, announced_signs_memory

    while True:
      try:
        with device as stream:
            if stream is None:
                print("Error: Could not get camera stream.")
                time.sleep(1)
                continue
            for frame in stream:
                current_time = time.time()
                labels_for_display = [] 

                # --- 1. Filter Detections ---
                valid_detections = frame.detections[frame.detections.confidence > CONF_THRESHOLD]

                # --- 2. Update and Clean Announcement Memory ---
                # Remove class IDs from memory if they haven't been announced recently
                keys_to_remove = [
                    class_id for class_id, timestamp in announced_signs_memory.items()
                    if current_time - timestamp > ANNOUNCE_MEMORY_DURATION
                ]
                if keys_to_remove:
                    for key in keys_to_remove:
                        del announced_signs_memory[key]

                # --- 3. Identify New, Announceable Signs in This Frame ---
                new_signs_this_frame = []
                for det_idx, score, class_id, _ in valid_detections:
                    labels_for_display.append(f"{model.labels[class_id]}: {score:.2f}") 
                    if class_id not in announced_signs_memory:
                        new_signs_this_frame.append({'label': model.labels[class_id], 'score': score, 'id': class_id})

                # --- 4. Decide Which Sign to Announce (if any) ---
                can_announce_now = (current_time - last_announcement_time > ANNOUNCE_COOLDOWN)

                if new_signs_this_frame and can_announce_now:
                    new_signs_this_frame.sort(key=lambda x: x['score'], reverse=True)
                    sign_to_announce = new_signs_this_frame[0]
                    print(f"Announcing: {sign_to_announce['label']} (Conf: {sign_to_announce['score']:.2f})")

                    tts_label = sign_to_announce['label'] 
                    thread = threading.Thread(target=announce_sign_thread_gtts, args=(tts_label,), daemon=True)
                    thread.start()

                    announced_signs_memory[sign_to_announce['id']] = current_time
                    last_announcement_time = current_time
                # --- End TTS Logic ---

                # --- 5. Annotate and Yield Frame ---
                annotator.annotate_boxes(frame, valid_detections, labels=labels_for_display, alpha=0.3, corner_radius=10)
                bgr_image = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
                (flag, encodedImage) = cv2.imencode(".jpg", bgr_image)
                if not flag: continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
      except Exception as e:
            print(f"Error in generate_frames loop: {e}")
            print("Attempting to reconnect to camera stream...")
            time.sleep(2) # Wait before retrying

@app.route("/")
def index():
    # (HTML remains the same)
    return f"""
    <html> <head><title>YOLOv8 Traffic Sign Detection with TTS</title></head> <body> <h1>Live Traffic Sign Detection Feed (with Audio)</h1> <p>Streaming from Raspberry Pi IMX500 Camera. Ensure Bluetooth audio is connected.</p> <img src="{url_for('video_feed')}" width="640" height="480"> </body> </html>
    """

@app.route("/video_feed")
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# --- Main Execution ---
if __name__ == '__main__':
    if os.system("which mpg123 > /dev/null") != 0:
         print("WARNING: 'mpg123' command not found. Audio playback might fail.")
         print("Please install it using: sudo apt install mpg123")
    
    print("Starting Flask server with cleaned gTTS enabled (requires internet).")
    app.run(host='0.0.0.0', port=8000, debug=False)
