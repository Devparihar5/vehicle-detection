from flask import Flask, render_template, request, Response, jsonify, send_from_directory
import os
import cv2
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO("yolov8n.pt")
video_path = None

# Shared dict for counting vehicles
vehicle_counts = {"car": 0, "bus": 0, "truck": 0, "motorbike": 0}

# Class ID to label map
class_map = {2: "car", 5: "bus", 7: "truck", 3: "motorbike"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['video']
    filename = secure_filename(file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(video_path)
    return render_template('index.html', stream=True, video=filename)

def generate_frames(video_path):
    global vehicle_counts
    vehicle_counts = {"car": 0, "bus": 0, "truck": 0, "motorbike": 0}

    cap = cv2.VideoCapture(video_path)
    seen_ids = {label: set() for label in vehicle_counts}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[2, 3, 5, 7], conf=0.5)
        result = results[0]

        for i in range(len(result.boxes)):
            cls_id = int(result.boxes.cls[i].item())
            track_id = int(result.boxes.id[i].item())
            label = class_map.get(cls_id)
            if label and track_id not in seen_ids[label]:
                seen_ids[label].add(track_id)
                vehicle_counts[label] += 1

        annotated_frame = result.plot()

        # Draw count table directly on the frame
        x, y0 = 20, 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (0, 255, 0)
        line_height = 25

        cv2.putText(annotated_frame, 'Vehicle Counts:', (x, y0), font, 0.7, (255, 255, 255), 2)

        for i, (k, v) in enumerate(vehicle_counts.items(), 1):
            text = f"{k.capitalize()}: {v}"
            y = y0 + i * line_height
            cv2.putText(annotated_frame, text, (x, y), font, font_scale, font_color, 2)

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
    if not os.path.exists(video_path):
        return "Video not found", 404
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
