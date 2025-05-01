from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
import cv2
import os

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
    global video_path, vehicle_counts
    vehicle_counts = {"car": 0, "bus": 0, "truck": 0, "motorbike": 0}
    file = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(video_path)
    return render_template('index.html', stream=True)

def generate_frames():
    global vehicle_counts
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    seen_ids = {
        "car": set(),
        "bus": set(),
        "truck": set(),
        "motorbike": set()
    }

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

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        frame_count += 1

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/counts')
def get_counts():
    return jsonify(vehicle_counts)

if __name__ == '__main__':
    app.run(debug=True)
