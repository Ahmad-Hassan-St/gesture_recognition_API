from flask import Flask, request, jsonify, Response
import cv2
import numpy as np
import joblib
import base64
from flask_cors import CORS
from src.hand_tracker_nms import HandTrackerNMS
import src.extra
import threading

app = Flask(__name__)
CORS(app)

PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"

connections = src.extra.connections
int_to_char = src.extra.classes

# Initialize the HandTrackerNMS with debugging
try:
    detector = HandTrackerNMS(
        PALM_MODEL_PATH,
        LANDMARK_MODEL_PATH,
        ANCHORS_PATH,
        box_shift=0.2,
        box_enlarge=1.3
    )
    print("HandTrackerNMS initialized successfully.")
except Exception as e:
    print(f"Error initializing HandTrackerNMS: {e}")

try:
    gesture_clf = joblib.load(r'models/gesture_clf.pkl')
    print("Gesture classifier loaded successfully.")
except Exception as e:
    print(f"Error loading gesture classifier: {e}")

detected_letter = ""
detected_letter_lock = threading.Lock()

def detect_gesture(frame):
    global detected_letter
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, bboxes, joints = detector(image)
    if points is not None:
        src.extra.draw_points(points, frame)
        pred_sign = src.extra.predict_sign(joints, gesture_clf, src.extra.classes)
        with detected_letter_lock:
            detected_letter = pred_sign
        # Overlay detected letter on frame
        cv2.putText(frame, pred_sign, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    return frame

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Gesture Detection API!"})

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        capture = cv2.VideoCapture(0)
        while True:
            hasFrame, frame = capture.read()
            if not hasFrame:
                break
            processed_frame = detect_gesture(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_detected_letter', methods=['GET'])
def get_detected_letter():
    global detected_letter
    with detected_letter_lock:
        return jsonify({"detected_letter": detected_letter})

@app.route('/detect_image', methods=['POST'])
def detect_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file:
        image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        detect_gesture(image)  # Update the image in place
        with detected_letter_lock:
            detected_letter_local = detected_letter  # Fetch the letter detected during image processing
        _, buffer = cv2.imencode('.jpg', image)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        return jsonify({"detected_letter": detected_letter_local, "image": processed_image})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=700)
