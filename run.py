import cv2
from src.hand_tracker_nms import HandTrackerNMS
import src.extra
import joblib

WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"

connections = src.extra.connections
int_to_char = src.extra.classes

detector = HandTrackerNMS(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)

gesture_clf = joblib.load(r'models\\gesture_clf.pkl')

cv2.namedWindow(WINDOW)
capture = cv2.VideoCapture(0)

# Set the frame width and height (e.g., 1280x720)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

word = []
letter = ""
staticGesture = 0

while True:
    hasFrame, frame = capture.read()
    if not hasFrame:
        break
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, bboxes, joints = detector(image)
    if points is not None:
        src.extra.draw_points(points, frame)
        pred_sign = src.extra.predict_sign(joints, gesture_clf, int_to_char)
        if letter == pred_sign:
            staticGesture += 1
        else:
            letter = pred_sign
            staticGesture = 0
        if staticGesture > 6:
            word.append(letter)
            staticGesture = 0

    if points is None:
        try:
            if word[-1] != " ":
                staticGesture += 1
                if staticGesture > 6:
                    word.append(" ")
                    staticGesture = 0
        except IndexError:
            print("list 'word' is empty")

    src.extra.draw_sign(word, frame, (50, 460))

    frame_resized = cv2.resize(frame, (1280, 720))
    cv2.imshow(WINDOW, frame_resized)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break
    if key == 8:  # Backspace key to delete the last letter
        try:
            del word[-1]
        except IndexError as e:
            print(e)
    if key == ord('c'):  # 'c' key to clear the text
        word = []

capture.release()
cv2.destroyAllWindows()
