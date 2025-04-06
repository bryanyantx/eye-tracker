import cv2
import dlib
import numpy as np
from pynput.mouse import Button, Controller 

from enum import Enum

class mouse_action(Enum):
    moveUp = 1
    moveDown = 2
    moveLeft = 3
    moveRight = 4
    rightClickDown = 5
    leftClickDown = 6
    rightClickUp = 7
    leftClickUp = 8


mouse = Controller()

scale = 10

def move_mouse(action):
    if(action == mouse_action.moveUp):
        mouse.move(0, scale)
    elif(action == mouse_action.moveDown):
        mouse.move(0, -scale)
    elif(action == mouse_action.moveRight):
        mouse.move(scale, 0)
    elif(action == mouse_action.moveDown):
        mouse.move(-scale, 0)

def mouse_click(action):
    if(action == mouse_action.leftClickDown):
        mouse.press(Button.left) 
    elif(action == mouse_action.rightClickDown):
        mouse.press(Button.right) 
    elif(action == mouse_action.leftClickUp):
        mouse.release(Button.left) 
    elif(action == mouse_action.rightClickUp):
        mouse.release(Button.right) 
    


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

def get_eye_landmarks(landmarks, left=True):
    """Extracts eye landmarks for left or right eye."""
    if left:
        return [landmarks.part(i) for i in range(36, 42)]
    else:
        return [landmarks.part(i) for i in range(42, 48)]

def eye_region(frame, landmarks, left=True):
    """Extract the eye region and return a thresholded version."""
    points = get_eye_landmarks(landmarks, left)
    eye_points = np.array([(p.x, p.y) for p in points], np.int32)

    # Get bounding rectangle around eye
    x, y, w, h = cv2.boundingRect(eye_points)
    eye = frame[y:y+h, x:x+w]

    # Apply grayscale and thresholding
    gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY_INV)

    return threshold_eye, eye_points, x, y

def detect_pupil(thresh_eye):
    """Detects pupil using contours and returns the center coordinates."""
    contours, _ = cv2.findContours(thresh_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cx, cy = x + w // 2, y + h // 2
        return cx, cy
    return None, None

def get_gaze_direction(cx, cy, w, h):
    """Determine gaze direction based on pupil position."""
    if cx < w // 3:
        return "Looking Right"
    elif cx > 2 * w // 3:
        return "Looking Left"
    else:
        return "Looking Center"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)

        # Process both eyes
        for is_left in [True, False]:
            thresh_eye, eye_points, x, y = eye_region(frame, landmarks, is_left)
            cx, cy = detect_pupil(thresh_eye)

            if cx is not None and cy is not None:
                # Draw pupil location
                cv2.circle(frame, (x + cx, y + cy), 3, (0, 255, 0), -1)
                
                # Get gaze direction
                direction = get_gaze_direction(cx, cy, thresh_eye.shape[1], thresh_eye.shape[0])
                eye_side = "Left Eye" if is_left else "Right Eye"
                cv2.putText(frame, f"{eye_side}: {direction}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow("Gaze Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
