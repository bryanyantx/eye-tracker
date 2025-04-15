import cv2
import dlib
import numpy as np
import numpy.typing as npt
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

SCALE = 10


def move_mouse(action: mouse_action) -> None:
    match action:
        case mouse_action.moveUp:
            mouse.move(0, SCALE)
        case mouse_action.moveDown:
            mouse.move(0, -SCALE)
        case mouse_action.moveRight:
            mouse.move(SCALE, 0)
        case mouse_action.moveLeft:
            mouse.move(-SCALE, 0)
        case _:
            pass
         
def mouse_click(action: mouse_action) -> None:
    match action:
        case mouse_action.rightClickDown:
            mouse.press(Button.right)
        case mouse_action.leftClickDown:
            mouse.press(Button.left)
        case mouse_action.rightClickUp:
            mouse.release(Button.right)
        case mouse_action.leftClickUp:
            mouse.release(Button.left)
        case _:
            pass
    
def get_eye_landmarks(landmarks: dlib.full_object_detection, left: bool=True) -> list[dlib.point]:
    """Extracts eye landmarks for left or right eye."""
    if left:
        return [landmarks.part(i) for i in range(36, 42)]
    else:
        return [landmarks.part(i) for i in range(42, 48)]


def eye_region(frame: npt.NDArray, landmarks: dlib.full_object_detection, left: bool=True) -> tuple[npt.NDArray, npt.NDArray, int, int]:
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


def detect_pupil(thresh_eye: np.ndarray) -> tuple[int, int]:
    """Detects pupil using contours and returns the center coordinates."""
    contours, _ = cv2.findContours(
        thresh_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        return x + w // 2, y + h // 2
    return None, None


def get_gaze_direction(cx: int, cy: int, shape: tuple[int, int]) -> str:
    """Determine gaze direction based on pupil position."""
    #print( cx, cy, shape)
    if cx < shape[1] // 3:
        print("looking right")
        return "Looking Right"
    elif cx > 2 * shape[1] // 3:
        print("looking left")
        return "Looking Left"
    elif shape[0] > 15 & cy > 9:
        print("looking up")
        return "Looking Center"
        
    elif  (shape[0] < 14) & (cy < 5):
        print("looking down")
        return "Looking Center"
        
    else:
        return "Looking Center"


def draw_pupil(frame: np.ndarray, shape: tuple[int, int], pupil_position: tuple[int, int], x: int, y: int, left=True):
    cx, cy = pupil_position
    if cx is None:
        return
    if cy is None:
        return

    # Draw pupil location
    cv2.circle(frame, (x + cx, y + cy), 3, (0, 255, 0), -1)

    # Get gaze direction
    direction = get_gaze_direction(cx, cy, shape)
    eye_side = "Left Eye" if left else "Right Eye"
    cv2.putText(frame, f"{eye_side}: {direction}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


def main() -> None:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    cap = cv2.VideoCapture(0)

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
                thresh_eye, _, x, y = eye_region(
                    frame, landmarks, is_left)
                pupil_position = detect_pupil(thresh_eye)

                draw_pupil(frame, thresh_eye.shape, pupil_position, x, y, is_left)

        cv2.imshow("Gaze Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
