import cv2
import dlib
import numpy as np
import numpy.typing as npt
from pynput.mouse import Button, Controller
from enum import Enum
import json

class mouse_action(Enum):
    MOVE_UP = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    RIGHT_CLICK_DOWN = 5
    LEFT_CLICK_DOWN = 6
    RIGHT_CLICK_UP = 7
    LEFT_CLICK_UP = 8

class mouse_direction(Enum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4
    CENTER = 5


# Constants
SCALE = 0 
CLICK_THRESHOLD = 0
EAR_THRESHOLD = 0
UP_SCALAR = 0
DOWN_SCALAR = 0

# Mouse state
mouse = Controller()
MOUSE_LEFT_DOWN = False
MOUSE_RIGHT_DOWN = False
COUNTER_LEFT = 0
COUNTER_RIGHT = 0

def move_mouse(action: mouse_action) -> None:
    """
    Move the mouse cursor according to the given action.

    This function moves the mouse cursor relative to its current position by the
    given scale. The scale is a constant defined as 10 pixels.

    Args:
        action: The action to perform on the mouse. This can be one of the
            following values:
                - mouse_action.MOVE_UP
                - mouse_action.MOVE_DOWN
                - mouse_action.MOVE_RIGHT
                - mouse_action.MOVE_LEFT
    """
    match action:
        case mouse_action.MOVE_UP:
            mouse.move(0, -SCALE)
        case mouse_action.MOVE_DOWN:
            mouse.move(0, SCALE)
        case mouse_action.MOVE_RIGHT:
            mouse.move(SCALE, 0)
        case mouse_action.MOVE_LEFT:
            mouse.move(-SCALE, 0)
         
def mouse_click(action: mouse_action) -> None:
    """
    Simulate mouse click actions based on the specified action.

    This function simulates mouse click actions by pressing or releasing the mouse
    buttons. It supports both left and right mouse button actions.

    Args:
        action: The action to perform on the mouse. This can be one of the
            following values:
                - mouse_action.RIGHT_CLICK_DOWN: Press the right mouse button.
                - mouse_action.LEFT_CLICK_DOWN: Press the left mouse button.
                - mouse_action.RIGHT_CLICK_UP: Release the right mouse button.
                - mouse_action.LEFT_CLICK_UP: Release the left mouse button.

    Returns:
        None
    """

    match action:
        case mouse_action.RIGHT_CLICK_DOWN:
            mouse.press(Button.right)
        case mouse_action.LEFT_CLICK_DOWN:
            mouse.press(Button.left)
        case mouse_action.RIGHT_CLICK_UP:
            mouse.release(Button.right)
        case mouse_action.LEFT_CLICK_UP:
            mouse.release(Button.left)
    
def get_eye_landmarks(landmarks: dlib.full_object_detection, left: bool=True) -> np.ndarray:
    """
    Extracts the landmarks for either the left or right eye from the given 68-landmark dlib shape.

    Args:
        landmarks: The dlib shape from which to extract the eye landmarks.
        left: Whether to extract the left eye (True) or right eye (False). Defaults to True.

    Returns:
        A numpy array of shape (6, 2) containing the x, y coordinates of the 6 eye landmarks.
    """
    points = [landmarks.part(i) for i in range(36, 42)] if left else [landmarks.part(i) for i in range(42, 48)]
    return np.array([(p.x, p.y) for p in points], np.int32)

def eye_region(frame: npt.NDArray, landmarks: dlib.full_object_detection, left: bool=True) -> tuple[npt.NDArray, npt.NDArray, int, int]:
    """
    Extracts the region of interest (ROI) for the specified eye from the given frame and 68-landmark dlib shape.

    The ROI is a box centered on the eye which is then used to detect the eye gaze. The returned values are:
        - thresh_eye: The thresholded image of the eye ROI, used to detect the pupil.
        - eye_points: The 6-landmark coordinates of the eye, used to detect the eye aspect ratio.
        - x, y: The coordinates of the top-left corner of the eye ROI in the original frame.

    Args:
        frame: The input frame from which to extract the eye ROI.
        landmarks: The dlib shape from which to extract the eye landmarks.
        left: Whether to extract the left eye (True) or right eye (False). Defaults to True.

    Returns:
        A tuple of (thresh_eye, eye_points, x, y) as described above.
    """
    
    eye_points = get_eye_landmarks(landmarks, left)

    # Center of eye
    center_x = np.mean(eye_points[:, 0]).astype(int)
    center_y = np.mean(eye_points[:, 1]).astype(int)

    # Fixed box size (can tweak for your face/camera)
    box_w, box_h = 60, 40
    x = max(center_x - (box_w >> 1), 0)
    y = max(center_y - (box_h >> 1), 0)

    eye = frame[y:y+box_h, x:x+box_w]

    gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    _, thresh_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY_INV)

    return thresh_eye, eye_points, x, y


def detect_pupil(thresh_eye: np.ndarray) -> tuple[int, int, int]:
    """
    Detects the pupil in a given thresholded eye image.

    The function iterates through the contours in the image and selects the largest contour that is larger than a certain size (100 pixels).
    The function then calculates the center of mass of the contour and returns the x and y coordinates of the center of mass and the contour itself.

    Args:
        thresh_eye: A thresholded image of the eye, used to detect the pupil.

    Returns:
        A tuple of (cx, cy, contour), where cx and cy are the coordinates of the center of mass of the pupil and contour is the contour of the pupil.
    """
    contours, _ = cv2.findContours(thresh_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy, contour

    return None, None, None

def get_gaze_direction(cx: int, cy: int, shape: tuple[int, int]) -> mouse_direction:
    """
    Determines the direction of the gaze based on the pupil location.

    The function takes the x and y coordinates of the pupil center and the shape of the frame as input.
    It returns the direction of the gaze as a mouse_direction enum.

    The direction is determined as follows:
        - Horizontal: If the pupil is on the left third of the frame, the direction is LEFT.
                      If the pupil is on the right third of the frame, the direction is RIGHT.
                      Otherwise, the direction is CENTER.
        - Vertical:   If the pupil is on the top quarter of the frame, the direction is UP.
                      If the pupil is on the bottom half of the frame, the direction is DOWN.
                      Otherwise, the direction is CENTER.

    If the horizontal direction is CENTER, the function returns the vertical direction.
    Otherwise, it returns the horizontal direction.

    Args:
        cx: The x coordinate of the pupil center.
        cy: The y coordinate of the pupil center.
        shape: The shape of the frame as a tuple (height, width).

    Returns:
        A mouse_direction enum representing the direction of the gaze.
    """
    width, height = shape[1], shape[0]
    
    # Horizontal
    horizontal = (
        mouse_direction.RIGHT if cx * 3 < width else
        mouse_direction.LEFT if cx * 3 > (width << 1) else
        mouse_direction.CENTER
    )

    # Vertical (tune as needed)
    vertical = (
        mouse_direction.UP if cy < height * UP_SCALAR else
        mouse_direction.DOWN if cy > height * DOWN_SCALAR else
        mouse_direction.CENTER
    )


    return vertical if horizontal == mouse_direction.CENTER else horizontal

def draw_pupils(frame: np.ndarray, landmarks: dlib.full_object_detection) -> None:  
    """
    Draws the detected pupils on the given frame.

    This function processes both the left and right eyes to locate and draw the pupils.
    It extracts the eye region, detects the pupil within the thresholded image, and then
    calls the draw_pupil function to render the pupil on the frame.

    Args:
        frame: The input frame on which to draw the pupils.
        landmarks: The dlib shape containing facial landmarks for the frame.

    Returns:
        None
    """

    for is_left in [True, False]:
        thresh_eye, _, x, y = eye_region(frame, landmarks, is_left)
        pupil_position = detect_pupil(thresh_eye)

        draw_pupil(frame, thresh_eye.shape, pupil_position, x, y, is_left)


def draw_pupil(frame: np.ndarray, shape: tuple[int, int], pupil_position: tuple[int, int, int], x: int, y: int, left=True)-> None:
    """
    Draws the detected pupil on the given frame and moves the mouse cursor accordingly.

    This function takes the given frame, the shape of the eye region, the detected pupil position,
    and the x, y coordinates of the top-left corner of the eye region in the original frame.
    It then draws a circle at the pupil position and determines the gaze direction based on the
    pupil position. Finally, it moves the mouse cursor according to the gaze direction and prints
    the gaze direction to the console.

    Args:
        frame: The input frame on which to draw the pupil.
        shape: The shape of the eye region.
        pupil_position: The detected pupil position.
        x: The x coordinate of the top-left corner of the eye region in the original frame.
        y: The y coordinate of the top-left corner of the eye region in the original frame.
        left: Whether the eye is the left eye (True) or right eye (False). Defaults to True.

    Returns:
        None
    """
    cx, cy, _ = pupil_position
    if cx is None:
        return
    if cy is None:
        return

    # Draw pupil location
    cv2.circle(frame, (x + cx, y + cy), 3, (0, 255, 0), -1)

    # Get gaze direction
    direction = get_gaze_direction(cx, cy, shape)
    match direction:
        case mouse_direction.UP:
            print("↑")
            move_mouse(mouse_action.MOVE_UP)
        case mouse_direction.DOWN:
            print("↓")
            move_mouse(mouse_action.MOVE_DOWN)
        case mouse_direction.RIGHT:
            print("→")
            move_mouse(mouse_action.MOVE_RIGHT)
        case mouse_direction.LEFT:
            print("←")
            move_mouse(mouse_action.MOVE_LEFT)
        case mouse_direction.CENTER:
            print("👁️👁️")

    eye_side = "Left Eye" if left else "Right Eye"
    cv2.putText(frame, f"{eye_side}: {direction.name}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

def eye_aspect_ratio(eye: np.ndarray) -> float:
    a = np.linalg.norm(eye[1] - eye[5])
    b = np.linalg.norm(eye[2] - eye[4])
    c = np.linalg.norm(eye[0] - eye[3])
    return (a + b) / (2.0 * c)

def click_mouse(ear_left: float, ear_right: float) -> None:
    """
    Simulates a mouse click action based on the given EAR values.

    The function takes two EAR values, one for the left eye and one for the right eye.
    If the EAR value for either eye is below the threshold, it increments the corresponding counter.
    If the counter exceeds the click threshold, it simulates a mouse click action and resets the counter.

    Args:
        ear_left: The EAR value for the left eye.
        ear_right: The EAR value for the right eye.
    """
    global MOUSE_LEFT_DOWN, MOUSE_RIGHT_DOWN, COUNTER_LEFT, COUNTER_RIGHT

    if ear_left < EAR_THRESHOLD:
        COUNTER_LEFT += 1
    else:
        if COUNTER_LEFT >= CLICK_THRESHOLD:
            # mouse_click(mouse_action.leftClickUp if MOUSE_LEFT_DOWN else mouse_action.leftClickDown)
            print("left click")
            MOUSE_LEFT_DOWN = not MOUSE_LEFT_DOWN
        COUNTER_LEFT = 0

    if ear_right < EAR_THRESHOLD:
        COUNTER_RIGHT += 1
    else:
        if COUNTER_RIGHT >= CLICK_THRESHOLD:
            # mouse_click(mouse_action.rightClickUp if MOUSE_RIGHT_DOWN else mouse_action.rightClickDown)
            print("right click")
            MOUSE_RIGHT_DOWN = not MOUSE_RIGHT_DOWN
        COUNTER_RIGHT = 0

def load_config() -> None:
    global SCALE, CLICK_THRESHOLD, EAR_THRESHOLD, UP_SCALAR, DOWN_SCALAR
    with open("config.json", "r") as f:
        config = json.load(f)
        
    SCALE = config["SCALE"]
    CLICK_THRESHOLD = config["CLICK_THRESHOLD"]
    EAR_THRESHOLD = config["EAR_THRESHOLD"]
    UP_SCALAR = config["UP_SCALAR"]
    DOWN_SCALAR = config["DOWN_SCALAR"]


def main() -> None:
    """
    Main function to perform real-time eye tracking and mouse control.

    This function captures video from the default camera, detects faces and 
    facial landmarks in each frame, and tracks eye movement to control the 
    mouse cursor. It utilizes dlib's frontal face detector and shape predictor 
    to obtain facial landmarks. The eye aspect ratio (EAR) is calculated for 
    both eyes to detect blinks and simulate mouse click actions. It draws 
    pupil positions on the frame and controls the mouse based on gaze 
    direction. The video feed displays in a window until 'q' is pressed.

    Returns:
        None
    """
    load_config()

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
            left_eye = get_eye_landmarks(landmarks, True)
            right_eye = get_eye_landmarks(landmarks, False)

            # Camera is mirrored so switch left and right
            ear_right, ear_left = eye_aspect_ratio(left_eye), eye_aspect_ratio(right_eye)
            draw_pupils(frame, landmarks)
            click_mouse(ear_left, ear_right)
            
        cv2.imshow("Gaze Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
