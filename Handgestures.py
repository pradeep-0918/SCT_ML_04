import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import pyttsx3
import os

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Text-to-Speech engine
engine = pyttsx3.init()


def speak(text):
    engine.say(text)
    engine.runAndWait()


# Create dataset directory
DATASET_DIR = "Sign-language-digits-dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

gesture_controls = {
    "Thumbs Up": "Increase Volume",
    "Thumbs Down": "Decrease Volume",
    "Open Palm": "Pause/Play",
    "Fist": "Mute/Unmute",
    "Victory": "Scroll Up",
    "Pointing": "Cursor Move",
    "Rock Sign": "Scroll Down",
    "Exit": "Exit Program"
}


def detect_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]

    if thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y:
        return "Thumbs Up"
    elif thumb_tip.y > index_tip.y and thumb_tip.y > middle_tip.y:
        return "Thumbs Down"
    elif all([index_tip.y < hand_landmarks.landmark[i].y for i in [6, 10, 14, 18]]):
        return "Open Palm"
    elif all([index_tip.y > hand_landmarks.landmark[i].y for i in [6, 10, 14, 18]]):
        return "Fist"
    elif index_tip.y < middle_tip.y and ring_tip.y > middle_tip.y and pinky_tip.y > ring_tip.y:
        return "Victory"
    elif index_tip.y < middle_tip.y and all([middle_tip.y > ring_tip.y, ring_tip.y > pinky_tip.y]):
        return "Pointing"
    elif all([thumb_tip.y < index_tip.y, pinky_tip.y < index_tip.y]):
        return "Rock Sign"
    return "Unknown Gesture"


cap = cv2.VideoCapture(0)
cap.set(cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

gesture_history = []
fist_start_time = None

screen_width, screen_height = pyautogui.size()
prev_x, prev_y = 0, 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(hand_landmarks)
            control_action = gesture_controls.get(gesture, "No Action")

            if gesture not in gesture_history:
                gesture_history.append(gesture)
                if len(gesture_history) > 3:
                    gesture_history.pop(0)
                speak(control_action)

            if gesture == "Thumbs Up":
                pyautogui.press("volumeup")
            elif gesture == "Thumbs Down":
                pyautogui.press("volumedown")
            elif gesture == "Open Palm":
                pyautogui.press("playpause")
            elif gesture == "Fist":
                if fist_start_time is None:
                    fist_start_time = time.time()
                elif time.time() - fist_start_time > 3:
                    speak("Exiting program")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
                pyautogui.press("volumemute")
            else:
                fist_start_time = None  # Reset timer

            if gesture == "Victory":
                pyautogui.scroll(10)
            elif gesture == "Rock Sign":
                pyautogui.scroll(-10)
            elif gesture == "Pointing":
                index_finger = hand_landmarks.landmark[8]
                x, y = int(index_finger.x * screen_width), int(index_finger.y * screen_height)
                smooth_x = int((prev_x + x) / 2)
                smooth_y = int((prev_y + y) / 2)
                pyautogui.moveTo(smooth_x, smooth_y, duration=0.1)
                prev_x, prev_y = smooth_x, smooth_y

            # Save dataset images
            gesture_path = os.path.join(DATASET_DIR, gesture)
            os.makedirs(gesture_path, exist_ok=True)
            img_name = os.path.join(gesture_path, f"{int(time.time())}.jpg")
            cv2.imwrite(img_name, frame)

            cv2.putText(frame, f"Gesture: {gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Action: {control_action}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.namedWindow("Hand Gesture Control", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Hand Gesture Control", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Hand Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()