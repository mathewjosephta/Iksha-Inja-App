import cv2
import mediapipe as mp
import numpy as np

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.8,
                       min_tracking_confidence=0.8)

face_mesh = mp_face.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
drawing = False
prev_point = None
canvas = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb)
    face_results = face_mesh.process(rgb)

    gesture_msg = ""

    # Improved hand detection
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        # Count Left & Right hands separately
        hand_labels = [hand.classification[0].label for hand in hand_results.multi_handedness]

        if "Left" in hand_labels and "Right" in hand_labels:
            cv2.putText(frame, "👋 Two Hands Detected - Exiting...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Eksha Inna Varakum 🎨", np.hstack((frame, canvas)))
            cv2.waitKey(1500)
            break

        # Process first detected hand
        hand_landmarks = hand_results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Finger count logic
        tips_ids = [8, 12, 16, 20]
        fingers_up = 0
        for idx in tips_ids:
            if hand_landmarks.landmark[idx].y < hand_landmarks.landmark[idx - 2].y:
                fingers_up += 1

        if fingers_up >= 3:
            drawing = True
            gesture_msg = "✍️ Drawing Active"
        else:
            drawing = False
            gesture_msg = "⏸ Paused"
    else:
        drawing = False
        gesture_msg = "🖐 No Hand Detected"

    # Nose tracking and drawing
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            nose = face_landmarks.landmark[1]
            x, y = int(nose.x * w), int(nose.y * h)

            if drawing:
                if prev_point is not None:
                    cv2.line(canvas, prev_point, (x, y), (255, 0, 255), 4)
                prev_point = (x, y)
            else:
                prev_point = None

            cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)

    # Status message
    if gesture_msg:
        cv2.putText(frame, gesture_msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 255), 2)

    # Show combined frame
    combined = np.hstack((frame, canvas))
    cv2.imshow("Eksha Inna Varakum 🎨", combined)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
