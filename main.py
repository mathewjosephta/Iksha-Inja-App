import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Set up canvas
canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255  # white canvas
drawing = False
prev_point = None

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose_tip = face_landmarks.landmark[4]
            h, w, _ = frame.shape
            x, y = int(nose_tip.x * w), int(nose_tip.y * h)

            if drawing and prev_point is not None:
                cv2.line(canvas, prev_point, (x, y), (0, 0, 255), 2)

            prev_point = (x, y)

    cv2.imshow("Eksha Inna Varakum - Drawing Canvas", canvas)
    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("d"):  # Toggle drawing
        drawing = not drawing
    elif key == ord("c"):  # Clear canvas
        canvas[:] = 255
    elif key == ord("s"):  # Save image
        cv2.imwrite("eksha_inna_drawing.png", canvas)
        print("Drawing saved!")

cap.release()
cv2.destroyAllWindows()
