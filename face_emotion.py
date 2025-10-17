import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
from deepface import DeepFace 
def detect_face_emotion():
    cap = cv2.VideoCapture(0)
    emotion = "Unknown"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
        except Exception as e:
            emotion = "No face detected"

        cv2.putText(frame, f"Emotion: {emotion}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("MindScope Facial Emotion", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return emotion
