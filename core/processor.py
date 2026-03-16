import cv2
import mediapipe as mp
import numpy as np
from models import get_mediapipe_options, load_custom_models

class GestureProcessor:
    def __init__(self):
        self.clf, self.label_encoder = load_custom_models()
        self.options = get_mediapipe_options()

        self.recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(self.options)

        self.mp_hands = mp.tasks.vision.HandLandmarksConnections
        self.mp_drawing = mp.tasks.vision.drawing_utils
        self.mp_drawing_styles = mp.tasks.vision.drawing_styles

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

        recognition_result = self.recognizer.recognize_for_video(mp_image, timestamp_ms)

        if recognition_result.hand_landmarks:
            for i, hand_landmarks in enumerate(recognition_result.hand_landmarks):
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            
                hand_label = recognition_result.gestures[i][0].category_name
                handness_val = 0 if hand_label == 'Left' else 1

                landmarks_array = [handness_val]
                for landmark in hand_landmarks:
                    landmarks_array.extend([landmark.x, landmark.y, landmark.z])
                
                features = np.array(landmarks_array).reshape(1, -1)

                prediction_idx = self.clf.predict(features)[0]
                prediction_prob = np.max(self.clf.predict_proba(features))
                gesture_name = self.label_encoder.inverse_transform([prediction_idx])[0]

                color = (0, 255, 0)
                display_text = f"Custom {hand_label}: {gesture_name} ({prediction_prob:.2f})"
                cv2.putText(frame, display_text, (20, 50 + (i * 40)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
        return frame
    
    def close(self):
        if self.recognizer:
            self.recognizer.close()
        
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):      
        self.close()