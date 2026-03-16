import os
import joblib
import mediapipe as mp

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MP_MODEL_PATH = os.path.join(BASE_DIR / 'models' / 'gesture_recognizer.task')
CUSTOM_MODEL_PATH = os.path.join(BASE_DIR / 'models' / 'gesture_model.joblib')
ENCODER_PATH = os.path.join(BASE_DIR / 'models'/ 'label_encoder.joblib')

def load_custom_models():
    if not all(os.path.exists(path) for path in [CUSTOM_MODEL_PATH, ENCODER_PATH]):
        raise FileNotFoundError("One or more model files are missing.")
    
    clf = joblib.load(CUSTOM_MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    return clf, label_encoder

def get_mediapipe_options():
    if not os.path.exists(MP_MODEL_PATH):
        raise FileNotFoundError(f"Mediapipe model file not found at {MP_MODEL_PATH}")
    
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MP_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return options