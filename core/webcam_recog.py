import cv2
from processor import GestureProcessor

def main():

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("\nWebcam opened successfully. Press 'q' to exit.")

    with GestureProcessor() as processor:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Error: Failed to read frame from webcam.")
                break

            processed_frame = processor.process_frame(frame)

            cv2.imshow('Gesture Recognition', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()