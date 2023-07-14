import mediapipe as mp
import cv2
import numpy as np


def hand_tracking(input_image):
    """
    Performs hand tracking on the input image.

    Parameters:
        input_image (ndarray): The input image represented as a NumPy array.

    Returns:
        tuple: A tuple containing the modified input image with landmarks drawn and the calculated distance.
    """
    # Detect hands in the image
    results = hands.process(input_image[:, :, ::-1])

    # Check if hands were detected
    distance = None
    if results.multi_hand_landmarks:
        # Loop through each detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the image
            mp_drawing.draw_landmarks(
                input_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the landmarks of the thumb tip and index fingertip
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Calculate the distance between the thumb tip and index fingertip
            distance = np.sqrt((thumb_tip.x - index_finger_tip.x) ** 2 + (thumb_tip.y - index_finger_tip.y) ** 2)

            # Convert the coordinates to image pixel values
            height, width, _ = input_image.shape
            thumb_x, thumb_y = int(thumb_tip.x * width), int(thumb_tip.y * height)
            index_x, index_y = int(index_finger_tip.x * width), int(index_finger_tip.y * height)

            # Draw a line from thumb_tip to index_finger_tip
            cv2.line(input_image, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 3)

    # Return the modified image and distance
    return input_image, distance


def run_webcam(input_webcam=0) -> None:
    """
    Runs the webcam application for image processing and visualization.
    Press 'q' to quit
    """
    cap = cv2.VideoCapture(input_webcam)
    font = cv2.FONT_HERSHEY_COMPLEX

    dis = 0
    while True:
        ret, frame = cap.read()
        assert ret, 'webcam does not return image!!!'

        frame, raw_dis = hand_tracking(frame)
        dis = int(raw_dis * 20) if raw_dis is not None else dis
        dis = min(dis, 5)

        cv2.putText(frame, f"Finger Distance: {'|' * (dis + 1)}", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.imshow('WebCam', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Load the MediaPipe Hands model
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Webcam number based on total webcams connected to your computer
    webcam_number = 0

    run_webcam(webcam_number)
    cv2.destroyAllWindows()
