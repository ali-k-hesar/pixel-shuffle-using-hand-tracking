import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import argparse


def jitter_block(input_image, block_size, randomness):
    """
    Applies block-based shuffling to an image.

    Parameters:
        input_image (ndarray): The input image represented as a NumPy array.
        block_size (int): The size of the square block in pixels.
        randomness (int): The randomness value controlling the intensity of jittering.

    Returns:
        ndarray: The shuffled image as a NumPy array.
    """
    height, width, _ = input_image.shape

    # Calculate the number of blocks in each dimension
    num_blocks_y = height // block_size
    num_blocks_x = width // block_size

    # Create a copy of the image to avoid modifying the original
    jittered_image = np.copy(input_image)

    for block_y in range(num_blocks_y):
        for block_x in range(num_blocks_x):
            # Calculate the coordinates of the block's top-left corner
            start_x = block_x * block_size
            start_y = block_y * block_size

            # Calculate random offsets for the block's position
            offset_x = np.random.randint(-randomness, randomness + 1)
            offset_y = np.random.randint(-randomness, randomness + 1)

            # Calculate the new position of the block
            new_x = max(0, min(width - block_size, start_x + offset_x))
            new_y = max(0, min(height - block_size, start_y + offset_y))

            # Extract the block from the original image
            block = input_image[start_y:start_y + block_size, start_x:start_x + block_size, :]

            # Place the block at the new position in the jittered image
            jittered_image[new_y:new_y + block_size, new_x:new_x + block_size, :] = block

    return jittered_image


def pixelate_image(input_image, division_block_size=2, dark_stretch_h=0, dark_stretch_v=0):
    """
    Applies Pixelation on input image.

    Parameters:
        input_image (ndarray): The input image represented as a NumPy array.
        division_block_size (int): The size of the square block in pixels.
        dark_stretch_h (int): The size of dark line in dimension 0 (horizontal).
        dark_stretch_v (int): The size of dark line in dimension 1 (vertical).

    Returns:
        ndarray: The Pixelated image as a NumPy array.
    """
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    input_image = np.stack([np.zeros_like(input_image), input_image, np.zeros_like(input_image)], axis=-1)

    height, width, _ = input_image.shape

    input_image[[i + j for i in range(0, height, division_block_size) for j in range(dark_stretch_h)], :, :] *= 0
    input_image[:, [i + j for i in range(0, width, division_block_size) for j in range(dark_stretch_v)], :] *= 0
    input_image = cv2.GaussianBlur(input_image, (5, 5), 0)

    return input_image


def bg_removal(input_image):
    """
    Performs background removal on the input image.

    Parameters:
        input_image (ndarray): The input image represented as a NumPy array.

    Returns:
        tuple: A tuple containing the modified input image with background removed and the output mask.
    """
    results = bg_model.predict(input_image.copy())[0]
    output_mask = np.zeros_like(input_image, dtype='float32')
    if results.masks:
        for i in range(len(results.boxes.boxes)):
            if int(results.boxes.boxes[i, -1].item()) == 0:
                output_mask = results.masks.masks.detach().cpu()[i][..., None]
                input_image = np.where(output_mask, input_image, 0)
    return input_image, output_mask


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
        ret, raw_frame = cap.read()
        assert ret, 'webcam does not return image!!!'

        frame, _ = bg_removal(raw_frame)
        _, raw_dis = hand_tracking(raw_frame)
        dis = int(raw_dis * 20) if raw_dis is not None else dis
        dis = min(dis, 5)

        frame = pixelate_image(frame, division_block_size=8, dark_stretch_h=2, dark_stretch_v=4)
        frame = jitter_block(frame, 4, [0, 1, 2, 6, 12, 28][dis])

        cv2.putText(frame, f"Entropy: {'|' * (dis + 1)}", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.imshow('WebCam', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='m',
                        help='Choose between "n" for nano, "s" for small, "m" for medium, "l" for large and "x" for X-large')
    parser.add_argument('--webcam-number', type=int, default=0,
                        help='Specific webcam number based on total webcams connected to your computer')
    opt = parser.parse_args()

    yolov8_model_type = opt.model_type
    print(f"model type: {yolov8_model_type}")
    webcam_number = opt.webcam_number

    # Load YOLOv8 model for segmentation and background removal
    bg_model = YOLO(f"bg_models/yolov8{yolov8_model_type}-seg.pt")

    # Load the MediaPipe Hands model
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    run_webcam(webcam_number)
    cv2.destroyAllWindows()

