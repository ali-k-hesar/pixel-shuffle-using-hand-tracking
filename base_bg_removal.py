import numpy as np
from ultralytics import YOLO
import cv2


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


def run_webcam(input_webcam=0) -> None:
    """
    Runs the webcam application for image processing and visualization.
    Press 'q' to quit
    """
    cap = cv2.VideoCapture(input_webcam)

    while True:
        ret, frame = cap.read()
        assert ret, 'webcam does not return image!!!'

        frame, _ = bg_removal(frame)

        cv2.imshow('WebCam', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Load YOLOv8 model for segmentation and background removal
    yolov8_model_type = 'm'  # Choose between "n" for nano, "s" for small, "m" for medium, "l" for large and "x" for X-large
    bg_model = YOLO(f"bg_models/yolov8{yolov8_model_type}-seg.pt")

    # Webcam number based on total webcams connected to your computer
    webcam_number = 0

    run_webcam(webcam_number)
    cv2.destroyAllWindows()
