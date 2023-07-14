import numpy as np
import cv2


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


def run_webcam(input_webcam) -> None:
    """
    Runs the webcam application for image processing and visualization.
    Press 'q' to quit
    """
    cap = cv2.VideoCapture(input_webcam)

    while True:
        ret, frame = cap.read()
        assert ret, 'webcam does not return image!!!'

        frame = pixelate_image(frame, division_block_size=block_size,
                               dark_stretch_h=horizontal_line, dark_stretch_v=vertical_line)

        cv2.imshow('WebCam', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    block_size, horizontal_line, vertical_line = 8, 2, 4
    # Webcam number based on total webcams connected to your computer
    webcam_number = 0

    run_webcam(webcam_number)
    cv2.destroyAllWindows()
