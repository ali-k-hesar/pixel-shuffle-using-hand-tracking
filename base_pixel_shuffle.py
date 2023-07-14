import numpy as np


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
