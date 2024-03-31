import numpy as np
from scipy.signal import convolve2d
from PIL import Image

def depthwise_convolution(image, kernel):
    """Performs depthwise convolution on an image."""
    if len(image.shape) == 3:
        channels = [convolve2d(image[:, :, i], kernel, mode='same', boundary='wrap') for i in range(image.shape[2])]
        return np.stack(channels, axis=2)
    else:
        return convolve2d(image, kernel, mode='same', boundary='wrap')

def pointwise_convolution(image, pointwise_filter):
    """Performs pointwise convolution on an image."""
    if len(image.shape) == 3 and len(pointwise_filter.shape) == 2:
        # Ensure the pointwise filter is in the correct shape for broadcasting
        pointwise_filter = pointwise_filter[:, :, np.newaxis]
        return np.sum(image * pointwise_filter, axis=2)
    else:
        raise ValueError("Image and filter dimensions do not match for pointwise convolution.")

def load_image(image_path):
    """Loads an image and converts it to a numpy array."""
    with Image.open(image_path) as img:
        return np.asarray(img)

def save_image(array, image_path):
    """Saves a numpy array as an image."""
    img = Image.fromarray(np.uint8(array))
    img.save(image_path)

def perform_convolution(image_path, kernel, pointwise_filter=None, mode='depthwise'):
    image = load_image(image_path)
    
    if mode == 'depthwise':
        convoluted_image = depthwise_convolution(image, kernel)
    elif mode == 'pointwise':
        if pointwise_filter is None:
            raise ValueError("Pointwise filter must be provided for pointwise convolution.")
        convoluted_image = pointwise_convolution(image, pointwise_filter)
    else:
        raise ValueError("Unsupported mode. Choose either 'depthwise' or 'pointwise'.")

    return convoluted_image

# Example usage
kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  # Example kernel for depthwise convolution
pointwise_filter = np.array([[1, 0, 1]])  # Example filter for pointwise convolution (to be used on depthwise convoluted image)

# Load an image (replace 'path_to_your_image' with your image path)
image_path = 'path_to_your_image'

# Perform depthwise convolution
depthwise_image = perform_convolution(image_path, kernel, mode='depthwise')

# Perform pointwise convolution on the result of depthwise convolution
pointwise_image = perform_convolution(depthwise_image, None, pointwise_filter=pointwise_filter, mode='pointwise')

# Save the convoluted images
save_image(depthwise_image, 'depthwise_convoluted_image.png')
save_image(pointwise_image, 'pointwise_convoluted_image.png')
