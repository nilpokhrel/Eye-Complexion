# load bs64 string image to PIL/cv2 image mode
from PIL import Image
import cv2
import numpy as np


def pil_to_numpy(pil_image):
    """Type conversion from PIL.Image.Image to numpy array

    Args:
        pil_image (PIL.Image.Image): Input image for Conversion

    Returns:
        np.ndarray: Output Image after Conversion
    """
    numpy_array_img = np.asarray(pil_image)
    return numpy_array_img


def array_to_pil_img(array_img):
    """Type conversion from numpy ndarray to PIL.Image.Image

    Args:
        array_img (np.ndarray): Input Image for Conversion

    Returns:
        PIL.Image.Image: Output Image after Conversion
    """
    arr_to_pil_img = Image.fromarray(array_img,'RGB')
    return arr_to_pil_img

def bgr_to_rgb_conversion(brg_image):
    """Image Color conversion

    Args:
        brg_image (np.ndarray): Input BGR ndarray image for conversion

    Returns:
        np.ndarray: Output RGB ndarray image after conversion
    """
    rgb_image =  cv2.cvtColor(brg_image, cv2.COLOR_BGR2RGB)
    return rgb_image

if __name__ == '__main__':
    pass

