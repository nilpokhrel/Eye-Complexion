# color transformation
import PIL
import cv2
import numpy as np

class ColorTransform:
    """Image Color Conversion

    Raises:
        TypeError: 'Image instance must be of numpy ndarray'
        TypeError: 'Only PIL image instance can be applied for CMYK conversion'

    Returns:
        Union(np.ndarray,PIL.Image.Image): Color Transformed Image
    """
    @classmethod
    def hsv(cls, img, full=True):
        if not isinstance(img, np.ndarray):
            raise TypeError('Image instance must be of numpy ndarray')
        if full:
            return cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    @classmethod
    def lab(cls, img):
        if not isinstance(img, np.ndarray):
            raise TypeError('Image instance must be of numpy ndarray')
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        return lab_img

    @classmethod
    def rgb(cls, img):
        if not isinstance(img, np.ndarray):
            raise TypeError('Image instance must be of numpy ndarray')
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb_img

    @classmethod
    def upper_lab(cls, img):
        """
        This function converts image of BGR to LAB color space.
        :param img:
        :return: LAB color space image
        """
        if not isinstance(img, np.ndarray):
            raise TypeError('Image instance must be of numpy ndarray')
        upper_lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        return upper_lab_img

    @classmethod
    def cmyk(cls, img, conversion_mode='CMYK'):

        if isinstance(img, PIL.Image.Image):
            converted_img = img.convert(mode=conversion_mode)
            return converted_img
        else:
            raise TypeError('Only PIL image instance can be applied for CMYK conversion')

    @classmethod
    def hls(cls, img, full=False):
        if not isinstance(img, np.ndarray):
            raise TypeError('Image instance must be of numpy ndarray')
        if full:
            hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS_FULL)
        else:
            hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        return hls_img

    @classmethod
    def y_cr_cb(cls, img):
        if not isinstance(img, np.ndarray):
            raise TypeError('Image instance must be of numpy ndarray')
        y_cr_cb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        return y_cr_cb_img

    @classmethod
    def luv(cls, img):
        if not isinstance(img, np.ndarray):
            raise TypeError('Image instance must be of numpy ndarray')
        luv_img = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)
        return luv_img

    @classmethod
    def upper_luv(cls, img):
        if not isinstance(img, np.ndarray):
            raise TypeError('Image instance must be of numpy ndarray')
        upper_luv_img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        return upper_luv_img

    @classmethod
    def yuv(cls, img):
        if not isinstance(img, np.ndarray):
            raise TypeError('Image instance must be of numpy ndarray')
        yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        return yuv_img


class ChannelExtraction:
    """Color Channel Extraction 

    Returns:
        array: Each channel array is returned packed into tuple. There are 3 and 4 channel returned by methods
    """
    @classmethod
    def channels(cls, image_to_channels, is_cmyk=False):
        """Extract Channel to np.ndarray and PIL.Image.Image type Images

        Args:
            image_to_channels (Union(np.ndarray,PIL.Image.Image)): Input Image to extract channels arrays of Image 
            is_cmyk (bool, optional): If is_cmyk is set to True, it extracts 4 channels array and if set to False it returns 3 channels array. Defaults to False.

        Returns:
            arrays: Returns 3 channel arrays if image is of type np.ndarray, else it returns 4 channel array on Image type PIL.Image.Image
        """
        # applicable to all other channels of numpy array image
        if isinstance(image_to_channels, np.ndarray):
            if not is_cmyk:
                channel1 = image_to_channels[:, :, 0]
                channel2 = image_to_channels[:, :, 1]
                channel3 = image_to_channels[:, :, 2]
                return channel1, channel2, channel3, None
            else:
                c_array, m_array, y_array, k_array = cv2.split(image_to_channels)
                # y_array, m_array, c_array, k_array = cv2.split(image_to_channels)
                return c_array, m_array, y_array, k_array
        else:
            # applicable to cmyk channels of pil image only
            # split all channels img
            c_arr, m_arr, y_arr, k_arr = image_to_channels.split()
            return c_arr, m_arr, y_arr, k_arr


if __name__ == '__main__':
    pass



