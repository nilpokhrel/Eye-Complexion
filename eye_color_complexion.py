
# compute color complexion

import numpy as np
import image_color_transformation as ict
import eye_color_analytic as eca
import load_image
from typing import List, Union, Dict, Tuple
import PIL
# COLOR SPACE REQUIRED
COLOR_SPACE = ['CMYK', 'RGB', 'LAB', 'HSV', 'Y_CR_CB', 'HLS']

# if set True it displays Intermediate Image processing , Detection, plotting landmarks and plotting ROI in eyes.
DEBUG_IMAGE_PROCESSING = True

class EyeComplexion:
    """Computes Face Detection ,determine eye and iris landmarks, determine ROI, averages all channels color from ROI image blobs.

    Raises:
        ValueError: 'Invalid color space type'; there are only limited color conversion defined.

    Returns:
        _type_: _description_
    """
    @classmethod
    def compute_mean(cls, image_array:Union[np.ndarray,List[int]]):
        """ Averages numeric values in arrays

        Args:
            image_array (Union[np.ndarray,List[int]]): calculates mean for image and list as Input

        Returns:
            float: Averaged image or array floating point values
        """
        if isinstance(image_array, np.ndarray):
            return np.round(np.mean(image_array), 2)
        return np.round(np.mean(np.array(image_array)), 2)

    
    @classmethod
    def get_color_space_means(cls, cropped_image:np.ndarray):
        """Computes averages of each color channel for each colors defined.

        Args:
            cropped_image (np.ndarray): Image within ROI cropped from original face image

        Raises:
            ValueError: Only computes for colors that are globally defined within COLOR_SPACE

        Returns:
            Dict[str,List[float]]: color name key  and list of averages for each channel as value
        """
        # DEFINE EMPTY OUTPUT DICTIONARY DATA STRUCTURE
        color_data_dict = dict()

        for color in COLOR_SPACE:
            if color == 'CMYK':
                # convert nd.array to PIL.Image.Image type
                cropped_pil_image = load_image.array_to_pil_img(cropped_image.copy())
                cmyk_img = ict.ColorTransform.cmyk(cropped_pil_image)
                c,m,y,k = ict.ChannelExtraction.channels(cmyk_img,is_cmyk=True)
                mean_c = cls.compute_mean(c)
                mean_m = cls.compute_mean(m)
                mean_y = cls.compute_mean(y)
                mean_k = cls.compute_mean(k)
                color_data_dict[color] = (mean_c, mean_m, mean_y, mean_k)
            elif color == 'LAB':
                lab_img = ict.ColorTransform.lab(cropped_image)
                l, a, b, _ = ict.ChannelExtraction.channels(lab_img)
                mean_l = cls.compute_mean(l)
                mean_a = cls.compute_mean(a)
                mean_b = cls.compute_mean(b)
                color_data_dict[color] = (mean_l, mean_a, mean_b)
            elif color == 'RGB':
                rgb_img = ict.ColorTransform.rgb(cropped_image)
                r, g, b, _ = ict.ChannelExtraction.channels(rgb_img)
                mean_r = cls.compute_mean(r)
                mean_g = cls.compute_mean(g)
                mean_b = cls.compute_mean(b)
                color_data_dict[color] = (mean_r, mean_g, mean_b)
            elif color == 'HSV':
                hsv_img = ict.ColorTransform.hsv(cropped_image, full=True)
                h, s, v, _ = ict.ChannelExtraction.channels(hsv_img)
                mean_h = cls.compute_mean(h)
                mean_s = cls.compute_mean(s)
                mean_v = cls.compute_mean(v)
                color_data_dict[color] = (mean_h, mean_s, mean_v)
            elif color == 'HLS':
                hls_img = ict.ColorTransform.hls(cropped_image, full=True)
                h, l, s, _ = ict.ChannelExtraction.channels(hls_img)
                mean_h = cls.compute_mean(h)
                mean_l = cls.compute_mean(l)
                mean_s = cls.compute_mean(s)
                color_data_dict[color] = (mean_h, mean_l, mean_s)
            elif color == 'Y_CR_CB':
                y_cr_cb_img = ict.ColorTransform.y_cr_cb(cropped_image)
                y, cr, cb, _ = ict.ChannelExtraction.channels(y_cr_cb_img)
                mean_y = cls.compute_mean(y)
                mean_cr = cls.compute_mean(cr)
                mean_cb = cls.compute_mean(cb)
                color_data_dict[color] = (mean_y, mean_cr, mean_cb)
            else:
                raise ValueError('Invalid color space type')
        return color_data_dict
    
    @classmethod
    def get_eye_color_complexion(cls,face_image:Union[PIL.Image.Image,np.ndarray],show_image_processing=False):
        """Procedures and method call for the Calculation of Eye Color Complexion

        Args:
            face_image (Union[PIL.Image.Image,np.ndarray]): Image from  Android app/ User or Test Face Image for Eye Color Complexion Analytic Color Space Data
            show_image_processing (bool, optional): If True Displays Intermediate Image processing within this method Calls. Defaults to False.

        Returns:
            Union[Tuple[Dict[Union[[str,Dict[Dict[str,Tuple[float]]]],None],np.ndarray],Tuple[None,str]: Dictionary of eye ROI color average and 
            Plotted ROI on Input Face Image
        """
        if not isinstance(face_image,np.ndarray):
            channel_color = ''.join(face_image.getbands())
            if channel_color.upper() != 'RGB':
                face_image = face_image.convert('RGB')
            face_image = load_image.pil_to_numpy(face_image)
        else:
            face_image = load_image.bgr_to_rgb_conversion(face_image)
            
        image_segment_dictionary,roi_masked_face_image = eca.RetrieveMaskedBlobImage.get_cropped_blob_image(face_image,plot_roi_added=show_image_processing)
        # if face with defined threshold for minimum face detection cannot detect face in given image; return None and error message
        if image_segment_dictionary is None:
            return image_segment_dictionary, roi_masked_face_image
        
        average_color_space_for_both_eyes = {}
        for key,image_blobs in image_segment_dictionary.items():
            if image_blobs is not None:
                non_zero_pixel_image = eca.FilterZeroMaskedPixels.filter_masked_zero_pixels(image_blobs)

                average_color = EyeComplexion.get_color_space_means(non_zero_pixel_image)
                average_color_space_for_both_eyes[key] = average_color
            else:
                average_color_space_for_both_eyes[key] = None
        return average_color_space_for_both_eyes, roi_masked_face_image


if __name__ == '__main__':
    pass

