

import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp 
import cv2 
from typing import List
import eye_roi_plot as plot_eye
mp_face_mesh = mp.solutions.face_mesh

# face detection threshold
MINIMUM_DETECTION_THRESHOLD = 0.6  # 0.5, 0.6. 0.7

LEFT_IRIS_INDICES = [474, 475, 476, 477]
RIGHT_IRIS_INDICES = [469, 470, 471, 472]

LEFT_EYE_POINT_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_POINT_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

MIN_VERTEX = 3 # 5
HORIZONTAL_EYELID_OFFSET = 4.5
VERTICAL_EYELID_OFFSET = 1.5
HORIZONTAL_IRIS_OFFSET = 1.09

HORIZONTAL_MAGNIFIER_REFERENCE = 64  
VERTICAL_MAGNIFIER_REFERENCE = 20  

# eyelid height by eyelid width ratio
EYE_OPEN_THRESHOLD = 0.18 # 0.12, 0.15, 0.18 
# minimum number of horizontal pixels to be fitted inside the ROI blobs computed
MIN_HORIZONTAL_PIXELS_FOR_TRIANGLE = 4


class ExtractEyeLandmarks:
    
    @classmethod
    def get_face_mesh_points(cls,face_image:np.ndarray):
        """Extracts coordinates of eyes landmarks

        Args:
            face_image (np.ndarray): Face Image 

        Returns:
            List[List[int]]: Eyelid landmark coordinates
        """
        with mp_face_mesh.FaceMesh(max_num_faces = 1, refine_landmarks = True, min_detection_confidence = MINIMUM_DETECTION_THRESHOLD) as face_mesh:
            rgb_frame = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            img_h, img_w = face_image.shape[:2]
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                return mesh_points # list of list
    
    @classmethod
    def get_landmark_coordinates(cls,face_mesh_points:List[List[int]],eye_lid_index_list:List[int]):
        """Extract Coordinates from face mesh using eye landmarks indices

        Args:
            face_mesh_points (List[List[int]]): Output from mediapipe 
            eye_lid_index_list (List[int]): Defined points of landmarks

        Returns:
            List[List[int]]: Coordinates of given Landmarks
        """
        return face_mesh_points[eye_lid_index_list]
    
class CropEyePartByVertices:
    """Crops ROI from face image using Zero Masking to non ROI  
    """
    @classmethod
    def create_roi_masked_box(cls,triangular_vertices:List[List[int]]):
        vertex1, vertex2, vertex3 = triangular_vertices
        # X1 and X2 along x-axis
        min_x_coordinate = min(vertex1[0], vertex2[0],vertex3[0])
        max_x_coordinate = max(vertex1[0], vertex2[0],vertex3[0])
        # Y1 and Y2 along y-axis
        min_y_coordinate = min(vertex1[1], vertex2[1],vertex3[1])
        max_y_coordinate = max(vertex1[1], vertex2[1],vertex3[1])
        box = [min_x_coordinate,min_y_coordinate,max_x_coordinate,max_y_coordinate]
        return box
    
    @classmethod
    def crop_image_inside_vertices(cls,img:np.ndarray,n_vertex_points:List[List[int]]=None):
        """Crops image inside the given vertices

        Args:
            img (np.ndarray): _description_
            n_vertex_points (List[List[int]], optional): _description_. Defaults to None.

        Raises:
            TypeError: Arises when image is other than np.ndarray type
            ValueError: Arises when number of vertices are less than 3.

        Returns:
            np.ndarray: Cropped image blobs
        """
        # Create a black mask with the same size as the image
        if not isinstance(img, np.ndarray):
            raise TypeError('Image must be ndarray type.')
        mask = np.zeros_like(img)
        # Define the triangular region (you can adjust the points based on your requirements)
            
        if n_vertex_points is None or len(n_vertex_points) < MIN_VERTEX:
            raise ValueError('At least 3 vertices(coordinates) must be passed as a parameter.')
        else:
            # vertices for any polygon
            n_vertex_points = np.array(n_vertex_points, dtype=np.int32)

        # Fill the triangular region in the mask with white color
        cv2.fillPoly(mask, [n_vertex_points], (255, 255, 255))

        # Bitwise AND operation to apply the mask to the image; 
        result_image = cv2.bitwise_and(img, mask)
        # Create a boolean mask for non-zero pixels
        # non_zero_mask = result_image.copy() != 0
        x1, y1, x2, y2 = cls.create_roi_masked_box(n_vertex_points)
        width_of_cropping_image = x2 - x1
        height_of_cropping_image = y2 - y1
        cropped_roi_image_section = result_image[y1:y1 + height_of_cropping_image, x1:x1 + width_of_cropping_image]
        return cropped_roi_image_section  # .astype(np.float32)


class DetermineVertices:
    """Determine Triangular vertices on given Eyelid and Iris coordinates
    """
    @classmethod
    def determine_triangular_vertices(cls,eyelid_coordinates:List[List[int]], iris_coordinates:List[List[int]],which_eye='left'):
        """Compute Coordinates for Triangular Vertices

        Args:
            eyelid_coordinates (List[List[int]]): Coordinates of eyelid in face image
            iris_coordinates (List[List[int]]): Coordinates of iris in face image
            which_eye (str, optional): Left and right eyes. Defaults to 'left'.

        Returns:
            List[Union[List[int],List[]]: Coordinates of Vertex either empty list or with integer values
        """
        # increases with increase in face dimensions
        eyelid_x1 = eyelid_coordinates[0][0]
        eyelid_x2 = eyelid_coordinates[8][0]
        eyelid_y1 = eyelid_coordinates[4][1]
        eyelid_y2 = eyelid_coordinates[12][1]
        eyelid_width = abs(eyelid_x2 - eyelid_x1)
        eyelid_height = abs(eyelid_y2 - eyelid_y1)
        
        eye_width_ratio = eyelid_width / HORIZONTAL_MAGNIFIER_REFERENCE
        eye_height_ratio = eyelid_height / VERTICAL_MAGNIFIER_REFERENCE
        # calculate the ratio of eyelid height by eyelid width; it helps to know how much eye is open with respect to the length of eye.
        eye_open_close_detection_ration = eyelid_height / eyelid_width
        # check if eye is open or closed comparing to the defined constant threshold; if closed return empty lists
        if eye_open_close_detection_ration < EYE_OPEN_THRESHOLD:
            left_and_right_iris_triangle_vertices = [[],[]]
            return left_and_right_iris_triangle_vertices
        
        
        IRIS_LEFT_SHIFT = False
        IRIS_RIGHT_SHIFT = False
        
        #  Section Eye at center from iris vertically and there are left and right part of eyes with one-one triangles 
        # --------------- TRIANGULAR VERTICES IN EYE TOWARD NOSE ---------------------------
        if which_eye == 'left':
            left_vertex_x = int((eyelid_coordinates[1][0] + eyelid_coordinates[2][0])/2)
            left_vertex_y = int((eyelid_coordinates[2][1] + eyelid_coordinates[14][1])/2)
            
        if which_eye == 'right':
            left_vertex_x = int((eyelid_coordinates[0][0] + eyelid_coordinates[1][0])/2)
            left_vertex_y = int((eyelid_coordinates[2][1] + eyelid_coordinates[14][1])/2)
            
        # x coordinate at circumference of iris
        left_vertex_x_at_iris = iris_coordinates[2][0]
        # vertex upper closer to iris
        left_lower_vertex_y = eyelid_coordinates[2][1]
        # vertex lower closer to iris
        # left_lower_vertex_y = eyelid_coordinates[14][1]
        left_upper_vertex_y = eyelid_coordinates[14][1]
        
        # --------------- TRIANGULAR VERTICES IN EYE AWAY FROM NOSE --------------------------
        # vertex touching iris 
        right_vertex_x_at_iris = iris_coordinates[0][0]
        # right_upper_vertex_y = eyelid_coordinates[6][1]
        right_lower_vertex_y = eyelid_coordinates[6][1]
        # right_lower_vertex_y = eyelid_coordinates[10][1]
        right_upper_vertex_y = eyelid_coordinates[10][1]
        
        # vertex away from iris
        if which_eye == 'left':
            right_vertex_x = int((eyelid_coordinates[7][0] + eyelid_coordinates[8][0])/2)
            right_vertex_y = int((eyelid_coordinates[6][1] + eyelid_coordinates[10][1])/2)
        
        if which_eye == 'right':
            right_vertex_x = int((eyelid_coordinates[6][0] + eyelid_coordinates[7][0])/2)
            right_vertex_y = int((eyelid_coordinates[6][1] + eyelid_coordinates[10][1])/2)
            
        # eye_width_ratio = eyelid_width / HORIZONTAL_MAGNIFIER_REFERENCE
        # eye_height_ratio = eyelid_height / VERTICAL_MAGNIFIER_REFERENCE
        
        
        # add offset to x and y coordinate values determined above where offset is multiplied by eye width and eye height ratio
        left_vertex_x = left_vertex_x + int(eye_width_ratio * HORIZONTAL_EYELID_OFFSET)
        right_vertex_x = right_vertex_x - int(eye_width_ratio * HORIZONTAL_EYELID_OFFSET)
        
        left_vertex_x_at_iris = left_vertex_x_at_iris - int(eye_width_ratio * HORIZONTAL_IRIS_OFFSET)
        if left_vertex_x_at_iris - left_vertex_x < MIN_HORIZONTAL_PIXELS_FOR_TRIANGLE:
            left_vertex_x_at_iris = left_vertex_x
            IRIS_LEFT_SHIFT = True
            
        right_vertex_x_at_iris = right_vertex_x_at_iris + int(eye_width_ratio * HORIZONTAL_IRIS_OFFSET)
        if right_vertex_x - right_vertex_x_at_iris < MIN_HORIZONTAL_PIXELS_FOR_TRIANGLE:
            right_vertex_x_at_iris = right_vertex_x
            IRIS_RIGHT_SHIFT = True
            
        left_upper_vertex_y = left_upper_vertex_y + int(eye_height_ratio * VERTICAL_EYELID_OFFSET)
        left_lower_vertex_y = left_lower_vertex_y - int(eye_height_ratio * VERTICAL_EYELID_OFFSET)
        right_upper_vertex_y = right_upper_vertex_y + int(eye_height_ratio * VERTICAL_EYELID_OFFSET)
        right_lower_vertex_y = right_lower_vertex_y - int(eye_height_ratio * VERTICAL_EYELID_OFFSET)
        
        # if iris is not in center of eye horizontal line, then some ROI part may be unclear or not visible
        # so we apply different strategies to consider visible eye ROI part only
        if not IRIS_LEFT_SHIFT and not IRIS_RIGHT_SHIFT:
            left_and_right_iris_triangle_vertices = [
                [[left_vertex_x,left_vertex_y],[left_vertex_x_at_iris,left_upper_vertex_y],[left_vertex_x_at_iris,left_lower_vertex_y]],
                [[right_vertex_x,right_vertex_y],[right_vertex_x_at_iris,right_upper_vertex_y],[right_vertex_x_at_iris,right_lower_vertex_y]]
            ]
            
        elif IRIS_LEFT_SHIFT and not IRIS_RIGHT_SHIFT:
            left_and_right_iris_triangle_vertices = [[],
                [[right_vertex_x,right_vertex_y],[right_vertex_x_at_iris,right_upper_vertex_y],[right_vertex_x_at_iris,right_lower_vertex_y]]]
            
        elif not IRIS_LEFT_SHIFT and IRIS_RIGHT_SHIFT:
            left_and_right_iris_triangle_vertices = [
                [[left_vertex_x,left_vertex_y],[left_vertex_x_at_iris,left_upper_vertex_y],[left_vertex_x_at_iris,left_lower_vertex_y]],[]]
        else:
            left_and_right_iris_triangle_vertices = [[],[]]
        return left_and_right_iris_triangle_vertices

class FilterZeroMaskedPixels:
    @classmethod
    def filter_masked_zero_pixels(cls,masked_zero_pixel_image:np.ndarray):
        """Flattens zero Masked Image by filtering non-zero rgb pixels to an array and recreate Image from Filtered array

        Args:
            masked_zero_pixel_image (np.ndarray): Image where ROI has non-zero rgb pixels and remaining pixels are zero

        Returns:
            np.ndarray: Recreated non-zero pixel image of Dimension (2,-1,3)
        """
        # Filter out pixels with all zeros
        non_zero_pixels = masked_zero_pixel_image[np.all(masked_zero_pixel_image != 0, axis=-1)]
        flattened_array = non_zero_pixels.flatten()
        n_flattened = len(flattened_array)
        # to count pixels by dividing flattened array by 3 (r,g,b) channels
        n_pixels = n_flattened / 3
        # to create image shape in even dimension check whether number of pixels are even or not
        is_even = True if n_pixels % 2 == 0 else False
        # if number of pixels are odd we subtract one pixels .i.e 3 array elements from the last position
        if not is_even:
            flattened_array = flattened_array[:-3]
        # recreate image of 2 by remaining array elements that fits in 2 rows;  shape of image will be (2,-1,3)
        recreated_image = flattened_array.reshape((2,-1,3))
        return recreated_image


class RetrieveMaskedBlobImage:
    @classmethod
    def get_cropped_blob_image(cls,face_image:np.ndarray,plot_roi_added=False):
        """Run methods to Detect Face , create triangular vertices, mask ROI (area inside ROI), Crop image inside ROI.

        Args:
            face_image (np.ndarray): Face Image
            plot_roi_added (bool, optional): Plot for debugging. Defaults to False.

        Returns:
            Union[Dict[str,Union[np.ndarray,None],Tuple[None,str]]: _description_
        """
        cropped_eyes_left_right_blobs = {}
        # EXTRACT LANDMARKS
        mesh = ExtractEyeLandmarks.get_face_mesh_points(face_image)
        if mesh is None:
            return None, 'Face not Found'
        left_eye_eyelid = ExtractEyeLandmarks.get_landmark_coordinates(mesh,LEFT_EYE_POINT_INDICES)
        left_iris = ExtractEyeLandmarks.get_landmark_coordinates(mesh,LEFT_IRIS_INDICES)
        right_eye_eyelid = ExtractEyeLandmarks.get_landmark_coordinates(mesh,RIGHT_EYE_POINT_INDICES)
        right_iris = ExtractEyeLandmarks.get_landmark_coordinates(mesh,RIGHT_IRIS_INDICES)
        
        if plot_roi_added:
            plot_eye.plot_dots_on_eye_landmarks(face_image.copy(),[left_iris,right_iris],[left_eye_eyelid,right_eye_eyelid])
            
        # determine triangle vertices
        left_eye_left_vertices,left_eye_right_vertices = DetermineVertices.determine_triangular_vertices(left_eye_eyelid,left_iris,which_eye='left')
        right_eye_left_vertices,right_eye_right_vertices = DetermineVertices.determine_triangular_vertices(right_eye_eyelid,right_iris,which_eye='right')
        all_vertices = [left_eye_left_vertices,left_eye_right_vertices,right_eye_left_vertices,right_eye_right_vertices]
        
        if plot_roi_added:
            triangle_vertex_added_image = plot_eye.plot_triangular_boundary_on_eyes(face_image.copy(),all_vertices,return_plotted_image=False)
        else:
            triangle_vertex_added_image = plot_eye.plot_triangular_boundary_on_eyes(face_image.copy(),all_vertices,return_plotted_image=True)
            
        blob_keys = ['left_left_blob','left_right_blob','right_left_blob','right_right_blob']
        
        for i, vertex_ in enumerate(all_vertices):
            if vertex_ != []:
                vertices_added_image = CropEyePartByVertices.crop_image_inside_vertices(face_image,n_vertex_points=vertex_)
                if plot_roi_added:
                    plot_eye.display_image(vertices_added_image,'Blob Images Cropped from Eyes with Mask')
                cropped_eyes_left_right_blobs[blob_keys[i]] = vertices_added_image
            else:
                cropped_eyes_left_right_blobs[blob_keys[i]] = None
        
        return cropped_eyes_left_right_blobs, triangle_vertex_added_image

if __name__ =="__main__":
    pass
 











