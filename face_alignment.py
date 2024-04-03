#  face alignment
import numpy as np
import cv2

MIN_ANGLE = 0.1

class FaceAlignment:
 
    @classmethod
    def rotate_vector(cls,vector, angle, center):

        # correct vector must be reverse in angle respect to face
        angle = -1 * angle
        # Convert angle from degrees to radians
        angle_rad = np.deg2rad(angle)
        
        # Translate the vector to the origin (subtract center coordinates)
        translated_vector = np.subtract(vector, center)
        
        # Perform rotation using rotation matrix
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad), np.cos(angle_rad)]])
        rotated_vector = np.dot(rotation_matrix, translated_vector)
        
        # Translate the rotated vector back to the original position (add center coordinates)
        rotated_vector = np.add(rotated_vector, center)
        int_rotated_vectors = (int(rotated_vector[0]),int(rotated_vector[1]))
        return int_rotated_vectors
    
    @classmethod
    def rotate_image(cls,face_image, angle, center):
        
        if -MIN_ANGLE < angle < MIN_ANGLE:
            return face_image
        # Get image dimensions
        height, width = face_image.shape[:2]
        # Calculate rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Apply rotation to the image
        rotated_image = cv2.warpAffine(face_image, rotation_matrix, (width, height))
        return rotated_image

    @classmethod
    def compute_rotation_angle(cls, eye_coordinates):
    
        x1, x2, y1, y2 = eye_coordinates
        vector1, vector2 = (x1,y1), (x2,y2)
        
        # Calculate the vector between the two points
        vector = np.subtract(vector2, vector1)
        
        # Calculate the angle of the vector relative to the horizontal axis
        angle_rad = np.arctan2(vector[1], vector[0])
        
        # Convert angle from radians to degrees
        angle_deg = np.rad2deg(angle_rad)
        
        return angle_deg, angle_rad
    
    @classmethod
    def xy_eye_mean(cls,landmarks):
        
        x1 = np.array([v[0] for v in landmarks.get('listLeftEye')]).mean()
        x2 = np.array([v[0] for v in landmarks.get('listRightEye')]).mean()
        y1 = np.array([v[1] for v in landmarks.get('listLeftEye')]).mean()
        y2 = np.array([v[1] for v in landmarks.get('listRightEye')]).mean()
        return (x1,x2,y1,y2)
    
    @classmethod
    def center_of_rotation(cls,eye_coordinates):
        
        x1, x2, y1, y2 = eye_coordinates
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        return int(x),int(y)
    
    @classmethod
    def rotate_landmarks(cls,angle,center,landmarks):
        if -MIN_ANGLE < angle < MIN_ANGLE:
            return landmarks
        # rotate each vector 
        corrected_landmarks = {}
        for key,vectors in landmarks.items():
            new_vectors = [cls.rotate_vector(points,angle,center) for points in vectors]
            corrected_landmarks[key] = new_vectors
        return corrected_landmarks
    
    @classmethod
    def face_image_landmark_alignment(cls,face_image,landmarks,plot=False):
        if face_image is None or landmarks is None:
            raise ValueError('Image and Face Landmarks must not be None.')
        
        x_y_coordinates = cls.xy_eye_mean(landmarks) 

        if isinstance(face_image,np.ndarray):
            deg,_ = cls.compute_rotation_angle(x_y_coordinates)
            rotation_center = cls.center_of_rotation(x_y_coordinates)
            new_rotated_img = cls.rotate_image(face_image,deg,rotation_center)
            aligned_vectors = cls.rotate_landmarks(deg,rotation_center,landmarks)
            if plot:
                cls.plot_image_point(new_rotated_img,rotation_center,aligned_vectors)
            return new_rotated_img, aligned_vectors

        else:
            from PIL import Image
            # Convert PIL image to NumPy array
            face_image = np.array(face_image)
            deg,_ = cls.compute_rotation_angle(x_y_coordinates)
            rotation_center = cls.center_of_rotation(x_y_coordinates)
            new_rotated_img = cls.rotate_image(face_image,deg,rotation_center)
            aligned_vectors = cls.rotate_landmarks(deg,rotation_center,landmarks)
            if plot:
                cls.plot_image_point(new_rotated_img,rotation_center,aligned_vectors)
            # Convert NumPy array to Pillow image
            pillow_image = Image.fromarray(new_rotated_img)
            return pillow_image, aligned_vectors
            

    @classmethod
    def plot_image_point(cls,face_image,rotation_center,landmarks):
        import matplotlib.pyplot as plt
        from matplotlib import image
        from matplotlib.patches import Rectangle
        
        # for landmark in landmarks:
        for _, point_list in landmarks.items():
            for points in point_list:
                plt.plot(points[0], points[1], marker='o', color='red')
        plt.plot(rotation_center[0], rotation_center[1], marker='o', color='green')
        plt.imshow(face_image)
        plt.show()
        
if __name__ == '__main__':
    pass
    
