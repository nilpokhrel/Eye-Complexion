

# plot eyelids and iris on face image
from typing import List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2

def plot_dots_on_eye_landmarks(face_image:np.ndarray,iris_coordinate:List[List[List[int]]],eyelid_coordinate:List[List[List[int]]]):
    """Plot dots in Eyelid and Iris landmarks on face Image.

    Args:
        face_image (np.ndarray): Input Face Image for Plotting.
        iris_coordinate (List[List[List[int]]]): Four Iris coordinates
        eyelid_coordinate (List[List[List[int]]]): Sixteen Eyelid coordinates 
    """
    plt.figure(figsize=(15, 10))
    # Plot image using Matplotlib
    plt.imshow(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    # unpack left and right eye landmarks
    left_iris, right_iris = iris_coordinate
    left_eye_eyelid, right_eye_eyelid = eyelid_coordinate
    # Plot points for the left eye
    plt.scatter(left_iris[:, 0], left_iris[:, 1], c='m', marker='o', label='Left Eye Points')
    plt.scatter(left_eye_eyelid[:, 0], left_eye_eyelid[:, 1], c='m', marker='o', label='Left Eye Points')
    plt.scatter(right_iris[:, 0], right_iris[:, 1], c='m', marker='o', label='Left Eye Points')
    plt.scatter(right_eye_eyelid[:, 0], right_eye_eyelid[:, 1], c='m', marker='o', label='Left Eye Points')
    plt.show()
    
    

def plot_triangular_boundary_on_eyes(face_image:np.ndarray,triangles_vertices:List[List[List[int]]],return_plotted_image=False):
    """Plot Triangular ROI on Eye  

    Args:
        face_image (np.ndarray): Face Image to plot ROI
        triangles_vertices (List[List[List[int]]]): At least one and at most four Coordinates of triangular vertices 
        return_plotted_image (bool, optional): IF True returns modified image with Triangular ROI else Displays plotted image with Triangular ROI. Defaults to False.

    Returns:
        np.ndarray: Original Face Image with Plotted triangular shape ROI in eyes
    """
    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.imshow(face_image)

    # Create Polygon patches for each triangle
    for _vertices in triangles_vertices:
        if _vertices != []:
            triangle = patches.Polygon(np.array(_vertices), closed=True, edgecolor='r', linewidth=1, facecolor='none')
            ax.add_patch(triangle)
            cv2.polylines(face_image, [np.array(_vertices)], isClosed=True, color=(255, 0, 0), thickness=1)

    # Set the aspect ratio to 'equal' for a proper display
    ax.set_aspect('equal')

    # Show the plot
    if return_plotted_image:
        return face_image
    plt.show()
    
# plot images for debugging
def display_image(image:np.ndarray,message:str):
    plt.imshow(image)
    plt.title(message)
    plt.show()


if __name__ == '__main__':
    pass