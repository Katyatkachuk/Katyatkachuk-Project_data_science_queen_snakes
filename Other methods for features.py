# Asymmetry 2 version
import cv2
import numpy as np
import os
import csv
from sklearn.cluster import KMeans

"""
    Check the symmetry of an input image.

    Args:
    - image_path (str): Path to the input image.

    Returns:
    - str: Symmetry classification of the image.
           "1" indicates both horizontal and vertical symmetry.
           "2" indicates either horizontal or vertical symmetry.
           "3" indicates no symmetry.
    """

def check_symmetry(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    left_half = img[:, :w // 2]
    right_half = img[:, w // 2:]
    right_half_flipped = cv2.flip(right_half, 1)
    top_half = img[:h // 2, :]
    bottom_half = img[h // 2:, :]
    bottom_half_flipped = cv2.flip(bottom_half, 0)

    def is_similar(part_a, part_b, threshold=5):
        if part_a.shape != part_b.shape:
            return False
        diff = cv2.absdiff(np.float32(part_a), np.float32(part_b))
        percent_diff = (np.sum(diff) / (255 * diff.size)) * 100
        return percent_diff < threshold

    horizontal_sym = is_similar(left_half, right_half_flipped)
    vertical_sym = is_similar(top_half, bottom_half_flipped)
    if horizontal_sym and vertical_sym:
        return "1"
    elif horizontal_sym or vertical_sym:
        return "2"
    else:
        return "3"


# colour 2 version

def color_analysis(image):
    """
    Analyze the colors present in the input image.

    Args:
    - image (numpy.ndarray): Input image (RGB).

    Returns:
    - num_colors (int): Number of distinct colors present in the image.
    """
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=6, random_state=0)
    kmeans.fit(image.reshape(-1, 3))

    # Filter out similar colors
    _, counts = np.unique(kmeans.labels_, return_counts=True)
    num_colors = len([count for count in counts if count >= 0.05 * image.size])

    return num_colors
