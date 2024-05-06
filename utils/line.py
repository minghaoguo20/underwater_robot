import cv2
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans


def get_lines(frame, lower_hsv, upper_hsv):
    """
    get lines from frame
    input: frame, lower_hsv, upper_hsv
    output: lines
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 12))
    mask1 = cv2.dilate(mask1, kernel1)
    mask1 = cv2.erode(mask1, kernel1)
    lines = cv2.HoughLinesP(
        mask1, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=50
    )
    if lines is not None:
        lines = np.squeeze(lines, axis=1)
    return lines


# def calculate_line_center_and_angle(x0, y0, x1, y1):
#     """
#     input the coordinates of two points of a line, calculate the center and angle of the line
#     input: x0, y0, x1, y1
#     output: center_x, center_y, angle
#     """
#     center_x = (x0 + x1) / 2
#     center_y = (y0 + y1) / 2

#     angle = math.atan2(y1 - y0, x1 - x0) * 180 / math.pi
#     # angle = 180 + angle if angle < 0 else angle
#     angle[angle < 0] = angle[angle < 0] + 180

#     return (center_x, center_y), angle


def get_center_and_angle_len(lines):
    """
    input: numpy array (index, coordinates)
        coordinates: x0, y0, x1, y1
    output: numpy array (index, data)
        data: center_x, center_y, angle, length
    """
    center_x = (lines[:, 0] + lines[:, 2]) / 2
    center_y = (lines[:, 1] + lines[:, 3]) / 2

    angle = (
        np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0]) * 180 / np.pi
    )
    # angle = 180 + angle if angle < 0 else angle
    angle[angle < 0] = angle[angle < 0] + 180

    length = np.sqrt(
        (lines[:, 2] - lines[:, 0]) ** 2 + (lines[:, 3] - lines[:, 1]) ** 2
    )

    return np.array([center_x, center_y, angle, length]).T


def cluster(data, n_clusters=2):
    """
    input: numpy array
    """
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    return kmeans.labels_


def process_lines(lines, center_and_angle):
    if lines is None:
        return False
