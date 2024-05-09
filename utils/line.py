import cv2
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans


def get_lines(frame, rgb_hsv, lower_hsv, upper_hsv, lower_rgb, upper_rgb):
    """
    get lines from frame
    input: frame, lower_hsv, upper_hsv
    output: lines
    output shape: (num_lines, 4)
    """
    if rgb_hsv == "hsv":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
    elif rgb_hsv == "bgr":
        mask1 = cv2.inRange(frame, lowerb=lower_rgb, upperb=upper_rgb)
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


def get_center_and_angle_len(lines, w=600, h=400):
    """
    input: numpy array (index, coordinates)
        coordinates: x0, y0, x1, y1
    output: numpy array (index, data)
        data: center_x, center_y, angle, length, idealx
    output shape: (num_lines, 5)
    idealx: the x coordinate of the line at the height of h / 2
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

    idealx = center_x + (h / 2 - center_y) / np.tan((180 - angle) * np.pi / 180)

    return np.array([center_x, center_y, angle, length, idealx]).T

def filter_lines(lines, center_and_angle, filter="length", threshold=100):
    """
    return the index of lines that meet the filter
    """
    if filter == "length":
        return np.where(center_and_angle[:, 3] > threshold)[0]

def cluster_idealx_alpha(tx_alpha, w=600, h=400, tx_threshold=30, alpha_threshold=5, min_lines=5):
    """
    input: numpy array (index, data)
        data: center_x, center_y, angle, length, idealx
    output: numpy array (index, data)
        data: cluster_index
    """
    idx_clusters = np.zeros((tx_alpha.shape[0], 1))
    max_index = 0
    clusters_added = [] # to append ((avg_tx, avg_alpha, idx, n, data[(tx, alpha), (), ...]))
    for idx in range(tx_alpha.shape[0]):
        flag_find_cluster = False
        tx, alpha = tx_alpha[idx]
        for cluster in clusters_added:
            avg_tx, avg_alpha, this_idx, this_n = cluster["avg_tx"], cluster["avg_alpha"], cluster["idx"], cluster["n"]
            if abs(tx - avg_tx) < tx_threshold and abs(alpha - avg_alpha) < alpha_threshold:
                cluster["data"].append((tx, alpha))
                cluster["avg_tx"] = (avg_tx * this_n + tx) / (this_n + 1)
                cluster["avg_alpha"] = (avg_alpha * this_n + alpha) / (this_n + 1)
                cluster["n"] += 1
                cluster["data"].append((tx, alpha))
                idx_clusters[idx] = this_idx
                flag_find_cluster = True
                break
        if not flag_find_cluster:
            max_index += 1
            clusters_added.append({
                "avg_tx": tx,
                "avg_alpha": alpha,
                "idx": max_index,
                "n": 1,
                "data": [(tx, alpha)]
            })
            idx_clusters[idx] = max_index
    idx_to_delete = []
    for cluster in clusters_added:
        if cluster["n"] < min_lines:
            idx_to_delete.append(cluster["idx"])
    for i in range(idx_clusters.shape[0]):
        if idx_clusters[i] in idx_to_delete:
            idx_clusters[i] = 0

    idx_used = np.unique(idx_clusters)

    return idx_clusters, idx_used

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

# class lines():
#     def __init__(self, num_lines=6):
#         self.num_lines = num_lines
#         self.done = []
#         self.todo = []
#         for i in range(num_lines):
#             self.todo.append(i)
    
#     def get_current_task(self):
#         if len(self.todo) == 0:
#             return None
#         return self.todo[0]

#     def finish_a_line(self):
#         if len(self.todo) == 0:
#             return False
#         self.done.append(self.todo.pop(0))
