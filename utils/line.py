import cv2
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans


def get_lines(frame, rgb_hsv, lower, upper):
    """
    get lines from frame
    input: frame, lower_hsv, upper_hsv
    output: lines
    output shape: (num_lines, 4)
    """
    if rgb_hsv == "hsv":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lowerb=lower, upperb=upper)
    elif rgb_hsv == "bgr":
        mask1 = cv2.inRange(frame, lowerb=lower, upperb=upper)
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


def cluster_idealx_alpha(
    tx_alpha, w=600, h=400, tx_threshold=30, alpha_threshold=5, min_lines=5
):
    """
    cluster the lines based on the idealx and alpha with threshold
    input: numpy array (index, data)
        data: center_x, center_y, angle, length, idealx
    output: numpy array (index, data)
        data: cluster_index
    """
    idx_clusters = np.zeros((tx_alpha.shape[0]), dtype=int)
    max_index = 0
    clusters_added = (
        []
    )  # to append ((avg_tx, avg_alpha, idx, n, data[(tx, alpha), (), ...]))
    for idx in range(tx_alpha.shape[0]):
        flag_find_cluster = False
        tx, alpha = tx_alpha[idx]
        for cluster in clusters_added:
            avg_tx, avg_alpha, this_idx, this_n = (
                cluster["avg_tx"],
                cluster["avg_alpha"],
                cluster["idx"],
                cluster["n"],
            )
            if (
                abs(tx - avg_tx) < tx_threshold
                and abs(alpha - avg_alpha) < alpha_threshold
            ):
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
            clusters_added.append(
                {
                    "avg_tx": tx,
                    "avg_alpha": alpha,
                    "idx": max_index,
                    "n": 1,
                    "data": [(tx, alpha)],
                }
            )
            idx_clusters[idx] = max_index
    idx_to_delete = []
    for cluster in clusters_added:
        if cluster["n"] < min_lines:
            idx_to_delete.append(cluster["idx"])
    for i in range(idx_clusters.shape[0]):
        if idx_clusters[i] in idx_to_delete:
            idx_clusters[i] = 0

    idx_used = np.unique(idx_clusters)
    idx_num = idx_used.shape[0] - 1

    return idx_clusters, idx_used, idx_num


def get_lines_and_cal(lines, cal):
    idx_clusters, idx_used, idx_num = cluster_idealx_alpha(cal[:, [4, 2]], min_lines=5)
    idx_used = np.delete(idx_used, np.where(idx_used == 0))
    if len(idx_used) == 0:
        return None, None

    lines_to_use = []
    cal_to_use = []
    for idx in idx_used:
        this_cluster_index = np.where(idx_clusters == idx)
        this_lines = lines[this_cluster_index]
        this_cal = cal[this_cluster_index]
        idx_max_length = np.argmax(this_cal[:, 3], axis=0)
        lines_to_use.append(this_lines[idx_max_length])
        cal_to_use.append(this_cal[idx_max_length])
    lines = np.stack(lines_to_use, axis=0)
    cal = np.stack(cal_to_use, axis=0)
    # print(f"lines: {lines}")
    # print(f"cal: {cal}")
    return lines, cal


def get_line_and_cal_to_follow(lines, cal, last_following_line, last_following_cal):
    # # find the most left line
    # idx_left = np.argmin(cal[:, 0], axis=0)
    # line_to_follow = lines[idx_left]
    # cal_to_follow = cal[idx_left]
    # # print(f"line_to_follow: {line_to_follow}")
    # # print(f"cal_to_follow: {cal_to_follow}")
    if last_following_line is None:
        idx_left = np.argmin(cal[:, 0], axis=0)
        line_to_follow = lines[idx_left]
        cal_to_follow = cal[idx_left]
        return line_to_follow, cal_to_follow
    dis = np.square(cal[:, 0] - last_following_cal[0]) + np.square(
        cal[:, 1] - last_following_cal[1]
    )
    idx_min_dis = np.argmin(dis, axis=0)
    line_to_follow = lines[idx_min_dis]
    cal_to_follow = cal[idx_min_dis]
    return line_to_follow, cal_to_follow


def in_center(line, w=600, h=400, x_percent=0.3, y_percent=0.3):
    x1, y1, x2, y2 = line
    x_min = w * (1 - x_percent) / 2
    x_max = w * (1 + x_percent) / 2
    y_min = h * (1 - y_percent) / 2
    y_max = h * (1 + y_percent) / 2
    if x1 > x_min and x1 < x_max and y1 > y_min and y1 < y_max:
        return True, (x1, y1)
    if x2 > x_min and x2 < x_max and y2 > y_min and y2 < y_max:
        return True, (x2, y2)
    return False, None


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
