import os,sys

import cv2
import numpy as np
from datetime import datetime
# import matplotlib.pyplot as plt
import math
# from sklearn.cluster import KMeans


def get_lines(frame, rgb_hsv, lower, upper):
    """
    get lines from frame
    input: frame, lower_hsv, upper_hsv
    output: lines
    output shape: (num_lines, 4)
    """
    # if rgb_hsv == "hsv":
    #     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #     mask1 = cv2.inRange(hsv, lowerb=lower, upperb=upper)
    # elif rgb_hsv == "bgr":
    #     mask1 = cv2.inRange(frame, lowerb=lower, upperb=upper)
    mask1 = cv2.inRange(frame, lowerb=lower, upperb=upper)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 12))
    mask1 = cv2.dilate(mask1, kernel1)
    mask1 = cv2.erode(mask1, kernel1)
    lines = cv2.HoughLinesP(
        mask1, 1, np.pi / 180, threshold=60, minLineLength=50, maxLineGap=200
    )
    if lines is not None:
        lines = np.squeeze(lines, axis=1)
    return lines


def reg_color(frame, lower, upper):
    percentage = reg_color_percentage(frame, lower, upper)
    return percentage


def reg_color_percentage(frame, lower, upper):
    mask1 = cv2.inRange(frame, lowerb=lower, upperb=upper)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 12))
    mask1 = cv2.dilate(mask1, kernel1)
    mask1 = cv2.erode(mask1, kernel1)
    percentage = np.count_nonzero(mask1) / mask1.size * 100
    return percentage

    
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


def get_center_and_angle_len(lines, w=640, h=480):
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


def cluster_idealx_alpha(tx_alpha, w=640, h=480, tx_threshold=50, alpha_threshold=10, min_lines=2):
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
        for this_cluster in clusters_added:
            avg_tx, avg_alpha, this_idx, this_n = (
                this_cluster["avg_tx"],
                this_cluster["avg_alpha"],
                this_cluster["idx"],
                this_cluster["n"],
            )
            if (
                abs(tx - avg_tx) < tx_threshold
                and abs(alpha - avg_alpha) < alpha_threshold
            ):
                this_cluster["data"].append((tx, alpha))
                this_cluster["avg_tx"] = (avg_tx * this_n + tx) / (this_n + 1)
                this_cluster["avg_alpha"] = (avg_alpha * this_n + alpha) / (this_n + 1)
                this_cluster["n"] += 1
                this_cluster["data"].append((tx, alpha))
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
    for this_cluster in clusters_added:
        if this_cluster["n"] < min_lines:
            idx_to_delete.append(this_cluster["idx"])
    for i in range(idx_clusters.shape[0]):
        if idx_clusters[i] in idx_to_delete:
            idx_clusters[i] = 0

    idx_used = np.unique(idx_clusters)
    idx_num = idx_used.shape[0] - 1

    return idx_clusters, idx_used, idx_num


def get_lines_and_cal(lines, cal, width, height, min_lines=2):
    idx_clusters, idx_used, idx_num = cluster_idealx_alpha(cal[:, [4, 2]], min_lines=min_lines, w=width, h=height)
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
    idx_to_remove = []
    for i in range(lines.shape[0]):
        line = lines[i]
        y_min = min(line[1], line[3])
        if y_min > 200:
            idx_to_remove.append(i)
    if len(idx_to_remove) < lines.shape[0]:
        lines = np.delete(lines, idx_to_remove, axis=0)
        cal = np.delete(cal, idx_to_remove, axis=0)

    if last_following_line is None:
        idx_left = np.argmin(cal[:, 0], axis=0)
        line_to_follow = lines[idx_left]
        cal_to_follow = cal[idx_left]
        return line_to_follow, cal_to_follow
    dis = np.square(cal[:, 0] - last_following_cal[0]) + np.square(cal[:, 1] - last_following_cal[1])
    idx_min_dis = np.argmin(dis, axis=0)
    line_to_follow = lines[idx_min_dis]
    cal_to_follow = cal[idx_min_dis]

    # irrelative_lines = np.array(lines)
    # irrelative_lines = np.delete(irrelative_lines, idx_min_dis, axis=0)
    irrelative_cal = np.array(cal)
    irrelative_cal = np.delete(irrelative_cal, idx_min_dis, axis=0)
    irrelative_lines_center_x_mean = np.mean(irrelative_cal[:, 0])

    return line_to_follow, cal_to_follow, idx_min_dis, irrelative_lines_center_x_mean


def get_left_line_and_cal(lines, cal):
    # find the most left line
    idx_left = np.argmin(cal[:, 0], axis=0)
    return lines[idx_left], cal[idx_left]


def in_center(line, w=640, h=480, x_percent=0.3, y_percent=0.3):
    if line is None:
        return False, None
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


# def cluster(data, n_clusters=2):
#     """
#     input: numpy array
#     """
#     kmeans = KMeans(n_clusters=n_clusters)
#     kmeans.fit(data)
#     return kmeans.labels_


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


def prolong_line(line, width=640, height=480):
    """
    prolong the line to the boundary of the image in bi-direction
    """
    x1, y1, x2, y2 = line
    if x1 == x2:
        return line
    if y1 == y2:
        return line
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    x1_new = 0
    y1_new = b
    x2_new = width
    y2_new = k * x2_new + b
    if y1_new < 0:
        y1_new = 0
        x1_new = -b / k
    if y1_new > height:
        y1_new = height
        x1_new = (height - b) / k
    if y2_new < 0:
        y2_new = 0
        x2_new = -b / k
    if y2_new > height:
        y2_new = height
        x2_new = (height - b) / k
    return np.array([x1_new, y1_new, x2_new, y2_new])


def confine_angle(angle, desired_min, angle_min=-90, angle_range=180):
    """
    confine the angle to the range of [angle_min, angle_min + range]
    """
    # print(f"{angle}", end="")
    while angle < desired_min:
        angle += angle_range
        # print(f" -> {angle}", end="")
    while angle >= desired_min + angle_range:
        angle -= angle_range
        # print(f" -> {angle}", end="")
    return angle


def linear_interpolation(x0, y0, x1, y1, x):
    """
    linear interpolation
    """
    return y0 + (y1 - y0) / (x1 - x0) * (x - x0)

    

def cal_speed_original(angle, cam_err, bias, max_speed):
    speed = 0
    if np.abs(cam_err) > 100:
        speed = 0
    elif np.abs(cam_err) > 50:
        speed = linear_interpolation(50, 5, 100, 0, np.abs(angle))
    else:
        speed = linear_interpolation(50, 5, 0, max_speed, np.abs(angle))
    # print(f"speed: {speed}", end="")
    speed += bias

    speed_ratio = 0
    if np.abs(angle) > 45:
        speed_ratio = 0
    elif np.abs(angle) > 10:
        speed_ratio = linear_interpolation(10, 0.4, 45, 0, np.abs(angle))
    else:
        speed_ratio = linear_interpolation(0, 1, 10, 0.4, np.abs(angle))
    # print(f"speed_ratio: {speed_ratio}")
    speed = speed * speed_ratio
    # print(f"speed: {speed}")
    return speed

def cal_speed(angle, error, line_to_follow, cal_to_follow, bias=6, max_speed=20, width=640, height=480):
    res = []
    speed = cal_speed_original(angle, error, bias=bias, max_speed=max_speed)

    y_min = min(line_to_follow[1], line_to_follow[3])
    y_max = max(line_to_follow[1], line_to_follow[3])
    x_min = min(line_to_follow[0], line_to_follow[2])
    x_max = max(line_to_follow[0], line_to_follow[2])

    # first_angle_then_speed = False
    # first_speed_then_angle = False
    
    center_dist_square = np.square(cal_to_follow[0] - width/2) + np.square(cal_to_follow[1] - height/2)
    radius1 = 240
    radius2 = 160
    if center_dist_square > radius1 * radius1:
        angle = 0
    elif center_dist_square > radius2 * radius2:
        angle = angle * 0.8

    if y_max < 120:
        speed = 3
    if y_min > 360:
        speed = -3

    # if speed < 0:
    #     speed = 128 - speed

    # speed = np.uint8(speed)

    # if first_angle_then_speed and not first_speed_then_angle:
    #     res.append([speed, angle, 0])
    #     res.append([speed, 0, error])
    # elif first_speed_then_angle and not first_angle_then_speed:
    #     res.append([max(speed, 1), 0, error])
    #     res.append([speed, angle, 0])
    # else:
    #     res.append([speed, angle, error])
    res.append([speed, angle, error])
    return res


def exist_center_line(lines, cal, width, height):
    if lines is None:
        return False, ""
    for i in range(lines.shape[0]):
        this_line = lines[i]
        this_cal = cal[i]

        y_max = max(this_line[1], this_line[3])
        y_min = min(this_line[1], this_line[3])
        x_max = max(this_line[0], this_line[2])
        x_min = min(this_line[0], this_line[2])

        if y_min > width*0.4 and x_min > height*0.35 and x_max < height*0.65 and np.abs(confine_angle(this_cal[2]-90, -90)) < 15:
            return True, f"line: ({this_line[0]}, {this_line[1]}), ({this_line[2]}, {this_line[3]}), cal: ({this_cal[0]}, {this_cal[1]}, {this_cal[2]})"
    return False, ""


if __name__ == "__main__":
    print(confine_angle(148.06, -90))
