import cv2
import os
from datetime import datetime
from datetime import timedelta

from utils.line import *


def which_task(frame):
    pass


def which_recognition(frame):
    pass


def main(cfg):
    cap = cv2.VideoCapture(0)
    last = datetime.now() - timedelta(seconds=100)

    while True:
        now = datetime.now()
        ret, frame = cap.read()

        if now > last + timedelta(seconds=cfg["sample_delta"]):
            last = now

            # classify task
            # 1. line tracking 
            # 2. object recognition
            task = which_task(frame)

            if task == "line_tracking":
                print("line_tracking")
            elif task == "object_recognition":
                print("object_recognition")
                subtask = which_recognition(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cfg["test_steps"] > 100:
            break
        cfg["test_steps"] += 1


    cap.release()
    cv2.destroyAllWindows()

def debug():
    img_path = "/home/rp24/file/test_img/aw4.jpg"

    lower_hsv = np.array([0, 0, 40])   # 直线色域 0 0 245
    upper_hsv = np.array([180, 255, 255])

    frame = cv2.imread(img_path)
    frame = cv2.resize(frame,(600, 400),)
    height, width, _ = frame.shape

    lines = get_lines(frame, lower_hsv, upper_hsv)
    if lines is None:
        print("no lines")
        return
    cal = get_center_and_angle_len(lines)

    cal.shape

if __name__ == '__main__':
    cfg = {
        "sample_delta": 1,
        "output_dir": "/home/rp24/file/debug_output",
        "test_steps": 100,
    }
    # main(cfg)

    debug()