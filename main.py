import cv2
import os
from datetime import datetime
from datetime import timedelta

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


if __name__ == '__main__':
    cfg = {
        "sample_delta": 1,
        "output_dir": "/home/rp24/file/debug_output",
        "test_steps": 100,
    }
    main(cfg)
