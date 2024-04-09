import cv2
import os
from datetime import datetime
from datetime import timedelta

if __name__ == '__main__':

    cfg = {
        "sample_delta": 1,
        "output_dir": "/home/rp24/file/debug_output",
        "test_steps": 100,
    }

    cap = cv2.VideoCapture(0)
    last = datetime.now() - timedelta(seconds=100)

    while True:
        now = datetime.now()
        ret, frame = cap.read()

        if now > last + timedelta(seconds=cfg["sample_delta"]):
            last = now

            # process frame


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cfg["test_steps"] > 100:
            break
        cfg["test_steps"] += 1


    cap.release()
    cv2.destroyAllWindows()

