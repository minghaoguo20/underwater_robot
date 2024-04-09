import cv2
import os
from datetime import datetime

cap = cv2.VideoCapture(0)

output_dir = "/home/rp24/file/debug_output"

idx = 0

while True:
    ret, frame = cap.read()

    # Perform operations on the frame here
    save_path = os.path.join(output_dir, f"frame_{idx}.jpg")
    cv2.imwrite(save_path, frame)
    # cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    idx += 1

    print(f"Frame {idx} saved")

    if idx > 100:
        break

cap.release()
cv2.destroyAllWindows()