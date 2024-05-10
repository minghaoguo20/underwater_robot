import cv2
import os
from datetime import datetime
from datetime import timedelta

from utils.line import *
from utils.gpio import *
from utils.img import *


def main(cfg):
    cap = cv2.VideoCapture(0)
    last = datetime.now() - timedelta(seconds=100)

    lower_rgb = (30, 20, 0)
    upper_rgb = (80, 80, 80)

    com = serial.Serial(port="/dev/ttyAMA0", baudrate=9600, stopbits=1, bytesize=8, parity="N")

    last_following_line = None
    last_following_cal = None
    prepare_to_transfer = False
    figure_recg = False
    figure_task = np.zeros(6)
    next_point = 1
    while True:
        now = datetime.now()
        ret, frame = cap.read()
        height, width, _ = frame.shape
        # cv2.imshow('frame', frame)

        if now > last + timedelta(seconds=cfg["sample_delta"]):
            last = now

            lines = get_lines(frame, "bgr", lower_rgb, upper_rgb)
            if lines is None:
                continue
            cal = get_center_and_angle_len(lines)

            lines, cal = get_lines_and_cal(lines, cal)
            if lines is None:
                continue

            xy_percent = 0.2
            if in_center(last_following_line, x_percent=xy_percent, y_percent=xy_percent):
                prepare_to_transfer = True

            line_nearest, cal_nearesr, idx_min_dis = get_line_and_cal_to_follow(
                lines, cal, last_following_line, last_following_cal
            )
            if prepare_to_transfer:
                min_dis = 100000
                del lines[idx_min_dis]
                del cal[idx_min_dis]
                for line, acal in zip(lines, cal):
                    if in_center(line, x_percent=xy_percent, y_percent=xy_percent):
                        dis = np.square(acal[0] - last_following_cal[0]) + np.square(
                            acal[1] - last_following_cal[1]
                        )
                        if dis < min_dis:
                            min_dis = dis
                            line_to_follow = line
                            cal_to_follow = acal
                            prepare_to_transfer = False
                        if figure_task[next_point] == 0:
                            res, confidence = recognize_figure(frame)
                            if confidence > 0.5:
                                figure_recg = True
                                figure_task[next_point] == 1
                                next_point += 1
                if prepare_to_transfer:
                    line_to_follow, cal_to_follow = line_nearest, cal_nearesr                
            else:
                line_to_follow, cal_to_follow = line_nearest, cal_nearesr
            
            # update
            last_following_line, last_following_cal = line_to_follow, cal_to_follow

            # calculate the angle and error
            cam_angle = cal_to_follow[2] - 90
            cam_err = (width / 2 - cal_to_follow[4]) * np.sin(
                np.pi - cal_to_follow[2] * np.pi / 180
            )

            # print(f"cam_angle: {cam_angle}", f"cam_err: {cam_err}")

            # send the angle and error to the STM32
            if figure_recg and figure_task[next_point] == 0:
                # senf msg of figure recg
                pass
            else:
                msg = pack_lora_msg(0, 0, cam_angle, cam_err)
            com.write(msg)

            # Save frame to directory
            save_path = os.path.join(
                cfg["output_dir"], f"frames_saved_{now.strftime('%Y%m%d%H%M%S')}.jpg"
            )
            # print(save_path)
            cv2.imwrite(save_path, frame)

            if cfg["test_steps"] == 0:
                break
            cfg["test_steps"] -= 1

        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    cap.release()
    cv2.destroyAllWindows()


def debug():
    img_path = "/home/rp24/file/test_img/aw4.jpg"
    img_path = "/home/rp24/file/test_img/1_2.png"
    print(f"img_path: {img_path}")

    lower_hsv = np.array([0, 0, 0])  # 直线色域 0 0 245
    upper_hsv = np.array([180, 200, 170])
    lower_rgb = (30, 20, 0)
    upper_rgb = (80, 80, 80)

    frame = cv2.imread(img_path)
    frame = cv2.resize(
        frame,
        (600, 400),
    )
    height, width, _ = frame.shape
    print(f"frame shape: {frame.shape}")

    lines = get_lines(frame, "bgr", lower_rgb, upper_rgb)
    cal = get_center_and_angle_len(lines)

    line_to_follow, cal_to_follow = get_lines_and_cal(lines, cal)

    cam_angle = cal_to_follow[2] - 90
    cam_err = (width / 2 - cal_to_follow[4]) * np.sin(
        np.pi - cal_to_follow[2] * np.pi / 180
    )

    print(f"cam_angle: {cam_angle}")
    print(f"cam_err: {cam_err}")

    # print(f"idx_clusters:\n {idx_clusters}")
    # print(f"idx_used:\n {idx_used}")
    # print(f"idx_num:\n {idx_num}")

    # x_min_idx = np.argmin(cal[:, 0], axis=0)

    # plt.imshow(frame)

    # # Add lines on the frame
    # num_start = 0
    # num_end = num_start + 44
    # for idx in range(lines[num_start:num_end, ...].shape[0]):
    #     x0, y0, x1, y1 = lines[idx]
    #     plt.plot([x0, x1], [y0, y1], color='red', linewidth=2)
    # x0, y0, x1, y1 = lines[x_min_idx]
    # plt.plot([x0, x1], [y0, y1], color='blue', linewidth=2)
    # print(f"blue alpha = {cal[x_min_idx, 2]}")

    # # Show the plot
    # plt.show()
    # # save the plot
    # plt.savefig(f"/home/rp24/code/underwater_robot/file/debug_output/o.png")

    # print(f"lines shape: {lines.shape}")
    # if lines is None:
    #     print("no lines")
    #     return
    # cal = get_center_and_angle_len(lines)
    # print(f"cal shape: {cal.shape}")
    # x_min_idx = np.argmin(cal[:, 0], axis=0)
    # print(f"x_min_idx: {x_min_idx}")
    # print(f"cal[x_min_idx]: {cal[x_min_idx]}")
    # print(f"lines[x_min_idx]: {lines[x_min_idx]}")
    # threshold = min(height, width) / 4
    # idx = filter_lines(lines, cal, threshold=threshold)
    # print(f"idx shape with threshold == {threshold}: {idx.shape}")


if __name__ == "__main__":
    dt = datetime.now()
    cfg = {
        "sample_delta": 0.2,
        "output_dir": f"/home/rp24/code/underwater_robot/file/debug_output_{dt.strftime('%Y%m%d%H%M%S')}",
        "test_steps": 1000,
    }
    os.makedirs(cfg["output_dir"], exist_ok=True)
    main(cfg)
    # debug()
