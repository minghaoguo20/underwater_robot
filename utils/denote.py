import os,sys

import cv2
import os

def denote_msg(denote_path, flag_msg_number, now, cam_angle=None, cam_err=None, lines=None, cal=None, line_nearest=None, cal_nearesr=None, ps=""):
    f = open(denote_path, 'a+')
    f.write(f"{now.strftime('%Y%m%d-%H:%M:%S')} flag_msg_number = {flag_msg_number}\n")
    if cam_angle is not None and cam_err is not None:
        f.write(f"cam_angle: {cam_angle}, cam_err: {cam_err}\n")

    if line_nearest is not None:
        f.write(f"$ line_nearest: ({line_nearest[0]}, {line_nearest[1]}), ({line_nearest[2]}, {line_nearest[3]}) \n")
    else:
        f.write(f"$ no line_nearest\n")
        # f.write(f"$ no line_nearest\n") if line_nearest.shape[0] == 0 else f.write(str(line_nearest))
    if cal_nearesr is not None:
        f.write(f"$ cal_nearesr: ({cal_nearesr[0]}, {cal_nearesr[1]}), ({cal_nearesr[2]}, {cal_nearesr[3]})\n")
    else:
        f.write(f"$ no cal_nearesr\n")
        # f.write(f"$ no cal_nearesr\n") if cal_nearesr.shape[0] == 0 else f.write(str(cal_nearesr))

    if lines is not None:
        f.write(f"$ lines:\n")
        for idx in range(lines.shape[0]):
            f.write(f"({lines[idx][0]}, {lines[idx][1]}), ({lines[idx][2]}, {lines[idx][3]})\n")
    else:
        f.write(f"$ no lines\n")
        # f.write(f"$ no lines\n") if lines.shape[0] == 0 else f.write(str(lines))
    if cal is not None:
        f.write(f"$ cal:\n")
        for idx in range(cal.shape[0]):
            f.write(f"({cal[idx][0]}, {cal[idx][1]}), ({cal[idx][2]}, {cal[idx][3]})\n")
    else:
        f.write(f"$ no cal\n")
        # f.write(f"$ no cal\n") if cal.shape[0] == 0 else f.write(str(cal))

    f.write(f"$ p.s.: \n{ps}\n")

    f.write(f"\n\n")
    f.close()

def save_img(frame_line, lines, line_to_follow, cam_angle, cam_err, speed, flag_msg_number, now, cfg, ps=""):
    # all lines
    if lines is not None:
        for this_line in lines:
            cv2.line(frame_line, (int(this_line[0]), int(this_line[1])), (int(this_line[2]), int(this_line[3]),), (255, 0, 0), 3)
    # line to follow
    if line_to_follow is None:
        line_to_follow = [-1, -1, -1, -1]
    cv2.line(frame_line, (int(line_to_follow[0]), int(line_to_follow[1])), (int(line_to_follow[2]), int(line_to_follow[3]),), (0, 255, 0), 3)
    # text
    if cam_angle is not None and cam_err is not None:
        text = f"angle/paw: {cam_angle:.2f}, err/side: {cam_err:.2f}"
    else:
        text = "angle/paw: None, err/side: None"
    cv2.putText(frame_line, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (192, 57, 43), 1, cv2.LINE_AA)
    text = f"speed: {speed}" # if speed < 128 else f"speed: {128 - speed}"
    cv2.putText(frame_line, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (192, 57, 43), 1, cv2.LINE_AA)
    text = now.strftime(f'line: ({line_to_follow[0]:.2f}, {line_to_follow[1]:.2f}), ({line_to_follow[2]:.2f}, {line_to_follow[3]:.2f})')
    cv2.putText(frame_line, text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (192, 57, 43), 1, cv2.LINE_AA)
    text = now.strftime(f'{flag_msg_number} %Y-%m-%d %H:%M:%S')
    cv2.putText(frame_line, text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (192, 57, 43), 1, cv2.LINE_AA)
    text = ps
    cv2.putText(frame_line, text, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (192, 57, 43), 1, cv2.LINE_AA)

    frame_name = f"frames_saved_{now.strftime('%Y%m%d%H%M%S')}.jpg"
    save_path = os.path.join(cfg["output_dir"], frame_name)
    cv2.imwrite(save_path, frame_line)

def hello():
    print()
    print(f"####################################################################")
    print(f"# Check our official GitHub / Gitee Repo to get the latest updates #")
    print(f"####################################################################")
    print()
