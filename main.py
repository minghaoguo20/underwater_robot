import os,sys
sys.path.append("/home/rp24/.local/lib/python3.9/site-packages")
# print(sys.path)

import cv2
import os
import time
from datetime import datetime
from datetime import timedelta

from utils.line import *
from utils.gpio import *
from utils.img import *
from utils.denote import *
from utils.motion import *
from utils.iojson import *

import time

def main(cfg):
    # 创建SIFT特征提取器
    nfeatures=100
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    # 创建FLANN匹配对象
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)

    paw, side, speed = 0.0, 0.0, 0.0
    last_speed, last_paw, last_side = 0.0, 0.0, 0.0
    blank_times = 0
    flag_path = "/home/rp24/code/underwater_robot/flag.json"
    dic = json.load(open(flag_path))
    final_flag = dic["final_flag"]
    result_save_dir = cfg["result_save_dir"]

    remove_edge_x = cfg["remove_edge_x"]
    remove_edge_y = cfg["remove_edge_y"]

    red_lower_rgb, red_upper_rgb = cfg["red_green_blue"]["red"]["lower_rgb"], cfg["red_green_blue"]["red"]["upper_rgb"]
    green_lower_rgb, green_upper_rgb = cfg["red_green_blue"]["green"]["lower_rgb"], cfg["red_green_blue"]["green"]["upper_rgb"]
    blue_lower_rgb, blue_upper_rgb = cfg["red_green_blue"]["blue"]["lower_rgb"], cfg["red_green_blue"]["blue"]["upper_rgb"]
    green_blue_percentage = cfg["green_blue_percentage"]

    confidence_threshold = cfg["confidence_threshold"]

    flag_msg_number = 0
    com = serial.Serial(port="/dev/ttyAMA0", baudrate=9600, stopbits=1, bytesize=8, parity="N")

    ret, find_camera = False, False
    try_time = 10
    for camera_id in range(2):
        for i in range(try_time):
            cap = cv2.VideoCapture(camera_id)
            ret, _ = cap.read()
            if ret:
                find_camera = True
                break
        if find_camera:
            break

    if ret:
        msg = pack_lora_msg(4, 1, 0, 0)
        com.write(msg)
        print(f"msg 4 1 0 0")
    else:
        msg = pack_lora_msg(4, 2, 0, 0)
        com.write(msg)
        print(f"msg 4 2 0 0")
        denote_path = os.path.join(cfg["output_dir"], "denote_no_camera.txt")
        denote_msg(denote_path, flag_msg_number, datetime.now(), None, None, None, None)
        flag_msg_number += 1
        exit()

    last = datetime.now() - timedelta(seconds=100)

    lower_rgb = cfg["line_color"]["lower_rgb"]
    upper_rgb = cfg["line_color"]["upper_rgb"]
    speed_bias = cfg["speed_bias"]
    last_irrelative_lines_center_x_mean = [0, 0, 0]
    last_followed_lines = [None, None, None]
    last_followed_cals = [None, None, None]
    last_angles = [None, None, None]
    last_sides = [None, None, None]
    max_speed = cfg["max_speed"]

    last_following_line, last_following_cal = None, None
    prepare_to_transfer = False # whether to transfer to the next line
    figure_recg = False
    figure_task = np.zeros(6)
    next_point = 0
    denote_path = os.path.join(cfg["output_dir"], "denote.txt")
    time_denote_path = os.path.join(cfg["output_dir"], "time_denote.txt")
    time_denote = np.zeros(16)

    for speed in range(8, 0, -2):
        msg = pack_lora_msg(0, speed, 0, 0)
        com.write(msg)
        time.sleep(0.2)

    last_creature_gap_num = -100
    creature_list = [-100]
    while True:
        if final_flag:
            msg = pack_lora_msg(1, 6, 0, 0)
            for _ in range(20):
                com.write(msg)
                time.sleep(0.05)

            time.sleep(180)
            exit()

            speed = 20
            cam_angle = -15

            msg = pack_lora_msg(0, speed, cam_angle, 0)
            for _ in range(10):
                com.write(msg)
                time.sleep(0.05)

            msg = pack_lora_msg(0, speed, 0, 0)
            while True:
                com.write(msg)
                time.sleep(0.05)
            # exit()

        time_denote_i = 0 # ################# measure time #################
        start = time.time() # ################# measure time #################

        now = datetime.now()
        ret, frame = cap.read()
        height, width, _ = frame.shape
        frame = frame[remove_edge_y:height-remove_edge_y, remove_edge_x:width-remove_edge_x]
        height, width, _ = frame.shape

        task = ""
        ps = ""

        if now - last < timedelta(seconds=cfg["sample_delta"]):
            continue
        last = now

        end = time.time() # ################# measure time #################
        duration = end - start # ################# measure time #################
        time_denote[time_denote_i] += duration # ################# measure time #################
        time_denote_i += 1 # ################# measure time #################
        start = time.time() # ################# measure time #################

        # get all lines
        lines = get_lines(frame, "bgr", lower_rgb, upper_rgb)

        end = time.time() # ################# measure time #################
        duration = end - start # ################# measure time #################
        time_denote[time_denote_i] += duration # ################# measure time #################
        time_denote_i += 1 # ################# measure time #################
        start = time.time() # ################# measure time #################

        temp = np.array(last_irrelative_lines_center_x_mean)
        temp = temp[temp != np.array(None)]
        irrelative_lines_bias = np.mean(temp) if len(temp) > 0 else 0
        if irrelative_lines_bias > 50:
            bias_right = True
        elif last_side > 0:
            bias_right = True
        else:
            bias_right = False
        
        if lines is None:
            # Save frame to directory
            frame_name = f"frames_saved_{now.strftime('%Y%m%d%H%M%S')}.jpg"
            # save_path = os.path.join(cfg["output_dir"], frame_name)
            # cv2.imwrite(save_path, frame)

            # denote_msg(denote_path, flag_msg_number, now, None, None, lines, None)
            flag_msg_number += 1
            blank_times += 1
            restore = False
            if last_following_line is not None:
                if min(last_following_line[1], last_following_line[3]) > height * 0.66:
                    restore = True
            if restore:
                speed = -last_speed
                cam_angle = 0
                cam_err = -last_side
                msg = pack_lora_msg(0, encode_speed(speed), cam_angle, cam_err)
            else:
                turn_right_angle = cfg["turn_right_angle"]
                cam_angle = turn_right_angle if bias_right else -turn_right_angle
                cam_err = -20
                speed = speed_bias/2
                msg = pack_lora_msg(0, encode_speed(speed), cam_angle, cam_err)
            # msg = pack_lora_msg(0, -last_speed/2, -last_paw/2, -last_side/2)
            com.write(msg)
            save_img(frame, lines=lines, line_to_follow=None, cam_angle=cam_angle, cam_err=cam_err, speed=speed, flag_msg_number=flag_msg_number, now=now, cfg=cfg, ps="no lines")
            continue

        cal = get_center_and_angle_len(lines, w=width, h=height)

        end = time.time() # ################# measure time #################
        duration = end - start # ################# measure time #################
        time_denote[time_denote_i] += duration # ################# measure time #################
        time_denote_i += 1 # ################# measure time #################
        start = time.time() # ################# measure time #################

        # get all clear and single lines
        lines, cal = get_lines_and_cal(lines, cal, width=width, height=height, min_lines=cfg["threshold"]["recg_min_line_num"])
        if lines is None:
            denote_msg(denote_path, flag_msg_number, now, None, None, lines, None, None, None, " {line is None}")
            flag_msg_number += 1
            
            temp_angle = 20 if bias_right else -20
            msg = pack_lora_msg(0, encode_speed(speed_bias/2), temp_angle, 0)
            # msg = pack_lora_msg(0, -last_speed/2, -last_paw/2, -last_side/2)
            com.write(msg)
            continue
        if last_following_line is None:
            last_following_line, last_following_cal = get_left_line_and_cal(lines, cal)

        blank_times = 0
        xy_percent = cfg["xy_percent"]
        if in_center(last_following_line, w=width, h=height, x_percent=xy_percent, y_percent=xy_percent)[0]:
            prepare_to_transfer = True
        prepare_to_transfer = False # DELETE!!!

        # get the nearest line
        line_nearest, cal_nearesr, idx_min_dis, irrelative_lines_center_x_mean = get_line_and_cal_to_follow(
            lines, cal, last_following_line, last_following_cal
        )
        last_irrelative_lines_center_x_mean.pop(0)
        last_irrelative_lines_center_x_mean.append(irrelative_lines_center_x_mean-cal_nearesr[0])

        # by default, we follow the nearest line. we process other cases later
        line_to_follow, cal_to_follow = line_nearest, cal_nearesr

        # case: need to transfer from one line to another. if cannot find proper line, follow the line got before
        if prepare_to_transfer:
            min_dis = 100000
            lines = np.delete(lines, idx_min_dis, axis=0)
            cal = np.delete(cal, idx_min_dis, axis=0)
            finded_new_line = False
            for line, acal in zip(lines, cal):
                if in_center(line, w=width, h=height, x_percent=xy_percent, y_percent=xy_percent)[0]:
                    dis = np.square(acal[0] - last_following_cal[0]) + np.square(
                        acal[1] - last_following_cal[1]
                    )
                    if dis < min_dis:
                        min_dis = dis
                        line_to_follow = line
                        cal_to_follow = acal
                        prepare_to_transfer = False
                        finded_new_line = True
                    if figure_task[next_point] == 0:
                        res, confidence = recognize_figure(frame)
                        if confidence > 0.2:
                            figure_recg = True
                            figure_task[next_point] = 1
            if finded_new_line:
                next_point += 1
            else:
                task = "tracking"

        # prolong the line to follow
        line_to_follow_prolonged = prolong_line(line_to_follow, height=height, width=width)
        cal_to_follow_prolonged = get_center_and_angle_len(np.array([line_to_follow]), w=width, h=height)[0]

        # update
        last_following_line, last_following_cal = line_to_follow, cal_to_follow
        last_followed_lines.pop(0)
        last_followed_lines.append(line_to_follow)
        last_followed_cals.pop(0)
        last_followed_cals.append(cal_to_follow)

        y_max = max(line_to_follow[1], line_to_follow[3])
        y_min = min(line_to_follow[1], line_to_follow[3])
        x_max = max(line_to_follow[0], line_to_follow[2])
        x_min = min(line_to_follow[0], line_to_follow[2])
        down_part = True if y_min / height > 0.66 and x_max / width > 0.66 else False

        # calculate the angle and error
        cam_angle = cal_to_follow_prolonged[2] - 90
        cam_angle = confine_angle(cam_angle, -90)
        cam_angle_ng90 = confine_angle(cam_angle, -90)
        abs_angle = np.abs(cam_angle)
        if x_max < width * 0.45 and abs_angle > 20:
            cam_angle = confine_angle(cam_angle, -120)
            ps += " {x_max < width * 0.45 , angle from -120}"
        elif x_min > width * 0.55 and abs_angle > 20:
            cam_angle = confine_angle(cam_angle, -30)
            ps += " {x_min > width * 0.55 , angle from -30}"

        ecl, ps_ecl = exist_center_line(lines, cal, width, height)
        if (y_min < 100) and (cam_angle_ng90 < -15) and (cam_angle_ng90 > -45) and (x_min > width*0.5) and (flag_msg_number > 100) and (lines.shape[0] > 1) and ecl:
            final_flag = True
            dic = json.load(open(flag_path))
            dic["final_flag"] = final_flag
            json.dump(dic, open(flag_path, "w+"))
            ps += " {final flag}"
            ps += ps_ecl
        else:
            ps += "{YES y_min < 100}" if y_min < 100 else "{NOT y_min < 100}"
            ps += "{YES cam_angle_ng90 < -15 }" if cam_angle_ng90 < -15  else "{NOT cam_angle < -15 }"
            ps += "{YES cam_angle_ng90 > -45}" if cam_angle_ng90 > -45 else "{NOT cam_angle > -45}"
            ps += "{YES width*0.5}" if width*0.5 else "{NOT width*0.5}"
            ps += "{YES flag_msg_number > 100}" if flag_msg_number > 100 else "{NOT flag_msg_number > 100}"
            ps += "{YES lines.shape[0] > 1}" if lines.shape[0] > 1 else "{NOT lines.shape[0] > 1}"
            ps += "{YES exist_center_line}" if ecl else "{NOT exist_center_line}"

        # if down_part:
        #     cam_angle = confine_angle(cam_angle, 0)
        if last_angles[-1] is not None:
            if last_angles[-1] - cam_angle > 160:
                cam_angle = confine_angle(cam_angle, (last_angles[-1] + cam_angle)/2)
                ps += " {last_angles[-1] is not None last_angles[-1] - cam_angle > 160}"
            elif cam_angle - last_angles[-1] > 160:
                cam_angle = confine_angle(cam_angle, (last_angles[-1] + cam_angle)/2 - 180)
                ps += " {last_angles[-1] is not None cam_angle - last_angles[-1] > 160}"

        cam_err = (cal_to_follow_prolonged[4] - width / 2) * np.sin(
            np.pi - cal_to_follow_prolonged[2] * np.pi / 180
        )
        cam_err = cam_err / 320 * 200

        # if abs(cam_angle) < cfg["threshold"]["angle"]:
        #     cam_angle = 0
        # if abs(cam_err) < cfg["threshold"]["error"]:
        #     cam_err = 0

        # send the angle and error to the STM32
        if figure_recg and figure_task[next_point] == 0:
            # senf msg of figure recg
            pass
        else:
            sols = cal_speed(cam_angle, cam_err, line_to_follow, cal_to_follow, speed_bias, max_speed=max_speed, width=width, height=height)

            first_batch = True
            for sol in sols:
                if not first_batch:
                    time.sleep(0.2)
                speed, paw, side = sol[0], sol[1], sol[2]
                # brake
                if last_speed > 10 and speed < 5 and speed >= 0:
                    msg = pack_lora_msg(0, encode_speed(-3), 0, 0)
                    for _ in range(20):
                        com.write(msg)

                if speed < 128:
                    msg = pack_lora_msg(0, encode_speed(speed), paw, side)
                else: 
                    msg = pack_lora_msg(0, encode_speed(speed), -paw, -side)

                com.write(msg)
                first_batch = False

            last_speed, last_paw, last_side = speed, paw, side

            last_angles.pop(0)
            last_angles.append(cam_angle)
            last_sides.pop(0)
            last_sides.append(cam_err)
        
        # if ((y_max < 180) or (y_min > 360)) and (flag_msg_number - last_creature_gap_num > 100):
        if ((y_max < 180) or (y_min > 360)) and (flag_msg_number - creature_list[-1] > 100):
            res, confidence = recognize_figure(frame[120:360, 180:500], sift, flann)
            # red_percentage = reg_color(frame, red_lower_rgb, red_upper_rgb)
            green_percentage = reg_color(frame, green_lower_rgb, green_upper_rgb)
            blue_percentage = reg_color(frame, blue_lower_rgb, blue_upper_rgb)
            # green = True if green_percentage > blue_percentage else False

            reg_succeed = True
            if green_percentage > green_blue_percentage:
                msg = pack_lora_msg(1, 1, 0, 0)
                res = "green"
            elif blue_percentage > green_blue_percentage:
                msg = pack_lora_msg(1, 2, 0, 0)
                res = "blue"
            elif confidence > confidence_threshold:
                if res == "octopus":
                    creature_flag = 3
                elif res == "shark":
                    creature_flag = 4
                elif res == "turtle":
                    creature_flag = 5
                msg = pack_lora_msg(1, creature_flag, 0, 0)
            else:
                reg_succeed = False

            if reg_succeed:
                for _ in range(8):
                    com.write(msg)
                    time.sleep(0.03)
                ps += f"flag_msg_number={flag_msg_number}, flag_msg_number={flag_msg_number}"
                last_creature_gap_num = flag_msg_number
                creature_list.append(flag_msg_number)
            
            cv2.imwrite(os.path.join(result_save_dir, f"{flag_msg_number}_{res}_{now.strftime('%Y%m%d%H%M%S')}.jpg"), frame)


        # end = time.time() # ################# measure time #################
        # duration = end - start # ################# measure time #################
        # time_denote[time_denote_i] += duration # ################# measure time #################
        # time_denote_i += 1 # ################# measure time #################
        # start = time.time() # ################# measure time #################

        # Save frame to directory
        # frame_name = f"frames_saved_{now.strftime('%Y%m%d%H%M%S')}.jpg"
        # save_path = os.path.join(cfg["output_dir"], frame_name)
        # print(save_path)
        # cv2.imwrite(save_path, frame)

        save_img(frame, lines, line_to_follow, cam_angle, cam_err, speed, flag_msg_number, now, cfg)

        denote_msg(denote_path, flag_msg_number, now, cam_angle, cam_err, lines, cal, line_nearest, cal_nearesr, ps)
        flag_msg_number += 1

        if cfg["test_steps"] == 0:
            break
        cfg["test_steps"] -= 1

        end = time.time() # ################# measure time #################
        duration = end - start # ################# measure time #################
        time_denote[time_denote_i] += duration # ################# measure time #################

        time_denote_percentage = time_denote / np.sum(time_denote)
        f = open(time_denote_path, 'a+')
        for i in range(time_denote.shape[0]):
            f.write(f"{time_denote[i]}, ")
        f.write("\n")
        for i in range(time_denote.shape[0]):
            f.write(f"{time_denote_percentage[i]}, ")
        f.write("\n")
        f.close()

        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # time.sleep(15)
    dt = datetime.now()
    print(f"start @ {dt.strftime('%Y-%m-%d_%H:%M:%S')}")
    cfg = {
        "sample_delta": 0.01,
        "output_dir": f"/home/rp24/code/underwater_robot/file/debug_output_{dt.strftime('%Y%m%d_%H%M%S')}",
        "result_save_dir" : f"/home/rp24/code/underwater_robot/result",
        "test_steps": 1000,
        "xy_percent": 0.9,
        "max_speed": 25,
        "threshold": {
            "error": 20,
            "angle": 5,
            "recg_min_line_num": 2
        },
        "line_color": {
            "lower_rgb": (0, 0, 0),
            # "upper_rgb": (110, 80, 40),
            # "upper_rgb": (130, 100, 50),
            "upper_rgb": (170, 130, 70),
        },
        "red_green_blue": {
            "red": {
                "lower_rgb": (0, 0, 127),
                "upper_rgb": (50, 50, 255),
            },
            "green": {
                "lower_rgb": (120, 200, 150),
                "upper_rgb": (160, 255, 180),
            },
            "blue": {
                "lower_rgb": (180, 80, 50),
                "upper_rgb": (210, 140, 70),
            },
        },
        "green_blue_percentage": 0.04,
        "speed_bias": 8,
        "turn_right_angle": 40,
        "remove_edge_x": 40,
        "remove_edge_y": 20,
        "confidence_threshold": 0.15
    }
    output_dir = cfg["output_dir"]
    count = 1
    while os.path.exists(output_dir):
        output_dir = f"{cfg['output_dir']}_{count}"
        count += 1
    os.makedirs(output_dir)
    cfg["output_dir"] = output_dir
    
    main(cfg)
    # debug()
    # realtime()
