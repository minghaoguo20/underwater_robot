import cv2 as cv
import numpy as np
import math
import serial
import time
import RPi.GPIO as GPIO
import segmentation as sp
import datetime


class blur(object):
    buffer = [0]
    flag = 0


class detect(object):
    flag = 0
    time = 0
    number = 0
    led = 0
    temp = 0
    buffer = np.array([0, 0, 0, 0])
    pic = 0
    cnt = 0
    length = 1000


class line_old(object):
    theta = 0
    rho = 0


def str_check(result):
    ret = 1
    if result[0] == result[1]:
        ret = 0
    elif result[0] == result[2]:
        ret = 0
    elif result[0] == result[3]:
        ret = 0
    elif result[1] == result[2]:
        ret = 0
    elif result[1] == result[3]:
        ret = 0
    elif result[2] == result[3]:
        ret = 0
    else:
        pass
    return ret


def transmit(x1, y1, x2, y2):
    if y1 >= y2:
        if x1 == x2:  # 解决arctan（）在直角的问题
            theta = np.pi / 2
        else:
            theta = math.atan((y1 - y2) / (x2 - x1))
        if y1 == h - 1:  # 直线碰到画面底部
            rho = (x1 + 1) / w
        else:
            rho = (x2 - (x2 - x1) * (h - y2) / (y1 - y2)) / w
        if theta < 0:  # theta<0修正
            theta = theta + np.pi  # theta计算完毕
        else:
            pass
    if y1 < y2:
        if x1 == x2:  # 解决arctan（）在直角的问题
            theta = np.pi / 2
        else:
            theta = math.atan((y2 - y1) / (x1 - x2))
        if y2 == h - 1:  # 直线碰到画面底部
            rho = (x2 + 1) / w
        else:
            rho = (x1 - (x1 - x2) * (h - y1) / (y2 - y1)) / w
        if theta < 0:
            theta = theta + np.pi
    return rho, theta


def str_detect(hsv, img, rho, theta, h):
    # center = (int(rho*w), h-1)
    center = (w / 2, h / 2)
    # if string.flag == 1:
    # string.number+=1
    rotate_matrix = cv.getRotationMatrix2D(
        center=center, angle=int(90 - theta / np.pi * 180), scale=1
    )
    hsv = cv.warpAffine(src=hsv, M=rotate_matrix, dsize=(w, h))
    img = cv.warpAffine(src=img, M=rotate_matrix, dsize=(w, h))
    binary1 = cv.inRange(hsv, lower_bk, upper_bk)  # 字母牌颜色（黑）
    binary2 = cv.inRange(hsv, lower_str, upper_str)  # 字母颜色（白）
    binary3 = cv.inRange(hsv, lower_bk_1, upper_bk_1)  # 字母牌颜色
    binary = cv.bitwise_or(binary1, binary2)
    binary = cv.bitwise_or(binary, binary3)  # 赛道二
    binary = cv.medianBlur(binary, 7)
    # binary[:, int(h/6)-1:int(h/6*5)-1]
    circles = cv.HoughCircles(
        binary,
        cv.HOUGH_GRADIENT,
        1,
        640,
        param1=50,
        param2=10,
        minRadius=39,
        maxRadius=70,
    )
    try:
        circles.size
    except:
        pass
    else:
        coefficientx = 1.2  # x的系数
        coefficienty = 0.5  # y的系数
        if len(circles) == 1:
            circles = np.uint16(np.around(circles))
            x = circles[0, 0, 0]
            y = circles[0, 0, 1]
            r = circles[0, 0, 2]
            if (
                x <= r + 10
                or y <= r + 10
                or x >= 640 - r - 10
                or y >= 480 - r - 10
                or r < 30
            ):
                return
            length = math.sqrt((x - 320) * (x - 320) + (y - 240) * (y - 240))
            roi = img[
                int(y - r * coefficienty * 1.3) : int(y + r * coefficienty * 0.7),
                int(x - r * coefficientx) : int(x + r * coefficientx * 1.3),
            ]
            # cv.circle(img, (x, y), r, (0, 0, 255), 2)
            roi = cv.resize(
                roi,
                (int(roi.shape[1] * 40 / roi.shape[0]), 40),
                interpolation=cv.INTER_AREA,
            )
            Characters_list = sp.Character_segmentation(roi)  # 进行字符分割
            if len(Characters_list) >= 4:  # 如果切割出来的字符块多于4个
                w_max = 0  # 图像最大宽度
                h = Characters_list[0].shape[0]  # 图像高度
                for i in range(4):  # 取切割图片的最大宽度
                    if Characters_list[i].shape[1] > w_max:
                        w_max = Characters_list[i].shape[1]
                start = 0
                if len(Characters_list) > 4:  # 排除两边干扰色块
                    for i in range(len(Characters_list)):
                        if Characters_list[i].shape[1] < 0.85 * w_max:
                            if start + 4 < len(Characters_list):
                                start += 1
                            else:
                                break
                        else:
                            print(Characters_list[i].shape[1], start)
                            break
                else:
                    pass
                result = np.empty(4, dtype=int)
                for i in range(start, start + 4):
                    bia = int(w_max * 1.5) - Characters_list[i].shape[1]
                    if bia < 0:
                        pass
                    else:
                        # 以下进行左右补全及尺寸修正，保证输出尺寸为（50，25）
                        Characters_list[i] = np.hstack(
                            (np.zeros((h, int(bia / 2)), np.uint8), Characters_list[i])
                        )
                        Characters_list[i] = np.hstack(
                            (
                                Characters_list[i],
                                np.zeros((h, bia - int(bia / 2)), np.uint8),
                            )
                        )
                    Characters_list[i] = cv.resize(
                        Characters_list[i], (25, 50), interpolation=cv.INTER_LINEAR
                    )
                    cv.imwrite(
                        str(string.cnt) + ".png", Characters_list[i]
                    )  # 在同级目录中保存图片.png
                    string.cnt += 1  # 图片编号+1
                    img_test = np.array(
                        [Characters_list[i].flatten()], dtype=np.float32
                    )  # 转化为行向量，浮点型！
                    response = svm.predict(img_test)[1]  # 用模型进行预测
                    # result[i] = num2str[int(response[0][0]) - 1]
                    result[i - start] = int(response[0][0])
                    print(num2str[int(response[0][0]) - 1], result[i - start])
                string.flag = 1
                # if length < string.length:
                string.buffer = result
                # string.length = length
                # t2 = datetime.datetime.now().strftime(ISOTIMEFORMAT)
                cv.imwrite("string.png", img)
                # cv.imwrite('str_'+t2+'_'+str(string.number)+'.png', img)     # 按当前'分-秒-保存帧的序号'来命名图片
                print(string.buffer)
            else:
                pass


def green_detect(hsv, img):
    t1 = time.perf_counter()
    if green.pic > 1:
        green.pic -= 1
        cv.imwrite(str(green.number) + "_" + str(green.pic) + ".png", img)
    if (
        t1 - green.time >= 0.3 and green.flag == 1 and green.temp > 0
    ):  # 打卡后间隔1秒再执行程序
        if green.led == 1:
            GPIO.output(3, GPIO.HIGH)
            green.led = 0
            green.temp -= 1
            green.time = t1
        else:
            GPIO.output(3, GPIO.LOW)
            green.led = 1
            green.time = t1
        if green.temp == 0:
            green.flag = 0
        else:
            pass
    if t1 - green.time >= 2.0 and green.flag == 0:  # 每间隔2秒执行一次打卡程序
        binary = cv.inRange(hsv, lower_g, upper_g)
        binary = cv.medianBlur(binary, 5)
        circles = cv.HoughCircles(
            binary,
            cv.HOUGH_GRADIENT,
            1,
            640,
            param1=50,
            param2=10,
            minRadius=40,
            maxRadius=70,
        )
        try:
            circles.size
        except:
            pass
        else:
            if len(circles) == 1:
                green.time = t1
                green.flag = 1
                green.number += 1
                print("led on")
                green.led = 1
                green.temp = green.number
                green.pic = 2
                string.flag = 2
                if green.number < 4:
                    GPIO.output(3, GPIO.LOW)
                    cv.imwrite(str(green.number) + "_" + str(green.pic) + ".png", img)
                else:
                    pass
            else:
                pass


def line_detectP(hsv, image):
    mask1 = cv.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)  # mask1灯带
    kernel1 = cv.getStructuringElement(cv.MORPH_RECT, (9, 12))
    mask1 = cv.dilate(mask1, kernel1)
    mask1 = cv.erode(mask1, kernel1)
    lines = cv.HoughLinesP(mask1, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=50)
    try:
        lines.size
    except:
        # print("detection error")
        pass
    else:
        length = len(lines)
        # print(length)
        if length <= 12:
            return
        theta = np.empty(length, dtype=float)
        rho = np.empty(length, dtype=float)
        for i in range(length):
            rho[i], theta[i] = transmit(
                lines[i, :, 0], lines[i, :, 1], lines[i, :, 2], lines[i, :, 3]
            )
        sort_theta = np.argsort(theta)  # 从小到大对theta排序
        theta = np.mean(theta[sort_theta[0 : int(length / 3)]])  # 前1/3小的值，取均值
        rho = np.mean(rho[sort_theta[0 : int(length / 3)]])
        if rho > 1:
            rho = 1
        if rho < 0:
            rho = 0
        # if theta<0.1:
        # theta = 0.1
        line.rho = rho
        line.theta = theta
        rho_s = str(round(rho, 3))
        theta_s = str(round(theta, 3))
        # print(rho_s,theta_s)
        if string.flag == 0 or string.flag == 2:
            temp = 0
            out = rho_s + " " + theta_s + " " + str(temp) + " " + stop
        else:
            str0 = str(string.buffer[0])
            str1 = str(string.buffer[1])
            str2 = str(string.buffer[2])
            str3 = str(string.buffer[3])
            out = (
                rho_s
                + " "
                + theta_s
                + " "
                + str(string.flag)
                + " "
                + str0
                + " "
                + str1
                + " "
                + str2
                + " "
                + str3
                + " "
                + stop
            )
            string.flag = 2
        print(out)
        ser.write(out.encode("utf-8"))
        # cv.imshow("result", image)


print("--------- Python line detection ---------")
GPIO.setmode(GPIO.BCM)
GPIO.setup(3, GPIO.OUT)
GPIO.output(3, GPIO.HIGH)
# f = 1
line = line_old()
green = detect()
string = detect()
ISOTIMEFORMAT = "%M_%S"
lower_hsv = np.array([0, 0, 240])  # 直线色域 0 0 245
upper_hsv = np.array([180, 255, 255])
# lower_g = np.array([10, 245, 60])      # 绿色圆饼色域
# upper_g = np.array([45, 255, 200])
lower_g = np.array([20, 170, 60])  # 绿色圆饼色域new
upper_g = np.array([45, 255, 200])
lower_bk = np.array([0, 100, 50])  # 字母牌色域
upper_bk = np.array([55, 220, 140])
lower_bk_1 = np.array([170, 100, 50])  # 字母牌色域1
upper_bk_1 = np.array([185, 220, 140])
# lower_str = np.array([10, 90, 180])   # string
# upper_str = np.array([30, 150, 240])
lower_str = np.array([10, 60, 150])  # string
upper_str = np.array([40, 150, 210])

num2str = np.array(
    [ "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",]
)
# svm = cv.ml.SVM_load("svm_str_1.mat")  # 加载模型数据
svm = None  # 加载模型数据

ser = serial.Serial("/dev/ttyAMA0", 115200)
stop = "\r\n"
fourcc = cv.VideoWriter_fourcc(*"XVID")
out = cv.VideoWriter("output0716.avi", fourcc, 30.0, (640, 480))

capture = cv.VideoCapture(-1)
capture.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)
capture.set(cv.CAP_PROP_EXPOSURE, float(0.05))
w = 640
h = 480
capture.set(cv.CAP_PROP_FRAME_WIDTH, w)  # 设置长
capture.set(cv.CAP_PROP_FRAME_HEIGHT, h)  # 设置宽
print("exposure={}".format(capture.get(15)))
print("frame rate={}".format(capture.get(5)))
t0 = time.perf_counter()
flag1 = 0
while True:
    flag1 = flag1 + 1
    if capture.isOpened() == False:
        capture.open(-1)
        capture.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)
        capture.set(cv.CAP_PROP_EXPOSURE, float(0.05))
        capture.set(cv.CAP_PROP_FRAME_WIDTH, w)  # 设置长
        capture.set(cv.CAP_PROP_FRAME_HEIGHT, h)  # 设置宽
    ret, frame = capture.read()
    if ret == True:
        # cv.imshow("input",frame)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        line_detectP(hsv, frame)
        if green.number < 4:
            green_detect(hsv, frame)
        else:
            pass
        if string.flag == 0:
            str_detect(hsv, frame, line.rho, line.theta, h)
        else:
            pass
        out.write(frame)
        # cv.imshow('frame', frame);
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
        t1 = time.perf_counter()
        if t1 - t0 >= 1:
            print("帧率=", flag1)
            flag1 = 0
            t0 = t1
    else:
        capture.open(-1)
        capture.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)
        capture.set(cv.CAP_PROP_EXPOSURE, float(0.05))
        capture.set(cv.CAP_PROP_FRAME_WIDTH, w)  # 设置长
        capture.set(cv.CAP_PROP_FRAME_HEIGHT, h)  # 设置宽
    c = cv.waitKey(10)
    if c == 27:  # ESC
        break

capture.release()
out.release()
cv.destroyAllWindows()
