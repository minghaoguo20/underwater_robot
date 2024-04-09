import cv2 as cv
import numpy as np

def line_detect(hsv, image, lower_hsv, upper_hsv):
    mask1 = cv.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)  # mask1灯带
    kernel1 = cv.getStructuringElement(cv.MORPH_RECT, (9, 12))
    mask1 = cv.dilate(mask1, kernel1)
    mask1 = cv.erode(mask1, kernel1)
    lines = cv.HoughLinesP(mask1, 1, np.pi/180, 100, minLineLength=100, maxLineGap=50)
    try:
        lines.size
    except:
        # print("detection error")
        pass
    else:
        length = len(lines)
        # print(length)
        if length<=12:
            return
        theta = np.empty(length, dtype=float)
        rho = np.empty(length, dtype=float)
        for i in range(length):
            rho[i], theta[i] = transmit(lines[i, :, 0], lines[i, :, 1], lines[i, :, 2], lines[i, :, 3])
        sort_theta = np.argsort(theta)      # 从小到大对theta排序
        theta = np.mean(theta[sort_theta[0:int(length/3)]])     # 前1/3小的值，取均值
        rho = np.mean(rho[sort_theta[0:int(length/3)]])
        if rho>1:
            rho=1
        if rho<0:
            rho=0
        #if theta<0.1:
            #theta = 0.1
        line.rho = rho
        line.theta = theta
        rho_s=str(round(rho,3))
        theta_s=str(round(theta,3))
        # print(rho_s,theta_s)
        if string.flag == 0 or string.flag==2:
            temp = 0
            out=rho_s+' '+theta_s+' '+str(temp)+' '+stop
        else:
            str0 = str(string.buffer[0])
            str1 = str(string.buffer[1])
            str2 = str(string.buffer[2])
            str3 = str(string.buffer[3])
            out=rho_s+' '+theta_s+' '+str(string.flag)+' '+str0+' '+str1+' '+str2+' '+str3+' '+stop
            string.flag = 2
        print(out)
        ser.write(out.encode('utf-8'))
        # cv.imshow("result", image)
