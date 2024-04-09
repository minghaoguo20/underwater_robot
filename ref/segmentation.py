encoding = 'utf-8'
import cv2 as cv
import numpy as np

"""接下来进行精确定位"""
def Precise_positioning(License):
    (high, width) = License.shape[:2]

    """
    # 此处进行对视场畸变的恢复

    points1 = np.float32([ [30,30], [10,40], [40,10], [5,15] ])
    points2 = np.float32([ [0,0], [h,0], [0,w], [h,w] ])

    # 变换矩阵
    mat = cv.getPerspectiveTransform(points2, points1)
    # 投影变换
    lic = cv.warpPerspective(License, mat, (440, 140))
    cv.imshow("lic",lic)
    """

    gray = cv.cvtColor(License, cv.COLOR_BGR2GRAY)

    """进行旋转调平"""
    # 获取结构元素
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    # 顶帽操作
    gray = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)
    ret, Threshold = cv.threshold(gray, 180, 255, cv.THRESH_BINARY)

    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    # edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=2)
    # edges = cv.morphologyEx(edges, cv.MORPH_OPEN, kernel, iterations=2)
    cv.imshow("edges", edges)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=20, maxLineGap=50)  # 找车牌的边缘

    # 找出最长的那根线
    MAXline = None
    MAXline_value = 0
    if lines is not None:
        # print("有找到线，在旋转了")
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if (x2 - x1) ** 2 + (y2 - y1) ** 2 > MAXline_value:
                MAXline_value = (x2 - x1) ** 2 + (y2 - y1) ** 2
                MAXline = line
            # cv.line(License, (x1, y1), (x2, y2), (0, 0, 255), 2)

        x1, y1, x2, y2 = MAXline[0]
        angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
        M = cv.getRotationMatrix2D((width // 2, high // 2), angle, 1.0)
        License = cv.warpAffine(License, M, (width, high))
    # cv.imshow("License1",License)

    gray = cv.cvtColor(License, cv.COLOR_BGR2GRAY)

    License_pos = cv.Canny(gray, 50, 150, apertureSize=3)
    # cv.imshow("License_pos", License_pos)
    lines = cv.HoughLinesP(License_pos, 1.0, np.pi / 180, 100, minLineLength=100, maxLineGap=30)  # 找车牌的边缘
    # 找出最长的那根线
    MAXline = None
    MAXline_value = 0
    yUP = 0
    yDown = 200
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if y1 > yUP and y1 < high / 2:
                yUP = y1
            if y1 < yDown and y1 > high / 2:
                yDown = y1
            # cv.line(License, (x1, y1), (x2, y2), (0, 0, 255), 2)

    License = License[yUP:yDown, :]

    return License


"""进行字符分割"""
def Character_segmentation(License):
    # 先进行预处理
    gray = cv.cvtColor(License, cv.COLOR_BGR2GRAY)

    ret, Threshold = cv.threshold(gray, 120, 255, cv.THRESH_BINARY)
    # 闭运算
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
    # Threshold = cv.morphologyEx(Threshold, cv.MORPH_CLOSE, kernel, iterations=2)
    # cv.imshow("Threshold", Threshold)

    white = []  # 记录每一列的白色像素总和
    black = []  # ..........黑色.......
    height = Threshold.shape[0]
    width = Threshold.shape[1]
    white_max = 0
    black_max = 0
    # 计算每一列的黑白色像素总和
    for i in range(width):
        s = 0  # 这一列白色总数
        t = 0  # 这一列黑色总数
        for j in range(height):
            if Threshold[j][i] == 255:
                s += 1
            if Threshold[j][i] == 0:
                t += 1
        white_max = max(white_max, s)
        black_max = max(black_max, t)
        white.append(s)
        black.append(t)
        # print(s)
        # print(t)

    # 分割图像
    Characters_list=[]
    white_thresholdL = 2   #下阈值，根据需要进行调整
    white_thresholdU = 31  # 上阈值，根据需要进行调整

    def if_continuity(A, B):  # 判断两列是否连续
        count = 0
        for i in range(len(A)):
            if (abs(A[i] - B[i]) == 0) and A[i] == 255 and B[i] == 255:  # 这表示两个列上的相邻两点都有值
                count += 1
        if count > 0:  # 两列有1个点连在一起就算有连续的
            out = 1
        else:
            out = 0
        return out

    def find_end(start_):  # 找到一个字母的结尾
        end_ = start_ + 1
        for m in range(start_ + 1, width - 1):
            # 如果超范围了，或者不连续了，就算找到了
            if (white[m] < white_thresholdL) or (white[m] > white_thresholdU) or \
                    (if_continuity(Threshold[1:height, m], Threshold[1:height, m - 1]) == 0):
                end_ = m
                break
        return end_

    n = 1
    start = 1
    end = 2
    while n < width - 2:
        n += 1
        # 如果这一列的白点数量处于合理区间，就认为是开始有字母了
        if (white[n] > white_thresholdL) and (white[n] < white_thresholdU):
            start = n
            end = find_end(start)
            n = end
            if end - start > 2:  # 这个值如果太小，较细的字母就识别不出来，过大就有可能误识别
                # cj = License[1:height, start:end]
                Characters_list.append(Threshold[1:height, start:end])
    return Characters_list


"""显示分割好的字符"""
def Show_Character(Characters_list):
    if len(Characters_list)==0:
        return
    Show=Characters_list[0]
    for i in range(1,len(Characters_list)):
        # Show.append(Characters_list[i])

        Shape=Characters_list[i].shape
        Show= np.hstack((Show, np.ones(Shape, np.uint8)*0, Characters_list[i]))
    cv.imshow("Split", Show)


"""进行字符识别"""
#def Character_recognition(License):
    #cv.imshow("Licensess", License)
    #gray = cv.cvtColor(License, cv.COLOR_BGR2GRAY)
    #ret, temp = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
    #cv.imshow("temp", temp)
    #it = Image.fromarray(cv.cvtColor(temp, cv.COLOR_GRAY2RGB))
    # code = pytesseract.image_to_string(it, config='-l chi_sim+eng --psm 6 --oem 3')
    # code = pytesseract.image_to_string(it, config='--psm 6')
    # code = pytesseract.image_to_string(it)
    #return code
