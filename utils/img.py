import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import math
import h5py


def getMatchNum(matches, ratio):
    """返回特征点匹配数量和匹配掩码"""
    matchesMask = [[0, 0] for i in range(len(matches))]
    matchNum = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio * n.distance:  # 将距离比率小于ratio的匹配点删选出来
            matchesMask[i] = [1, 0]
            matchNum += 1
    return (matchNum, matchesMask)


def get_res(queryPath, sampleFrame, nfeatures, des_h5, use_des_h5=True):
    # comparisonImageList = []  # 记录比较结果
    resultList = []

    # 创建SIFT特征提取器
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    # 创建FLANN匹配对象
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)

    des2_save = np.zeros((3, 8, nfeatures, 128))
    class_types = ["octopus", "shark", "turtle"]

    if sampleFrame is str:
        sampleImage = cv2.imread(sampleFrame, 0)
    else:
        sampleImage = sampleFrame
    kp1, des1 = sift.detectAndCompute(sampleImage, None)  # 提取样本图片的特征
    for parent, dirnames, filenames in os.walk(queryPath):
        # parent: '/Users/gmh/myfile/course/current/zju43_水下机器人设计/dataset/image_recg/all_figure_wo_color/'
        # dirnames: []
        # filenames: ['turtle8.png', 'shark8.png', '.DS_Store', 'octopus5.png', 'octopus4.png', 'octopus6.png', 'octopus7.png', 'octopus3.png', 'octopus2.png', 'octopus1.png', 'octopus8.png', 'turtle1.png', 'shark1.png', 'shark3.png', 'turtle2.png', 'turtle3.png', 'shark2.png', 'shark6.png', 'turtle7.png', 'turtle6.png', 'shark7.png', 'shark5.png', 'turtle4.png', 'turtle5.png', 'shark4.png']
        for p in filenames:
            if p == ".DS_Store":
                continue
            # print(f"p: {p}", end=", ")
            datatype = p.split(".")[0][:-1]
            dataidx = int(p.split(".")[0][-1])
            dataname = p.split(".")[0]
            p = queryPath + p
            queryImage = cv2.imread(p, 0)
            # print(f"queryImage: {queryImage.shape}", end=", ")
            if use_des_h5:
                # kp2, des2 = sift.detectAndCompute(queryImage, None)  # 提取比对图片的特征
                # print(f"saving {dataname}")
                # with h5py.File(os.path.join(des_h5, f"{dataname}.h5"), 'w') as hf:
                #     hf.create_dataset("des", data=des2)

                with h5py.File(os.path.join(des_h5, f"{dataname}.h5"), "r") as hf:
                    des2 = hf["des"][:]
            else:
                kp2, des2 = sift.detectAndCompute(
                    queryImage, None
                )  # 提取比对图片的特征
            # print(f"des1: {des1.shape}, des2: {des2.shape}")
            if des2 is None:
                continue
                # des2 = np.zeros((nfeatures, 128))
            matches = flann.knnMatch(
                des1, des2, k=2
            )  # 匹配特征点，为了删选匹配点，指定k为2，这样对样本图的每个特征点，返回两个匹配
            (matchNum, matchesMask) = getMatchNum(
                matches, 0.9
            )  # 通过比率条件，计算出匹配程度
            matchRatio = matchNum * 100 / len(matches)
            # drawParams = dict(
            #     matchColor=(0, 255, 0),
            #     singlePointColor=(255, 0, 0),
            #     matchesMask=matchesMask,
            #     flags=0,
            # )
            # comparisonImage = cv2.drawMatchesKnn(
            #     sampleImage, kp1, queryImage, kp2, matches, None, **drawParams
            # )
            # comparisonImageList.append((comparisonImage, matchRatio))  # 记录下结果
            resultList.append((datatype, dataname, matchRatio))

    # comparisonImageList.sort(key=lambda x: x[1], reverse=True)  # 按照匹配度排序
    result = {
        "octopus": [mr[2] for mr in resultList if mr[0] == "octopus"],
        "shark": [mr[2] for mr in resultList if mr[0] == "shark"],
        "turtle": [mr[2] for mr in resultList if mr[0] == "turtle"],
    }
    result_avg = {
        "octopus": sum(result["octopus"]) / len(result["octopus"]),
        "shark": sum(result["shark"]) / len(result["shark"]),
        "turtle": sum(result["turtle"]) / len(result["turtle"]),
    }
    result_avg = sorted(result_avg.items(), key=lambda x: x[1], reverse=True)
    res = result_avg[0][0]
    # count = len(comparisonImageList)
    # column = 4
    # row = math.ceil(count / column)
    # # 绘图显示
    # figure, ax = plt.subplots(row, column)
    # for index, (image, ratio) in enumerate(comparisonImageList):
    #     ax[int(index / column)][index % column].set_title("Similiarity %.2f%%" % ratio)
    #     ax[int(index / column)][index % column].imshow(image)
    # plt.show()
    return res, result_avg


def recognize_figure(
    frame,
    queryPath="/Users/gmh/myfile/course/current/zju43_水下机器人设计/dataset/image_recg/all_figure_wo_color/",
    des_h5="/Users/gmh/oasis/code/course/underwater_robot/des_dataset",
    nfeatures=100,
):
    res, result_avg = get_res(queryPath, frame, nfeatures, des_h5)
    return res, result_avg[0][1]


def main():
    queryPath = "/Users/gmh/myfile/course/current/zju43_水下机器人设计/dataset/image_recg/all_figure_wo_color/"  # 图库路径
    samplePath = "/Users/gmh/myfile/course/current/zju43_水下机器人设计/dataset/image_recg/test/IMG_9320.JPG"  # 样本图片
    des_h5 = "/Users/gmh/oasis/code/course/underwater_robot/des_dataset"
    nfeatures = 100
    res = get_res(queryPath, samplePath, nfeatures, des_h5)
    print(res)


if __name__ == "__main__":
    main()
