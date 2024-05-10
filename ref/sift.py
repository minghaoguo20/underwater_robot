import cv2
import numpy as np
import os


# 自己绘制匹配连线
def drawMatchesKnn_cv2(img1, kp1, img2, kp2, goodMatch):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1 : w1 + w2] = img2

    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]
    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
    cv2.imshow("match", vis)

sift = cv2.SIFT_create(nfeatures=100)

def get_kp_des(file_path, sift):
    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des

file_dir = "/Users/gmh/myfile/course/current/zju43_水下机器人设计/dataset/image_recg"
biology = ["octopus", "shark", "turtle"]

for bio in biology:
    print()
    for i in range(1, 9):
        file = os.path.join(file_dir, bio, f"{i}.png")
        # print(file)
        kp, des = get_kp_des(file, sift)
        print(f"bio: {bio}, i: {i}, kp: {len(kp)}, des: {des.shape}")

exit()




file1 = "/Users/gmh/myfile/course/current/zju43_水下机器人设计/dataset/image_recg/shark/1.png"
file2 = "/Users/gmh/myfile/course/current/zju43_水下机器人设计/dataset/image_recg/shark/2.png"
img1 = cv2.imread(file1)
img2 = cv2.imread(file2)
sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 采用暴力匹配
matcher = cv2.BFMatcher()
matches = matcher.knnMatch(des1, des2, k=2)  # k=2,表示寻找两个最近邻

# 采用最近邻近似匹配
# FLANN_INDEX_KDTREE = 0  # 建立FLANN匹配器的参数
# indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # 配置索引，密度树的数量为5
# searchParams = dict(checks=50)  # 指定递归次数
# matcher = cv2.FlannBasedMatcher(indexParams, searchParams)  # 建立FlannBasedMatcher对象
# matches = matcher.knnMatch(des1, des2, k=2)  # k=2,表示寻找两个最近邻

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

out_img1 = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
out_img1[:h1, :w1] = img1
out_img1[:h2, w1 : w1 + w2] = img2
out_img1 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, out_img1)

good_match = []
for m, n in matches:
    if (
        m.distance < 0.5 * n.distance
    ):  # 如果第一个邻近距离比第二个邻近距离的0.5倍小，则保留
        good_match.append(m)

out_img2 = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
out_img2[:h1, :w1] = img1
out_img2[:h2, w1 : w1 + w2] = img2
# p1 = [kp1[kpp.queryIdx] for kpp in good_match]  # kp1中挑选处的关键点
# p2 = [kp2[kpp.trainIdx] for kpp in good_match]  # kp2中挑选处的关键点
out_img2 = cv2.drawMatches(img1, kp1, img2, kp2, good_match, out_img2)
# drawMatchesKnn_cv2(img1, kp1, img2, kp2, good_match)


cv2.imshow("out_img1", out_img1)
cv2.imshow("out_img2", out_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
