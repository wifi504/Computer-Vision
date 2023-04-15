# coding:utf8
import cv2
import numpy as np
from matplotlib import pyplot as plt

def GrayHist(img):
    grayHist = np.zeros(256,dtype=np.uint64)
    for v in range(256):
        grayHist[v] = np.sum(img==v)
    return grayHist

def threshTwoPeaks(image):

    # 计算灰度直方图
    hist = GrayHist(image)
    # 寻找灰度直方图的最大峰值对应的灰度值
    maxLoc = np.where(hist == np.max(hist)) #maxLoc 中存放的位置
    firstPeak = maxLoc[0][0]
    # 寻找灰度直方图的第二个峰值对应的灰度值
    elementList = np.arange(256,dtype = np.uint64)
    measureDists = np.power(elementList - firstPeak,2) * hist

    maxLoc2 = np.where(measureDists == np.max(measureDists))
    secondPeak = maxLoc2[0][0]

    # 找到两个峰值之间的最小值对应的灰度值，作为阈值
    thresh = 0
    if secondPeak > firstPeak:
        firstPeak,secondPeak=secondPeak,firstPeak
    temp = hist[secondPeak:firstPeak]
    minloc = np.where(temp == np.min(temp))
    thresh = secondPeak + minloc[0][0] + 1
    # 找到阈值之后进行阈值处理，得到二值图
    threshImage_out = image.copy()
    # 大于阈值的都设置为255
    threshImage_out[threshImage_out > thresh] = 255
    # 小于阈值的都设置为0
    threshImage_out[threshImage_out <= thresh] = 0
    return thresh, threshImage_out

if __name__ == "__main__":

    img = cv2.imread('F:/test.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    th, img_new = threshTwoPeaks(img_gray)
    th1,img_new_1 = cv2.threshold(img_gray, 0, 255,  cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    print(th,th1)
    plt.subplot(131), plt.imshow(img_gray, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(img_new, cmap='gray')
    plt.title('Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(img_new_1, cmap='gray')
    plt.title('CV2 Image1'), plt.xticks([]), plt.yticks([])
    plt.show()