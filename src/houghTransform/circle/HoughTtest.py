# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import math

from HoughT import Hough_transform
from Cannyd import Canny

Path = '../../../img/Chiral-carbon.png'  # 图片路径
Save_Path = 'Test/'  # 结果保存路径
Reduced_ratio = 2  # 为了提高计算效率，将图片进行比例缩放所使用的比例值
Gaussian_kernel_size = 3
HT_high_threshold = 45
HT_low_threshold = 25
Hough_transform_step = 5
Hough_transform_threshold = 80

if __name__ == '__main__':
    start_time = time.time()

    img_gray = cv2.imread(Path, cv2.IMREAD_GRAYSCALE)
    img_RGB = cv2.imread(Path)
    y, x = img_gray.shape[0:2]  # 获取了灰度图像 img_gray 的行数和列数
    img_gray = cv2.resize(img_gray, (int(x / Reduced_ratio), int(y / Reduced_ratio)))  # 图片缩放
    img_RGB = cv2.resize(img_RGB, (int(x / Reduced_ratio), int(y / Reduced_ratio)))

    plt.subplot(131)
    plt.imshow(img_RGB)
    plt.title('img_RGB')
    plt.axis('off')

    # canny
    print('Canny ...')
    canny = Canny(Gaussian_kernel_size, img_gray, HT_high_threshold, HT_low_threshold)
    canny.canny_algorithm()
    cv2.imwrite(Save_Path + "canny_result.jpg", canny.img)

    # hough
    print('Hough ...')
    Hough = Hough_transform(canny.img, canny.angle, Hough_transform_step, Hough_transform_threshold)
    circles = Hough.Calculate()
    for circle in circles:
        cv2.circle(img_RGB, (math.ceil(circle[0]), math.ceil(circle[1])), math.ceil(circle[2]), (255, 0, 0), 2)
    cv2.imwrite(Save_Path + "hough_result.jpg", img_RGB)
    print('Finished!')

    end_time = time.time()
    print("running time" + str(end_time - start_time))

    plt.subplot(132)
    plt.imshow(canny.img)
    plt.title('canny.img')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(img_RGB)
    plt.title('img_RGB')
    plt.axis('off')

    plt.show()
