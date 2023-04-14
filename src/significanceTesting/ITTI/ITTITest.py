import cv2
import matplotlib.pyplot as plt
import numpy as np


def readImg(path):
    # 读取图片
    img = cv2.imread(path)
    img_cv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 调整图片大小
    resized_img = cv2.resize(img_cv, (640, 480))
    return resized_img


# 构建高斯金字塔
def build_gaussian_pyramid(image, levels):
    # 构建空元组来存储高斯金字塔图像
    pyramid = ()

    # 首先将输入图像放入金字塔元组中
    pyramid += (image,)

    # 对输入图像进行高斯滤波和下采样，生成金字塔图像
    for i in range(levels - 1):
        # 进行高斯滤波
        blur = cv2.GaussianBlur(pyramid[i], (3, 3), 0)
        # 进行下采样
        downsample = cv2.pyrDown(blur)
        # 将结果加入金字塔元组中
        pyramid += (downsample,)

    return pyramid


# 构建亮度高斯金字塔
def build_intensity_gaussian_pyramid(image, levels):
    # 将彩色图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 构建空元组来存储高斯金字塔图像
    pyramid = ()

    # 首先将灰度图像放入金字塔元组中
    pyramid += (gray,)

    # 对灰度图像进行高斯滤波和下采样，生成金字塔图像
    for i in range(levels - 1):
        # 进行高斯滤波
        blur = cv2.GaussianBlur(pyramid[i], (3, 3), 0)
        # 进行下采样
        downsample = cv2.pyrDown(blur)
        # 将结果加入金字塔元组中
        pyramid += (downsample,)

    return pyramid


# 构建颜色高斯金字塔
def build_color_gaussian_pyramid(image, levels):
    # 构建空元组来存储颜色高斯金字塔图像

    pyramid = ()

    # 对输入图像进行通道分离，生成 BGR 三个颜色通道的图像
    b, g, r = cv2.split(image)

    # 首先将红色特征放入金字塔元组中
    red_feature = r - (g + b) / 2
    pyramid += (red_feature,)

    # 然后将绿色特征放入金字塔元组中
    green_feature = g - (r + b) / 2
    pyramid += (green_feature,)

    # 接着将蓝色特征放入金字塔元组中
    blue_feature = b - (r + g) / 2
    pyramid += (blue_feature,)

    # 最后将黄色特征放入金字塔元组中
    yellow_feature = (r + g) / 2 - np.abs(r - g) / 2 - b
    pyramid += (yellow_feature,)

    # 对每个特征图像进行高斯滤波和下采样，生成颜色高斯金字塔图像
    for i in range(3):
        for j in range(3):
            # 进行高斯滤波
            for k in range(4):
                pyramid_level = cv2.GaussianBlur(pyramid[k * 4 + i * 3 + j], (3, 3), 0)
                pyramid = pyramid[:k * 4 + i * 3 + j] + (pyramid_level,) + pyramid[k * 4 + i * 3 + j + 1:]

    return pyramid[4:]  # 不需要返回红、绿、蓝、黄四个特征图像


def main():
    img = readImg(PATH)
    pyramid = build_gaussian_pyramid(img, 9)
    brightness_pyramid = build_intensity_gaussian_pyramid(img, 9)
    red_color_pyramid = build_color_gaussian_pyramid(img, 9)[0]

    # 显示 pyramid
    fig, axs = plt.subplots(3, 3)
    fig.suptitle('Color Pyramid')
    for i in range(3):
        for j in range(3):
            axs[i, j].imshow(pyramid[i * 3 + j])
    # plt.show()

    # 显示 brightness_pyramid
    fig_brightness, axs_brightness = plt.subplots(3, 3)
    fig_brightness.suptitle('Brightness Pyramid')
    for i in range(3):
        for j in range(3):
            axs_brightness[i, j].imshow(brightness_pyramid[i * 3 + j], cmap='gray')
    # plt.show()

    # 显示 red_color_pyramid
    fig_red, axs_red = plt.subplots(3, 3)
    fig_red.suptitle('Red Color Pyramid')
    for i in range(3):
        for j in range(3):
            axs_red[i, j].imshow(red_color_pyramid[i * 3 + j], cmap='gray')
    plt.show()


if __name__ == '__main__':
    PATH = "../../../img/scenery1.png"
    main()
