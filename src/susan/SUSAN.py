##https://blog.csdn.net/xiaohuolong1827/article/details/123698593

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cm


def get_susan_mask():
    mask = np.ones((7, 7))
    mask[0, 0] = 0
    mask[0, 1] = 0
    mask[0, 5] = 0
    mask[0, 6] = 0
    mask[1, 0] = 0
    mask[1, 6] = 0
    mask[5, 0] = 0
    mask[5, 6] = 0
    mask[6, 0] = 0
    mask[6, 1] = 0
    mask[6, 5] = 0
    mask[6, 6] = 0
    return mask


def susan_corner_detect(img_src, t=10):
    susan_mask = get_susan_mask()
    img = img_src.copy()
    row_s, col_s = 3, 3
    row_e, col_e = img_src.shape[0] - 3, img.shape[1] - 3
    n_max = 0
    n_arr = 37 * np.ones(img.shape)  # 初始认为没有角点
    for r in range(row_s, row_e):  # 遍历所有行
        for c in range(col_s, col_e):  # 遍历所有列
            susan_zone = img[r - 3:r + 3 + 1, c - 3:c + 3 + 1]  # 获取矩形区域
            susan_zone = susan_zone[susan_mask != 0]  # 使用mask截取圆形区域
            r0 = img[r, c]
            similarity = np.exp(-((1.0 * susan_zone - r0) / t) ** 6)
            n = np.sum(similarity)
            if n > n_max:
                n_max = n
            n_arr[r, c] = n
    g = n_max / 2
    R = np.zeros(img.shape)
    index = n_arr < g  # 小于g，认为是可能的角点，越小，可能性越大
    R[index] = g - n_arr[index]  # 取反，所以R越大，是角点的可能性越大

    plt.figure()
    plt.title("edge")
    plt.imshow((6.37 * n_arr).astype(np.uint8), cmap=cm.gray)
    return R


def gravity_filter(img_src, corner_src, t=10, F=1.5):
    x_label = np.zeros((7, 7))
    y_label = np.zeros((7, 7))
    x_label[:, 0] = -3
    x_label[:, 1] = -2
    x_label[:, 2] = -1
    x_label[:, -1] = 3
    x_label[:, -2] = 2
    x_label[:, -3] = 1
    y_label[0, :] = -3
    y_label[1, :] = -2
    y_label[2, :] = -1
    y_label[4, :] = 1
    y_label[5, :] = 2
    y_label[6, :] = 3
    print(x_label, "\r\n", y_label)  # 查看矩形区域内x、y轴信息

    img = img_src.copy()
    row_s, col_s = 3, 3
    row_e, col_e = img_src.shape[0] - 3, img.shape[1] - 3
    corner = corner_src.copy()
    susan_mask = get_susan_mask()
    for r in range(row_s, row_e):
        for c in range(col_s, col_e):
            if corner[r, c] == 0:  # 对于不是角点的位置，就没必要进行后续计算了
                continue
            susan_zone = img[r - 3:r + 3 + 1, c - 3:c + 3 + 1]  # 获取矩形区域
            r0 = img[r, c]
            similarity = np.exp(-((1.0 * susan_zone - r0) / t) ** 6)
            g_x = np.sum(similarity[susan_mask == 1] * x_label[susan_mask == 1]) / np.sum(
                similarity[susan_mask == 1])  # 使用mask截取圆形区域
            g_y = np.sum(similarity[susan_mask == 1] * y_label[susan_mask == 1]) / np.sum(
                similarity[susan_mask == 1])  # 使用mask截取圆形区域
            distance = np.sqrt(g_x ** 2 + g_y ** 2)
            if distance < F:
                corner[r, c] = 0
    return corner


def corner_nms(corner, kernal=3):
    out = corner.copy()
    row_s = int(kernal / 2)
    row_e = out.shape[0] - int(kernal / 2)
    col_s, col_e = int(kernal / 2), out.shape[1] - int(kernal / 2)
    for r in range(row_s, row_e):
        for c in range(col_s, col_e):
            if corner[r, c] == 0:  # 不是可能的角点
                continue
            zone = corner[r - int(kernal / 2):r + int(kernal / 2) + 1, c - int(kernal / 2):c + int(kernal / 2) + 1]
            index = corner[r, c] < zone
            (x, y) = np.where(index == True)
            if len(x) > 0:  # 说明corner[r,c]不是最大，直接归零将其抑制
                out[r, c] = 0
            else:
                out[r, c] = 255
    return out


if __name__ == '__main__':
    img_src = cv2.imread('../../img/Eiffel Tower.png', -1)
    if len(img_src.shape) == 3:
        img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    corner = susan_corner_detect(img_src)

    img_show = cv2.cvtColor(img_src, cv2.COLOR_GRAY2BGR)
    img_show[corner != 0] = (255, 0, 0)
    plt.figure()
    plt.title("original corners")
    plt.imshow(img_show, cmap=cm.gray)

    cor_g = gravity_filter(img_src, corner)
    img_show2 = cv2.cvtColor(img_src, cv2.COLOR_GRAY2BGR)
    img_show2[cor_g != 0] = (255, 0, 0)
    plt.figure()
    plt.title("corners-gravity ")
    plt.imshow(img_show2, cmap=cm.gray)

    cor_g_nms = corner_nms(cor_g)
    img_show3 = cv2.cvtColor(img_src, cv2.COLOR_GRAY2BGR)
    img_show3[cor_g_nms != 0] = (255, 0, 0)
    plt.figure()
    plt.title("corners-gravity-nms ")
    plt.imshow(img_show3, cmap=cm.gray)
    plt.show()
