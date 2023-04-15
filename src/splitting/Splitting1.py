#https://blog.csdn.net/Zhaohui_Zhang/article/details/120522932

#coding:utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt
 
image = cv2.imread('F:/test.jpg')#导入图像
grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#彩色转灰度
gray = cv2.GaussianBlur(grayimage, (3,3), 0)#进行高斯滤波
 
ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  #方法选择为THRESH_OTSU
 
plt.figure(figsize=(5,5))
plt.subplot(), plt.imshow(th1, "gray")
plt.title("Otsu,threshold is " + str(ret1)), plt.xticks([]), plt.yticks([])