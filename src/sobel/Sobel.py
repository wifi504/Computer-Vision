import numpy as np
import cv2
img = cv2.imread('./img/test1.png', 0)
mask_x = cv2.Sobel(img,cv2.CV_64F, 1, 0) # 计算x方向梯度
mask_y = cv2.Sobel(img,cv2.CV_64F, 0, 1)
img_x = cv2.convertScaleAbs(mask_x)
img_y = cv2.convertScaleAbs(mask_y)
mask = cv2.addWeighted(img_x, 0.5, img_y, 0.5, 0)  # 按权相加
Archie = cv2.resize(mask, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)
cv2.imshow('Archie', Archie)
cv2.waitKey(0)
cv2.destroyAllWindows()
