import cv2
import numpy as np
import matplotlib.pyplot as plt

Path = '../../../img/bubble.png'
img = cv2.imread(Path)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
new_img = rgb_img.copy()
circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=20, maxRadius=90)
print(circles)
circles = np.uint16(np.around(circles))
print(circles)
for i in circles[0, :]:
    cv2.circle(new_img, (i[0], i[1]), i[2], (255, 0, 0), 10)
    # 圆心
    cv2.circle(new_img, (i[0], i[1]), 2, (255, 0, 0), 10)
plt.imshow(new_img)
plt.show()
