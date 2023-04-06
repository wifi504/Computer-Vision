import cv2
import numpy as np
import matplotlib.pyplot as plt

Path = '../../../img/house.png'
img = cv2.imread(Path)
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
new_img = rgb_img.copy()
edges = cv2.Canny(grey_img, 150, 200, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 150, minLineLength=200, maxLineGap=20)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(new_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
plt.imshow(new_img)
plt.show()
