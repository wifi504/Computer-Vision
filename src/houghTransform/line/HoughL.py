import cv2
import numpy as np
import matplotlib.pyplot as plt

Path = '../../../img/house.png'
img = cv2.imread(Path)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 180)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
new_img = rgb_img.copy()
print(lines)
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(new_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
plt.imshow(new_img)
plt.show()
