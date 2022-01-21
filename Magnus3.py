import cv2
import imageio
import numpy as np
from matplotlib import pyplot as plt


source = cv2.imread('01.jpeg')
height, width, channels = source.shape
blank_image = np.zeros((height, width, channels), np.uint8)

# Filters to denoise the image
median = cv2.medianBlur(source, 5)
denoise = cv2.GaussianBlur(median, (5, 5), 0)

# ROI if needed
roi = denoise  #[360:590, 6:1250]  # y:y+h, x:x+w
r, g, b = cv2.split(roi)[:3]  # splitting in RGB

# Border detection
canny = cv2.Canny(r, 30, 300)  # for border detection
contours = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  # , hierarchy instead of [0]
contorno = np.vstack(contours).squeeze()  # stack them together followed by squeeze to remove redundant axis.

# HSV
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
lower_red = np.array([8, 30, 130])
upper_red = np.array([20, 90, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)  # Threshold the HSV image to get only blue colors

# Bitwise-AND mask and original image
res = cv2.bitwise_and(roi, roi, mask=mask)

# finding vertex with Shi Tomasi and good features to track
corners = cv2.goodFeaturesToTrack(r, 4, 0.01, 10)  # set to find maximum 4 vertex
corners = np.int0(corners)

corns = []  # save all vertex and draw these
k = 0
for i in corners:
    x, y = i.ravel()
    cv2.circle(blank_image, (x, y), 2, [255, 120, 255], -1)
    corns.append((x, y))
    k = k + 1  # 0 BR - 1 TL - 2 TR - 3 BL

print(f'{k} vertex\n BottomRight: {corns[0]}\n TopLeft: {corns[1]}\n TopRight: {corns[2]}\n BottomLeft: {corns[3]}')
projection = (corns[0][0], corns[3][1])
print(f'Vertex to join with a red line {corns[3]} {corns[0]}, projection in a white line: {projection}')
# real_width = np.linalg.norm(corns[1] - corns[3])
# real_length = np.linalg.norm(cornsp[3] - corns[0])

# Drawings what what ?
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    # cv2.rectangle(blank_image, (x, y), (x+w, y+h), (0, 255, 0), 1)  # bounding box

cv2.drawContours(roi, contours, -1, (255, 255, 255), 1)
cv2.line(blank_image, corns[3], corns[0], (0, 0, 255), thickness=1, lineType=8)
# cv2.line(blank_image, corns[3], projection, (0, 255, 255), thickness=1, lineType=8)  # projection from Bottom Left

bottom_profile = []
for i in range(contorno.shape[0]):
    if contorno[i][1] >= corns[3][1]:
        bottom_profile.append(list(contorno[i]))
    if contorno[i][0] >= corns[0][0]:
        break

bottom_profile = np.array(bottom_profile, np.int32)
bottom_profile = bottom_profile.reshape((-1, 1, 2))
cv2.drawContours(blank_image, bottom_profile, -1, (0, 255, 255), 1)  # [bottom_profile] to draw

cv2.imshow('', blank_image)
cv2.waitKey()
