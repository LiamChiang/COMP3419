from imageio import imread
from matplotlib import pyplot as plt
import numpy as np
import math
import cv2 as cv
import skimage as ski

#import image
img = imread("./sample_lab2/BinaryTestImage.jpg");
plt.imshow(img)
# plt.show()

#convert image into grey function
g_img = np.zeros((img.shape[0], img.shape[1]))
for x in range(img.shape[1]):
    for y in range(img.shape[0]):
        g_img[y, x] = 0.21267*img[y, x, 0] + 0.715160*img[y, x, 1] + 0.072169*img[y , x, 2]
#convert image into grey
plt.imshow(g_img, cmap = plt.cm.gray)
# plt.show()

# print(g_img.shape)
#convert image to binary function
bi_img = g_img >128
plt.imshow(bi_img, cmap = plt.cm.gray)
# print(bi_img.shape)
# plt.show()

bi_img = ((g_img >128) + np.zeros(g_img.shape)) * 255
plt.imshow(bi_img, cmap = plt.cm.gray)
# print(bi_img.shape)
# plt.show()

img = cv.imread("./sample_lab2/j.png", 0)
plt.imshow(img, cmap = plt.cm.gray)
plt.show()
kernel = np.ones((5,5), np.uint8)
erosion = cv.erode(img,kernel,iterations = 1)
plt.imshow(erosion, cmap = plt.cm.gray)
plt.show()
dilation = cv.dilate(img,kernel,iterations = 1)
plt.imshow(dilation, cmap = plt.cm.gray)
plt.show()

erosion = cv.erode(img, kernel, iterations = 1)
dilation = cv.dilate(img,kernel,iterations = 1)
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
plt.imshow(erosion, cmap = plt.cm.gray)
plt.show()
plt.imshow(dilation, cmap = plt.cm.gray)
plt.show()
plt.imshow(closing, cmap = plt.cm.gray)
plt.show()
plt.imshow(opening, cmap = plt.cm.gray)
plt.show()
exit()