import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.interpolate as si
# Read the image and create a blank mask
# img = cv2.imread('images.png')
# h,w = img.shape[:2]
# mask = np.zeros((h,w), np.uint8)
#
# # Transform to gray colorspace and threshold the image
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#
# # Search for contours and select the biggest one and draw it on mask
# contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
# cnt = max(contours, key=cv2.contourArea)
# cv2.drawContours(mask, [cnt], 0, 255, -1)
#
# # Perform a bitwise operation
# res = cv2.bitwise_and(img, img, mask=mask)
#
#
#
# img = res
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret,threshed = cv2.threshold(gray,170,255,cv2.THRESH_BINARY)
#
# # find contours without approx
# cnts = cv2.findContours(threshed,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[-2]
#
# # get the max-area contour
# cnt = sorted(cnts, key=cv2.contourArea)[-1]
#
# # calc arclentgh
# arclen = cv2.arcLength(cnt, True)
#
# # do approx
# eps = 0.0005
# epsilon = arclen * eps
# approx = cv2.approxPolyDP(cnt, epsilon, True)
#
# result = np.array([i[0] for i in approx])
# for i in result:
#     i[1] -= 250
#     i[1] = - i[1]
# result = np.append(result, np.array(result[0]).reshape((1,2)), 0)
# plt.plot(*result.T, '.r')
# plt.show()

# x = [6, -2, 4, 6, 8, 14, 6]
# y = [-3, 2, 5, 0, 5, 2, -3]
#
# xmin, xmax = min(x), max(x)
# ymin, ymax = min(y), max(y)
#
# n = len(x)
# plotpoints = 50
#
#
# k = 3
# knotspace = range(n)
# knots = si.InterpolatedUnivariateSpline(knotspace, knotspace, k=k).get_knots()
# # knots = [0,1.5,3,4.5,6]
# knots_full = np.concatenate(([knots[0]]*k, knots, [knots[-1]]*k))
#
# tckX = knots_full, x, k
# tckY = knots_full, y, k
#
# splineX = si.UnivariateSpline._from_tck(tckX)
# splineY = si.UnivariateSpline._from_tck(tckY)
#
#
# tP = np.linspace(knotspace[0], knotspace[-1], plotpoints)
# xP = splineX(tP)
# yP = splineY(tP)
# for i in range(plotpoints):
#     xP[i] += (np.random.rand() - 0.5) * 0.5
#     yP[i] += (np.random.rand() - 0.5) * 0.5
# plt.plot(xP, yP, '.')
#
# x = xP
# y = yP
# xmin, xmax = min(x), max(x)
# ymin, ymax = min(y), max(y)
#
# n = len(x)
# plotpoints = 100
#
#
# k = 3
# knotspace = range(n)
# knots = si.InterpolatedUnivariateSpline(knotspace, knotspace, k=k).get_knots()
# # knots = [0,1.5,3,4.5,6]
# knots_full = np.concatenate(([knots[0]]*k, knots, [knots[-1]]*k))
#
# tckX = knots_full, x, k
# tckY = knots_full, y, k
# smoothness = 0.8
#
# splineX = si.UnivariateSpline(knotspace, x, s = smoothness)
# splineY = si.UnivariateSpline(knotspace, y, s = smoothness)
#
# splineX.set_smoothing_factor(smoothness)
# splineY.set_smoothing_factor(smoothness)
#
#
# tP = np.linspace(knotspace[0], knotspace[-1], plotpoints)
# xP = splineX(tP)
# yP = splineY(tP)
#
# plt.plot(xP, yP, '.-')
#
# plt.show()
a = np.array([1,2]).reshape((2,1))
b = np.array([2,3]).reshape((2,1))
c = np.concatenate([a,b],axis=1)
print(c)
print(np.flip(c, axis=0))