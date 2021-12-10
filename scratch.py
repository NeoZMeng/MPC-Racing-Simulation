import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import interpolate


def originalPointsFromPic(imgFile, startingIndex: int = 0, reverse=False) -> np.ndarray:
    img = cv2.imread(imgFile)
    # h, w = img.shape[:2]
    # mask = np.zeros((h, w), np.uint8)
    #
    # # Transform to gray colorspace and threshold the image
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #
    # # Search for contours and select the biggest one and draw it on mask
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cnt = max(contours, key=cv2.contourArea)
    # cv2.drawContours(mask, [cnt], 0, 255, -1)
    #
    # # Perform a bitwise operation
    # res = cv2.bitwise_and(img, img, mask=mask)
    #
    # img = res
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshed = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

    # find contours without approx
    cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]

    # get the max-area contour
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    # calc arclentgh
    arclen = cv2.arcLength(cnt, True)

    # do approx
    eps = 0.0005
    epsilon = arclen * eps
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    result = np.array([i[0] for i in approx], dtype=float)
    for i in result:
        i[1] -= 250
        i[1] = - i[1]
    if reverse:
        result = np.flip(result, axis=0)
    if startingIndex != 0:
        result = np.concatenate([result[startingIndex:], result[:startingIndex]], axis=0)
    result = np.append(result, np.array(result[0]).reshape((1, 2)), 0)

    return result


def smoothenPoints(points: np.ndarray, smoothness: float, detailFactor: int = 1, plotComparison=False) -> np.ndarray:
    t = [0]
    for i in range(len(points) - 1):
        t.append(np.linalg.norm(points[i + 1] - points[i]) + t[i])
    x, y = points.T
    splineX = interpolate.UnivariateSpline(t, x, s=smoothness)
    splineY = interpolate.UnivariateSpline(t, y, s=smoothness)
    tNew = []
    if detailFactor <= 1:
        tNew = t
    else:
        for i in range(len(t) - 1):
            tNew.extend(np.linspace(t[i], t[i + 1], detailFactor, endpoint=False))
        tNew.append(tNew[0])
    xNew = splineX(tNew)
    yNew = splineY(tNew)

    result = np.concatenate([xNew.reshape((len(tNew), 1)), yNew.reshape((len(tNew), 1))], axis=1)
    if plotComparison:
        plt.plot(x, y, ".")
        plt.plot(xNew, yNew)
        plt.show()

    return splineY



actualLength = 3600  # laguna seca
actualTrackWidth = 12  # laguna seca, averaged 12m
maxBend = 10 / 180 * np.pi
maxDist = 10

orgPoints = originalPointsFromPic("laguna.png", 100)


splineY = smoothenPoints(orgPoints, 45, 5, plotComparison=False)

data = [type(i) for i in splineY._data]
evalArgs = np.asarray(splineY._eval_args)
print(data)
print(evalArgs)





