import matplotlib.pyplot as plt
import pyomo.environ as pyo
import cv2
import numpy as np
from scipy import interpolate

splineMaxAngleInterpolationFactor = 0.9


def originalPointsFromPic(imgFile, startingIndex: int = 0, reverse = False) -> np.ndarray:
    img = cv2.imread(imgFile)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    # Transform to gray colorspace and threshold the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Search for contours and select the biggest one and draw it on mask
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [cnt], 0, 255, -1)

    # Perform a bitwise operation
    res = cv2.bitwise_and(img, img, mask=mask)

    img = res
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
    result = np.append(result, np.array(result[0]).reshape((1, 2)), 0)
    if reverse:
        pass
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

    return result


def convertTrack(points: np.ndarray, length: float, width: float, maxDist: float, maxBend: float) -> np.ndarray:
    nPoints = points.shape[0] - 1

    clippingDist = np.tan(maxBend) * width / 2

    t = [0]
    for i in range(len(points) - 1):
        t.append(np.linalg.norm(points[i + 1] - points[i]) + t[i])
    scale = length / t[-1]
    t *= scale
    x, y = (points * scale).T
    splineX = interpolate.UnivariateSpline(t, x)
    splineY = interpolate.UnivariateSpline(t, y)
    data = []
    d = 0 # distance analyzed
    while d < length:
        break



    # data & rawData is a list of track sample points
    # data: [[x, y, direction, length]]
    # for i in range(nPoints):
    #     vec = np.array([x[i + 1] - x[i], y[i + 1] - y[i]])
    #     dir = np.arctan(vec[1] / vec[0] if vec[0] != 0 else np.inf)
    #     length = np.linalg.norm(vec)
    #     rawData.append([*scaledPoints[i], dir, length])
    # rawData = np.asarray(rawData)
    #
    # return rawData
    return data


class Track:
    def __init__(self, data: np.ndarray, width: float):
        self.data = data
        self.fixedWidth = width

    @classmethod
    def trackFromPicture(cls,
                         imgFile: str,
                         length: float,
                         width: float,
                         maxDist: float,
                         maxBend: float
                         ) -> "Track":
        orgPoints = originalPointsFromPic(imgFile)
        smoothedPoints = smoothenPoints(orgPoints, 3)
        return Track(convertTrack(smoothedPoints, length, width, maxDist, maxBend), width)

    @classmethod
    def trackFromDataFile(cls, dataFile: str, width):
        pass

    def length(self) -> float:
        return np.sum(self.data[:, 3])


if __name__ == "__main__":
    actualLength = 3602.  # laguna seca
    actualTrackWidth = 12.  # laguna seca, averaged 12m
    maxBend = 5 / 180 * np.pi

    orgPoints = originalPointsFromPic("images.png")
    # finding the right starting location
    # plt.plot(*orgPoints.T)
    # plt.plot(*orgPoints[125], 'or')
    # plt.show()

    smoothedPoints = smoothenPoints(orgPoints, 30, 5, True)

    testTrack = Track(convertTrack(smoothedPoints, actualLength, actualTrackWidth, 10, maxBend), actualTrackWidth)
    # # testTrack = Track.trackFromPicture("images.png", 1, None, 1, maxBend)
    #
    # scale = actualLength / testTrack.length()
    # print("track length with no scaling:", testTrack.length())
    # print("scale factor:", scale)
    # # scaledTrack = Track.trackFromPicture("images.png", scale, actualTrackWidth, 10, maxBend)
