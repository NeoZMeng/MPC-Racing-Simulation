import matplotlib.pyplot as plt
import pyomo.environ as pyo
import cv2
import numpy as np
from scipy import interpolate
import json, codecs

splineMaxAngleInterpolationFactor = 0.75

startingDirTestingLen = 0.1


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

    return result


def getDirLen(point1: np.ndarray, point2: np.ndarray) -> (float, float):
    vec = point2 - point1
    dir = np.arctan(np.inf * vec[1] if vec[0] == 0 else vec[1] / vec[0])
    if vec[0] < 0:
        dir += np.pi
        if dir > np.pi:
            dir -= 2 * np.pi
    length = np.linalg.norm(vec)
    return dir, length


def convertTrack(points: np.ndarray, trackLen: float, width: float, maxDist: float, maxBend: float,
                 getClipData=False) -> np.ndarray:
    nPoints = points.shape[0] - 1
    t = [0]
    for i in range(len(points) - 1):
        t.append(np.linalg.norm(points[i + 1] - points[i]) + t[i])
    scale = trackLen / t[-1]
    t = np.asarray(t) * scale
    x, y = (points * scale).T
    splineX = interpolate.UnivariateSpline(t, x)
    splineY = interpolate.UnivariateSpline(t, y)

    clippingDist = np.tan(maxBend) * width / 2
    clippedPoints = []

    data = []
    d = 0  # distance analyzed
    dir = 0

    point1 = np.array([splineX(0), splineY(0)])
    point2 = np.array([splineX(startingDirTestingLen), splineY(startingDirTestingLen)])
    dir0, _ = getDirLen(point1, point2)
    while d < trackLen:
        tempd = maxDist
        while tempd >= clippingDist:
            point2 = np.array([splineX(d + tempd), splineY(d + tempd)])
            dir, length = getDirLen(point1, point2)
            bend = dir - dir0
            if abs(bend) > np.pi:
                bend = bend - 2 * np.pi if bend > 0 else bend + 2 * np.pi
            if abs(bend) < maxBend:
                data.append([*point1, bend, length, dir])
                break
            else:
                tempd *= maxBend / abs(bend) * splineMaxAngleInterpolationFactor
        if tempd < clippingDist:
            print("encountered clipping at: ", point1, tempd, bend)
            clippedPoints.append(point1)

        dir0 = dir
        point1 = point2
        d += tempd
    data.append(data[-1])
    if getClipData:
        return np.asarray(data), np.asarray(clippedPoints)
    else:
        return np.asarray(data)

    return data


class Track:
    def __init__(self, data: np.ndarray, width: float):
        self.data = data
        self.fixedWidth = width/2
        self.numOfSegments = len(self.data) - 1

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
    def trackFromDataFile(cls, dataFile: str):
        dataText = codecs.open(dataFile, 'r', encoding='utf-8').read()
        data, width = json.loads(dataText)
        return Track(np.asarray(data), width)

    def length(self) -> float:
        return np.sum(self.data[:, 3])

    def segment(self, start: int, length: int) -> np.ndarray:
        if start + length > len(self.data):
            beginningLen = start + length - len(self.data) + 1
            return np.vstack([self.data[start:-1], self.data[:beginningLen]])
        else:
            return self.data[start:start + length]

    def __getitem__(self, key: int):
        return self.data[key]

    def segmentCall(self, index: int) -> callable:
        def func(model, k):
            return model.track[model.start + k][index]

        return func

    def pos(self, index: int, offset: float = 0):
        if offset == 0:
            return self[index][:2]
        x, y, _, _, dir = self[index]
        return [x - offset * np.sin(dir), y + offset * np.cos(dir)]


if __name__ == "__main__":
    # creating tracks
    actualLength = 3600  # laguna seca
    actualTrackWidth = 12  # laguna seca, averaged 12m
    maxBend = 10 / 180 * np.pi

    orgPoints = originalPointsFromPic("laguna.png", 100)
    # finding the right starting location
    # plt.plot(*orgPoints.T)
    # plt.plot(*orgPoints[100], 'or')
    # plt.show()
    #
    smoothedPoints = smoothenPoints(orgPoints, 45, 5, plotComparison=False)
    # # smoothedPoints = smoothenPoints(smoothedPoints, 60, 1)
    trackData, clippedPoints = convertTrack(smoothedPoints, actualLength, actualTrackWidth, 10, maxBend,
                                            getClipData=True)
    # smoothedPoints = smoothenPoints(trackData[:,:2], 45, 1)
    # trackData, clippedPoints = convertTrack(smoothedPoints, actualLength, actualTrackWidth, 10, maxBend, getClipData=True)

    testTrack = Track(trackData, actualTrackWidth)
    x, y = trackData.T[0:2]
    # plt.plot(x, y, '.r')
    # plt.plot(*clippedPoints.T, 'ob')
    # plt.show()
    data = trackData.tolist()
    data = [data, actualTrackWidth]
    json.dump(data, codecs.open('lagunaSeca.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True,
              indent=4)

    # # testTrack = Track.trackFromPicture("images.png", 1, None, 1, maxBend)
    #
    # scale = actualLength / testTrack.length()
    # print("track length with no scaling:", testTrack.length())
    # print("scale factor:", scale)
    # # scaledTrack = Track.trackFromPicture("images.png", scale, actualTrackWidth, 10, maxBend)
    track = Track.trackFromDataFile('lagunaSeca.json')
    print(track.length())
    plt.plot(*track.data.T[:2], '.')
    plt.show()
