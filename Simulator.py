import numpy as np
import matplotlib.pyplot as plt


class Simulator:
    def __init__(self, car, track):
        self.car = car
        self.track = track
        self.x = 0
        self.y = 0
        self.v = 0
        self.th = 0

    def init(self, x, y, v, th):
        self.x = x
        self.y = y
        self.v = v
        self.th = th

    def run(self, a, psi, dt):
        x, y, v, th = self.x, self.y, self.v, self.th
        r = self.car.wb / np.tan(psi)
        d = a * dt * dt / 2 + v * dt
        dth = d / r

        # pos
        ldif = 2 * r * np.sin(dth / 2)
        self.x = x + ldif * np.cos(th + dth / 2)
        self.y = y + ldif * np.sin(th + dth / 2)

        # velocity
        self.v = v + a * dt

        # heading
        self.th = th + dth
        return [self.x, self.y, self.v, self.th]

    def runBatch(self, inputs):
        outputs = [[self.x, self.y, self.v, self.th]]
        for i in inputs:
            outputs.append(self.run(*i))
        return np.asarray(outputs)

    def plotTrace(self, outputs, plotObj=None, start: int = 0):
        plot = plotObj
        N = outputs.shape[0]
        carPos = outputs[:, :2]
        if plotObj is None:
            fig = plt.figure()
            plot = fig.add_subplot(aspect=1, xlim=(0, 1000), ylim=(0, 1000))
            self.track.plotTrack(plotObj=plot)
        plot.plot(*carPos.T, '-g', linewidth=0.6)
        if plotObj is None:
            plt.show()

