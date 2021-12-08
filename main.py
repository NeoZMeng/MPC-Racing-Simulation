import sys
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
from pyomo.core.base.PyomoModel import Model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('seaborn-pastel')

from Car import Car
from Track import Track
from RaceProblem import RaceProblem


def getTimeLine(trackData, N, xOpt, uOpt) -> [float]:
    result = [0]
    for k in range(N):
        b, v, th = xOpt[k]
        a, psi = uOpt[k]
        tht = trackData[k][2]
        lt = trackData[k][3]
        d = lt + (lt * th + b) * tht
        dt = d / v
        result.append(result[-1] + dt)
    return result


def getLapTime(trackData, N, xOpt, uOpt, overlapped=False, start=0, trackLen=None) -> float:
    result = 0
    sampleRange = range(N)
    if overlapped:
        sampleRange = range(trackLen - start, 2 * trackLen - start + 1)
    for k in sampleRange:
        b, v, th = xOpt[k]
        a, psi = uOpt[k]
        tht = trackData[k][2]
        lt = trackData[k][3]
        d = lt + (lt * th + b) * tht
        dt = d / v
        result += dt
    return result


def lapTimeFormat(time: float) -> str:
    minutes = int(time // 60)
    seconds = int(time % 60)
    fractions = time % 1
    fractStr = f'{fractions:.3f}'[2:]
    return f'{minutes:02d}:{seconds:02d}.{fractStr}'


if __name__ == "__main__":
    F82 = Car(m=1500, wb=2.85, P=300000, stL=0.8, bf=1.3675, h=0.46457)
    car = F82
    track = Track.trackFromDataFile('lagunaSeca.json')
    problem = RaceProblem(car, track)

    tireTraction = 1
    initialSpeed = 10
    xOpt = [[-5.5, initialSpeed, 0]]
    uOpt = []
    M = track.numOfSegments

    # sys inputs:
    # if no inputs: run whole lap in one solve
    # if only one input, run whole lap in multiple solve, N = input1
    # if two inputs, run whole lap in multiple solve, N = input1, jump = input2
    # if three inputs, run partial lap, N = input1, start = input2, M = input3, M is how many solves
    # if four inputs, treat as three inputs with additional jump = input4

    if len(sys.argv) == 1:
        start = 0
        N = track.numOfSegments + 1
        M = N
        # N = track.numOfSegments + 120
        # start = track.numOfSegments - 40
        feas, xOpt, uOpt, JOpt, model = problem.solve(N,
                                                      start=start,
                                                      x0=xOpt[-1])

        xOpt = np.asarray(xOpt)
        uOpt = np.asarray(uOpt)

    else:
        N = int(sys.argv[1])
        M = track.numOfSegments
        start = 0
        jump = 1
        if len(sys.argv) > 3:
            start = int(sys.argv[2])
            M = int(sys.argv[3])
        if len(sys.argv) == 3 or len(sys.argv) == 5:
            jump = int(sys.argv[-1])
        if jump < 1:
            print('jump should not be less than 1, procceeding with jump = 1\n')

        xOpt_ = []
        uOpt_ = []
        feas = True
        for i in range(start, start + M * jump, jump):
            stepStart = i
            try:
                feas, xOpt_, uOpt_, JOpt = problem.solve(N,
                                                         start=stepStart,
                                                         x0=xOpt[-1],
                                                         xinf=[None, None, 0])
            except ValueError as err:
                # use old prediction
                feas, xOpt_, uOpt_, JOpt = problem.solve(N,
                                                         start=stepStart,
                                                         x0=xOpt[-1],
                                                         xinf=[None, None, 0],
                                                         tee=True)

                feas = False

            if not feas:
                # xOpt will change if infeasible
                xOpt_ = xOpt_[1:, :]
                uOpt_ = uOpt_[1:, :]
            xOpt.extend(xOpt_[1:jump + 1])
            uOpt.extend(uOpt_[0:jump])
            print(i, feas)

        xOpt = np.asarray(xOpt)
        uOpt = np.asarray(uOpt)

    M *= jump

    carPos = np.array([track.pos(start + k, xOpt[k, 0]) for k in range(M)])
    carHeading = np.array([xOpt[k][2] + track.direction(k) for k in range(M)])
    carInput = uOpt

    # time spent
    t = getTimeLine(track.segment(start, M), M, xOpt, uOpt)
    print("lap time: ", lapTimeFormat(t[-1]))

    # lap time calculation
    # t = getLapTime(track.segment(0, N), N, xOpt, uOpt, overlapped=True, start=start, trackLen=track.numOfSegments)
    # print("lap time: ", lapTimeFormat(t))

    fig = plt.figure()
    plot = fig.add_subplot(aspect=1, xlim=(0, 1000), ylim=(0, 1000))
    trackAsGraph = track.trackAsGraph()
    plot.plot(*trackAsGraph[0].T, '-r', linewidth=0.3)
    plot.plot(*trackAsGraph[1].T, '-r', linewidth=0.3)
    plot.plot(*carPos.T, '-b', linewidth=0.6)

    plt.show()

    # adding animation
    # carSize = 5
    # car, = plot.plot([], [], '-b')
    # interval = 20  # milliseconds
    # speedUp = 5
    # dtFrame = interval * speedUp / 1000
    # animIndex = []
    # for k in range(M):
    #     animIndex.extend([k] * int((t[k+1] / dtFrame) - len(animIndex)))
    #
    #
    # def init():
    #     car.set_data([], [])
    #     return car,
    #
    #
    # def animate(i):
    #     k = animIndex[i]
    #     x = carPos[k][0]
    #     y = carPos[k][1]
    #     th = carHeading[k]
    #     carAnimX = [x + carSize * np.cos(th),
    #                 x + carSize * np.cos(th + np.pi * 3 / 4) / 2,
    #                 x + carSize * np.cos(th - np.pi * 3 / 4) / 2,
    #                 x + carSize * np.cos(th)]
    #     carAnimY = [y + carSize * np.sin(th),
    #                 y + carSize * np.sin(th + np.pi * 3 / 4) / 2,
    #                 y + carSize * np.sin(th - np.pi * 3 / 4) / 2,
    #                 y + carSize * np.sin(th)]
    #     car.set_data(carAnimX, carAnimY)
    #     return car,
    #
    #
    # anim = FuncAnimation(fig, animate, init_func=init, frames=int(t[-1] / dtFrame), interval=interval,
    #                      blit=True)
    # anim.save('track.gif', writer='ffmpeg')
    # plt.show()
