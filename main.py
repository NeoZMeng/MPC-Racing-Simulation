import sys
import numpy as np
import matplotlib.pyplot as plt
from time import time

plt.style.use('seaborn-pastel')

from Car import Car
from Track import Track
from Model5 import RaceProblem
from Simulator import Simulator

if __name__ == "__main__":
    F82 = Car(m=1500, wb=2.85, P=300000, stL=0.8, bf=1.3675, h=0.46457)
    car = F82
    track = Track.trackFromDataFile('lagunaSeca.json')
    problem = RaceProblem(car, track)
    problem.tireTraction = 1
    problem.initSpeed = 55
    problem.refreshModel()
    simulator = Simulator(car, track)
    x, y, v, th = *track.pos(0, problem.x0[0]), problem.x0[1], track.direction(0, problem.x0[2])
    simulator.init(x, y, v, th)
    fig = plt.figure()
    plot = fig.add_subplot(aspect=1, xlim=(-100, 1000), ylim=(0, 1100))
    track.plotTrack(plotObj=plot)

    # enable the line below to disable soft constraints
    # problem.slack = False

    # sys inputs:
    # if no inputs: run whole lap in one solve
    # if only one input, run whole lap in multiple solve, N = input1
    # if three inputs, run partial lap, N = input1, start = input2, M = input3, M is how many solves

    if len(sys.argv) == 1:
        start = 0
        N = track.numOfSegments + 1

        t0 = time()
        problem.solveAtOnce(fastestLap=True)
        print('time spent:', time() - t0)
        problem.printLapTime()
        dt = problem.getTimeLine(get_dt=True)[1:].reshape((N, 1))
        inputs = np.concatenate([problem.uOpt, dt], axis=1)
        xSim = simulator.runBatch(inputs)
        simulator.plotTrace(xSim, plotObj=plot)
        problem.plotTrace(plotObj=plot)
        # problem.generateAnim()
        plt.show()

    else:
        N = int(sys.argv[1])
        M = track.numOfSegments + 1
        start = 0
        jump = 1
        if len(sys.argv) > 3:
            start = int(sys.argv[2])
        if len(sys.argv) > 2:
            M = int(sys.argv[3])

        t0 = time()
        try:
            problem.solveMPC(N, M, start=start, feedback=False, releaseErr=False)
        except Exception as err:
            plt.show()
            raise
        print('time spent:', time() - t0)
        problem.printLapTime()

        dt = problem.getTimeLine(get_dt=True)[1:].reshape((M, 1))
        inputs = np.concatenate([problem.uOpt, dt], axis=1)
        xSim = simulator.runBatch(inputs)
        simulator.plotTrace(xSim, plotObj=plot)
        problem.plotTrace(plotObj=plot)
        plt.show()