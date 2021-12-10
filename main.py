import sys
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
from pyomo.core.base.PyomoModel import Model
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-pastel')

from Car import Car
from Track import Track
from Model2 import RaceProblem


if __name__ == "__main__":
    F82 = Car(m=1500, wb=2.85, P=300000, stL=0.8, bf=1.3675, h=0.46457)
    car = F82
    track = Track.trackFromDataFile('lagunaSeca.json')
    problem = RaceProblem(car, track)
    problem.tireTraction = 1
    problem.initSpeed = 55
    problem.refreshModel()



    # sys inputs:
    # if no inputs: run whole lap in one solve
    # if only one input, run whole lap in multiple solve, N = input1
    # if three inputs, run partial lap, N = input1, start = input2, M = input3, M is how many solves

    if len(sys.argv) == 1:
        start = 0
        N = track.numOfSegments + 1
        # N = track.numOfSegments + 120
        # start = track.numOfSegments - 40
        problem.solveAtOnce()
        problem.printLapTime()
        problem.plotTrace()
        # problem.generateAnim()

    else:
        N = int(sys.argv[1])
        M = track.numOfSegments
        start = 0
        jump = 1
        if len(sys.argv) > 3:
            start = int(sys.argv[2])
            M = int(sys.argv[3])

        problem.solveMPC(N, M, start=start, releaseErr=False)
        problem.printLapTime()
        problem.plotTrace()






    #     xOpt_ = []
    #     uOpt_ = []
    #     feas = True
    #     for i in range(start, start + M * jump, jump):
    #         stepStart = i
    #         try:
    #             feas, xOpt_, uOpt_, JOpt = problem.solve(N,
    #                                                      start=stepStart,
    #                                                      x0=xOpt[-1],
    #                                                      xinf=[None, None, 0])
    #         except ValueError as err:
    #             # use old prediction
    #             print(err)
    #
    #             feas = False
    #
    #         if not feas:
    #             # xOpt will change if infeasible
    #             xOpt_ = xOpt_[1:, :]
    #             uOpt_ = uOpt_[1:, :]
    #         xOpt.extend(xOpt_[1:jump + 1])
    #         uOpt.extend(uOpt_[0:jump])
    #         print(i, feas)
    #
    #     xOpt = np.asarray(xOpt)
    #     uOpt = np.asarray(uOpt)
    #
    # M *= jump



