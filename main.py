import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
from pyomo.core.base.PyomoModel import Model
import numpy as np
from math import sqrt, sin, cos, atan, pi
import matplotlib.pyplot as plt

from Car import Car
from Track import Track

# x contains: position_x, position_y, heading, speed, turning_rate
# u contains: acceleration, steering_angle, delta_t


nx = 3
nu = 2
vMin = 20
vMax = 300
thetaMax = 0.5
# dtSet = [0.1, 1]

solver = pyo.SolverFactory('ipopt')

solver.options['halt_on_ampl_error'] = 'yes'
solver.options['print_level'] = 5


def vars(model, k):
    return (model.x[k, 0],
            model.x[k, 1],
            model.x[k, 2],
            model.u[k, 0],
            model.u[k, 1],
            model.trackDir[k],
            model.trackLen[k])


def totalTime(model):
    result = 0
    for k in model.tmidx:
        b, v, th, a, psi, tht, lt = vars(model, k)
        # d = lt / pyo.cos(th) + (lt * pyo.tan(th) + b) * pyo.sin(tht) / pyo.cos(th - tht)
        d = lt + (lt * th + b) * tht
        # dt = (pyo.sqrt(2 * a * d + v * v) - v) / a
        dt = d / v
        result += dt
    return result


def solve(N: int,
          car: Car,
          track: Track,
          maxTraction2: float = 1,
          start: int = 0,
          x0=None,
          xinf=None) -> (bool, np.ndarray, np.ndarray, float, Model):
    model = pyo.ConcreteModel()

    model.N = N
    model.car = car
    model.track = track
    model.start = start

    model.xidx = pyo.Set(initialize=range(nx))
    model.uidx = pyo.Set(initialize=range(nu))
    model.tidx = pyo.Set(initialize=range(N + 1))
    model.tmidx = pyo.Set(initialize=range(N))

    model.bMax = model.track.fixedWidth
    model.maxTraction2 = maxTraction2
    model.trackDir = pyo.Param(model.tmidx, initialize=model.track.segmentCall(2))
    model.trackLen = pyo.Param(model.tmidx, initialize=model.track.segmentCall(3))

    model.x = pyo.Var(model.tidx, model.xidx, initialize=1)
    model.u = pyo.Var(model.tmidx, model.uidx, initialize=1)

    model.obj = pyo.Objective(rule=totalTime)

    # state constraint
    def constrRule0(model, k):
        b, v, th, a, psi, tht, lt = vars(model, k)
        return model.x[k + 1, 0] == b + lt * th

    def constrRule1(model, k):
        b, v, th, a, psi, tht, lt = vars(model, k)
        dt = (lt + (lt * th + b) * tht) / v
        return model.x[k + 1, 1] == v + a * dt

    def constrRule2(model, k):
        b, v, th, a, psi, tht, lt = vars(model, k)
        dt = (lt + (lt * th + b) * tht) / v
        omg = v * psi / model.car.wb
        return model.x[k + 1, 2] == th + omg * dt - tht

    model.constr0 = pyo.Constraint(model.tmidx, rule=constrRule0)
    model.constr1 = pyo.Constraint(model.tmidx, rule=constrRule1)
    model.constr2 = pyo.Constraint(model.tmidx, rule=constrRule2)

    # state constraint
    model.stateConstr00 = pyo.Constraint(model.tmidx, rule=lambda model, k: -model.bMax <= model.x[k, 0])
    model.stateConstr01 = pyo.Constraint(model.tmidx, rule=lambda model, k: model.x[k, 0] <= model.bMax)
    model.stateConstr1 = pyo.Constraint(model.tmidx, rule=lambda model, k: vMin <= model.x[k, 1])
    model.stateConstr20 = pyo.Constraint(model.tmidx, rule=lambda model, k: -thetaMax <= model.x[k, 2])
    model.stateConstr21 = pyo.Constraint(model.tmidx, rule=lambda model, k: model.x[k, 2] <= thetaMax)

    # traction
    def tractionRule(model, k):
        b, v, th, a, psi, tht, lt = vars(model, k)
        omg = v * psi / model.car.wb
        return a * a + omg * omg * v * v <= model.maxTraction2

    model.tractionConstr = pyo.Constraint(model.tmidx, rule=tractionRule)

    # input constraints
    def inputConstrRule0(model, k):
        b, v, th, a, psi, tht, lt = vars(model, k)
        P = model.car.P
        m = model.car.m
        return a <= P / m / v

    def inputConstrRule1(model, k):
        b, v, th, a, psi, tht, lt = vars(model, k)
        psiMax = model.car.stL
        return (-psiMax, psi, psiMax)

    model.inputConstr0 = pyo.Constraint(model.tmidx, rule=inputConstrRule0)
    model.inputConstr1 = pyo.Constraint(model.tmidx, rule=inputConstrRule1)

    # initial condition:
    if x0 is not None:
        model.initConstr = pyo.Constraint(model.xidx, rule=lambda model, i: model.x[0, i] == x0[i])
    if xinf is not None:
        model.initConstr = pyo.Constraint(model.xidx, rule=lambda model, i: model.x[N, i] == xinf[i])

    # end condition:

    results = solver.solve(model, tee=False)

    feas = results.solver.termination_condition == TerminationCondition.optimal
    xOpt = np.asarray([[model.x[t, i]() for i in model.xidx] for t in model.tidx])
    uOpt = np.asarray([[model.u[t, i]() for i in model.uidx] for t in model.tmidx])
    JOpt = 0

    return feas, xOpt, uOpt, JOpt, model


def getTimeLine(trackData, N, xOpt, uOpt):
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


if __name__ == "__main__":
    car = Car(m=1500, wb=2.85, P=300000, stL=0.8)
    track = Track.trackFromDataFile('lagunaSeca.json')

    start = 0
    N = 0
    M = track.numOfSegments - N

    xOpt = [[0, 55, 0]]
    uOpt = []
    feas, xOpt, uOpt, JOpt, model = solve(M,
                                          car,
                                          track,
                                          maxTraction2=9.8 ** 2,
                                          start=start,
                                          x0=xOpt[-1])

    # for i in range(0,M,10):
    #     start = i
    #
    #     feas, xOpt_, uOpt_, JOpt, model = solve(N,
    #                                           car,
    #                                           track,
    #                                           maxTraction2=9.8**2,
    #                                           start=start,
    #                                           x0=xOpt[-1])
    #     xOpt.extend(xOpt_[1:11])
    #     uOpt.extend(uOpt_[0:10])
    #     print(feas)
    # print(xOpt)
    # print(uOpt)
    start = 0
    xOpt = np.asarray(xOpt)
    uOpt = np.asarray(uOpt)
    t = getTimeLine(track.segment(0, M), M, xOpt, uOpt)
    carPos = np.array([track.pos(start + k, xOpt[k, 0]) for k in range(M + 1)])
    aLat = uOpt[:, 1] * xOpt[:-1, 1] / car.wb * xOpt[:-1, 1]
    print(t[-1])

    fig = plt.figure()
    plot1 = fig.add_subplot(111, aspect=1)
    # plot2 = fig.add_subplot(222)
    # plot3 = fig.add_subplot(223)
    # plot4 = fig.add_subplot(224)
    # plot1.plot(*track.data.T[:2], '-r')
    trackAsGraph = track.trackAsGraph()
    plot1.plot(*trackAsGraph[0].T, '-r', linewidth=0.3)
    plot1.plot(*trackAsGraph[1].T, '-r', linewidth=0.3)
    plot1.plot(*carPos.T, '-b', linewidth=0.6)
    # plot2.plot(t, xOpt.T[1])
    # plot3.plot(t[:-1], aLat ** 2 + uOpt[:, 0] ** 2)
    # plot4.plot(t[:-1], uOpt.T[1])

    plt.show()
