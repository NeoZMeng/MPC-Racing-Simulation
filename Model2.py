import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
from pyomo.core.base.PyomoModel import Model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from Car import Car
from Track import Track

solver = pyo.SolverFactory('ipopt')
solver.options['halt_on_ampl_error'] = 'yes'
solver.options['print_level'] = 5

# x contains: position_x, position_y, heading, speed, turning_rate
# u contains: acceleration, steering_angle, delta_t

g = 9.8

nx = 3
nu = 2
vMin = 5
thetaMax = 0.5


# helper function to simplify constraint coding
def vars(model, k):
    return (model.x[k, 0],
            model.x[k, 1],
            model.x[k, 2],
            model.u[k, 0],
            model.u[k, 1],
            model.trackDir[k],
            model.trackLen[k])


def lapTimeFormat(time: float) -> str:
    minutes = int(time // 60)
    seconds = int(time % 60)
    fractions = time % 1
    fractStr = f'{fractions:.3f}'[2:]
    return f'{minutes:02d}:{seconds:02d}.{fractStr}'


class RaceProblem:
    def __init__(self, car: Car, track: Track):
        self.car = car
        self.track = track
        self.model = None
        self.xOpt = None
        self.uOpt = None
        self.JOpt = None
        self.tireTraction = 1
        self.initSpeed = vMin + 1
        self.startOffset = -track.fixedWidth * 0.8
        self.x0 = [self.startOffset, self.initSpeed, 0]
        self.xinf = [None, None, 0]

    def refreshModel(self):
        self.model = None
        self.xOpt = None
        self.uOpt = None
        self.JOpt = None
        self.x0 = [self.startOffset, self.initSpeed, 0]
        self.xinf = [None, None, 0]

    def totalTime(self, model):
        result = 0
        for k in model.tmidx:
            b, v, th, a, psi, tht, lt = vars(model, k)
            d = lt + (lt * th + b) * tht
            sq_ = v * v + 2 * a * d
            dt = (pyo.sqrt(sq_) - v) / a
            result += dt
        return result

    def solve(self,
              N: int,
              start: int = 0,
              x0=None,
              xinf=None,
              fastestLap=False,
              tee=False) -> (bool, np.ndarray, np.ndarray, float, Model):
        model = pyo.ConcreteModel()
        self.model = model

        model.N = N
        model.car = self.car
        model.track = self.track
        model.start = start

        model.xidx = pyo.Set(initialize=range(nx))
        model.uidx = pyo.Set(initialize=range(nu))
        model.tidx = pyo.Set(initialize=range(N + 1))
        model.tmidx = pyo.Set(initialize=range(N))

        model.bMax = model.track.fixedWidth
        model.tireTraction = self.tireTraction
        model.trackDir = pyo.Param(model.tmidx, initialize=model.track.segmentCall(2))
        model.trackLen = pyo.Param(model.tmidx, initialize=model.track.segmentCall(3))

        model.x = pyo.Var(model.tidx, model.xidx, initialize=lambda model, k, i: [0, 30, 0][i])
        model.u = pyo.Var(model.tmidx, model.uidx, initialize=0.01)

        model.obj = pyo.Objective(rule=self.totalTime)

        # special constraint
        def specialConstr0(model, k):
            b, v, th, a, psi, tht, lt = vars(model, k)
            d = lt + (lt * th + b) * tht
            sq_ = v * v + 2 * a * d
            return sq_ >= vMin

        model.specialConstr0 = pyo.Constraint(model.tmidx, rule=specialConstr0)

        # state constraint
        def constrRule0(model, k):
            b, v, th, a, psi, tht, lt = vars(model, k)
            return model.x[k + 1, 0] == b + lt * th

        def constrRule1(model, k):
            b, v, th, a, psi, tht, lt = vars(model, k)
            d = lt + (lt * th + b) * tht
            sq_ = v * v + 2 * a * d
            dt = (pyo.sqrt(sq_) - v) / a
            return model.x[k + 1, 1] == v + a * dt

        def constrRule2(model, k):
            b, v, th, a, psi, tht, lt = vars(model, k)
            d = lt + (lt * th + b) * tht
            sq_ = v * v + 2 * a * d
            dt = (pyo.sqrt(sq_) - v) / a
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
            return a * a + omg * omg * v * v <= (model.tireTraction * g) ** 2

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

        if fastestLap:
            # init == end condition
            model.initEndConstr = pyo.Constraint(model.xidx, rule=lambda model, i: model.x[0, i] == model.x[N, i])
        else:
            # initial condition:
            if x0 is not None:
                model.initConstr = pyo.Constraint(model.xidx, rule=lambda model, i: model.x[0, i] == x0[i] \
                    if x0[i] is not None else pyo.Constraint.Skip)

            # end condition:
            if xinf is not None:
                model.endConstr = pyo.Constraint(model.xidx, rule=lambda model, i: model.x[N, i] == xinf[i] \
                    if xinf[i] is not None else pyo.Constraint.Skip)

        # init = end

        results = solver.solve(model, tee=tee)

        feas = results.solver.termination_condition == TerminationCondition.optimal
        xOpt = np.asarray([[model.x[t, i]() for i in model.xidx] for t in model.tidx])
        uOpt = np.asarray([[model.u[t, i]() for i in model.uidx] for t in model.tmidx])
        JOpt = pyo.value(model.obj)

        return feas, xOpt, uOpt, JOpt

    def getTimeLine(self, start: int = 0) -> [float]:
        N = self.uOpt.shape[0]
        trackData = self.track.segment(start, N + start)
        result = [0]
        for k in range(N):
            b, v, th = self.xOpt[k]
            a, psi = self.uOpt[k]
            tht = trackData[k][2]
            lt = trackData[k][3]
            d = lt + (lt * th + b) * tht
            dt = d / v
            result.append(result[-1] + dt)
        return result

    def getLapTime(self, start=0) -> float:
        if self.JOpt is None:
            self.JOpt = self.getTimeLine(start=start)[-1]
        return self.JOpt

    def solveAtOnce(self, N: int = None, start: int = 0, fastestLap=False, tee=False):
        if N is None:
            N = self.track.numOfSegments + 1

        feas, self.xOpt, self.uOpt, self.JOpt = self.solve(N,
                                                           start,
                                                           x0=self.x0,
                                                           xinf=self.xinf,
                                                           fastestLap=fastestLap,
                                                           tee=tee)
        return feas

    def solveMPC(self, N: int, M: int, start: int = 0, releaseErr=False):
        xOpt = []
        uOpt = []
        xOpt_ = []
        uOpt_ = []
        feas = []
        nextx0 = self.x0
        nextxinf = self.xinf
        lastxOpt = None
        lastuOpt = None
        for i in range(start, start + M):
            stepStart = i
            try:
                feas_, xOpt_, uOpt_, JOpt_ = self.solve(N,
                                                        start=stepStart,
                                                        x0=nextx0,
                                                        xinf=nextxinf)
            except Exception as err:
                if lastxOpt is None:
                    releaseErr = True
                if releaseErr:
                    self.xOpt = np.asarray(xOpt)
                    self.uOpt = np.asarray(uOpt)
                    self.plotTrace()
                    feas_, xOpt_, uOpt_, JOpt_ = self.solve(N,
                                                            start=stepStart,
                                                            x0=nextx0,
                                                            xinf=nextxinf,
                                                            tee=True)
                else:
                    # treat as infeasible
                    print(err)
                    feas_ = False
            print(i, feas_)
            feas.append(feas_)
            lastxOpt = xOpt_ if feas_ else lastxOpt[1:]
            lastuOpt = uOpt_ if feas_ else lastuOpt[1:]
            nextx0 = lastxOpt[1, :]
            xOpt.append(lastxOpt[1])
            uOpt.append(lastuOpt[0])
        self.xOpt = np.asarray(xOpt)
        self.uOpt = np.asarray(uOpt)
        return feas

    def printLapTime(self):
        t = self.getLapTime()
        print("lap time: ", lapTimeFormat(t))

    def plotTrace(self, start: int = 0):
        N = self.uOpt.shape[0]
        carPos = np.array([self.track.pos(start + k, self.xOpt[k, 0]) for k in range(N)])

        fig = plt.figure()
        plot = fig.add_subplot(aspect=1, xlim=(0, 1000), ylim=(0, 1000))
        trackAsGraph = self.track.trackAsGraph()
        plot.plot(*trackAsGraph[0].T, '-r', linewidth=0.3)
        plot.plot(*trackAsGraph[1].T, '-r', linewidth=0.3)
        plot.plot(*carPos.T, '-b', linewidth=0.6)

        plt.show()

    def generateAnim(self, start: int = 0):
        t = self.getTimeLine()
        N = self.uOpt.shape[0]
        carPos = np.array([self.track.pos(start + k, self.xOpt[k, 0]) for k in range(N)])
        carHeading = np.array([self.xOpt[k][2] + self.track.direction(k) for k in range(N)])

        fig = plt.figure()
        plot = fig.add_subplot(aspect=1, xlim=(0, 1000), ylim=(0, 1000))
        trackAsGraph = self.track.trackAsGraph()
        plot.plot(*trackAsGraph[0].T, '-r', linewidth=0.3)
        plot.plot(*trackAsGraph[1].T, '-r', linewidth=0.3)

        carSize = 5
        car, = plot.plot([], [], '-b', linewidth=0.6)
        interval = 20  # milliseconds
        speedUp = 5
        dtFrame = interval * speedUp / 1000
        animIndex = []
        for k in range(N):
            animIndex.extend([k] * int((t[k + 1] / dtFrame) - len(animIndex)))

        def init():
            car.set_data([], [])
            return car,

        def animate(i):
            k = animIndex[i]
            x = carPos[k][0]
            y = carPos[k][1]
            th = carHeading[k]
            carAnimX = carPos.T[0][:k + 1]
            carAnimY = carPos.T[1][:k + 1]
            carAnimX = np.concatenate([carAnimX, [x + carSize * np.cos(th),
                                                  x + carSize * np.cos(th + np.pi * 3 / 4) / 2,
                                                  x + carSize * np.cos(th - np.pi * 3 / 4) / 2,
                                                  x + carSize * np.cos(th)]])
            carAnimY = np.concatenate([carAnimY, [y + carSize * np.sin(th),
                                                  y + carSize * np.sin(th + np.pi * 3 / 4) / 2,
                                                  y + carSize * np.sin(th - np.pi * 3 / 4) / 2,
                                                  y + carSize * np.sin(th)]])
            car.set_data(carAnimX, carAnimY)

            return car,

        anim = FuncAnimation(fig, animate, init_func=init, frames=int(t[-1] / dtFrame), interval=interval,
                             blit=True)
        anim.save('track.gif', writer='ffmpeg')
        plt.show()
