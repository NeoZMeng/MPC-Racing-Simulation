import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
from pyomo.core.base.PyomoModel import Model
import numpy as np

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
        self.tireTraction = 1
        self.initSpeed = 0
        self.startOffset = -track.fixedWidth * 0.9
        self.x0 = [self.startOffset, self.initSpeed, 0]
        self.xinf = [None, None, 0]

    def totalTime(self, model):
        result = 0
        for k in model.tmidx:
            b, v, th, a, psi, tht, lt = vars(model, k)
            # d = lt / pyo.cos(th) + (lt * pyo.tan(th) + b) * pyo.sin(tht) / pyo.cos(th - tht)
            d = lt + (lt * th + b) * tht
            # dt = (pyo.sqrt(2 * a * d + v * v) - v) / a
            dt = d / v
            result += dt
        return result

    def solve(self,
              N: int,
              start: int = 0,
              x0=None,
              xinf=None,
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

        model.x = pyo.Var(model.tidx, model.xidx, initialize=1)
        model.u = pyo.Var(model.tmidx, model.uidx, initialize=1)

        model.obj = pyo.Objective(rule=self.totalTime)

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
            # return a * a + omg * omg * v * v <= ((1 + 0.09 * v * v / model.car.m) * model.tireTraction * g) ** 2
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

        # initial condition:
        if x0 is not None:
            model.initConstr = pyo.Constraint(model.xidx, rule=lambda model, i: model.x[0, i] == x0[i] \
                if xinf[i] is not None else pyo.Constraint.Skip)

        # end condition:
        if xinf is not None:
            model.endConstr = pyo.Constraint(model.xidx, rule=lambda model, i: model.x[N, i] == xinf[i] \
                if xinf[i] is not None else pyo.Constraint.Skip)

        results = solver.solve(model, tee=tee)

        feas = results.solver.termination_condition == TerminationCondition.optimal
        xOpt = np.asarray([[model.x[t, i]() for i in model.xidx] for t in model.tidx])
        uOpt = np.asarray([[model.u[t, i]() for i in model.uidx] for t in model.tmidx])
        JOpt = pyo.value(model.obj)

        return feas, xOpt, uOpt, JOpt

    def getTimeLine(self, trackData, N, xOpt, uOpt) -> [float]:
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

    def getLapTime(self, trackData, N, xOpt, uOpt, overlapped=False, start=0, trackLen=None) -> float:
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

    def solveAtOnce(self, N: int = None, start: int = 0):
        if N is None:
            N = self.track.numOfSegments + 1

        feas, self.xOpt, self.uOpt, JOpt = self.solve(N, start, x0=self.x0, xinf=self.xinf)

