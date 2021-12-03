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


nx = 5
nu = 2
# dtSet = [0.1, 1]

solver = pyo.SolverFactory('ipopt')

solver.options['halt_on_ampl_error'] = 'no'
solver.options['print_level'] = 5

#
# # calculate distance traveled
# def distance(model: Model):
#     # TODO: implementation
#     #  simple solution: calculate distance traveled
#     #  quality solution: calculate track length traveled
#
#     # simple solution
#     distanceTraveled = 0
#     for t in range(model.N):
#         distanceTraveled += model.x[t, 3] * dt
#     return distanceTraveled
#
#
# # def lapTime(model: Model):
# #     timeUsed = 0.
# #     for t in model.tmidx:
# #         timeUsed += model.u[t, 2]
# #     return timeUsed
#
#
# # constraint check: x(k+1) = g(x(k), u(k))
# def nextStateConstrs(model: Model, tSet):
#     # TODO: implementation
#     #   simple model
#     #   quality model
#
#     # simple model
#     model.nextStateConstrs0 = pyo.Constraint(tSet, rule=lambda model, t: model.x[t + 1, 0] == model.x[t, 0] + model.x[
#         t, 3] * pyo.cos(model.x[t, 2]) * dt)
#     model.nextStateConstrs1 = pyo.Constraint(tSet, rule=lambda model, t: model.x[t + 1, 1] == model.x[t, 1] + model.x[
#         t, 3] * pyo.sin(model.x[t, 2]) * dt)
#     model.nextStateConstrs2 = pyo.Constraint(
#         tSet, rule=lambda model, t: model.x[t + 1, 2] == model.x[t, 2] + model.x[t, 4] * dt)
#     model.nextStateConstrs3 = pyo.Constraint(
#         tSet, rule=lambda model, t: model.x[t + 1, 3] == model.x[t, 3] + model.u[t, 0] * dt)
#     model.nextStateConstrs4 = pyo.Constraint(tSet, rule=lambda model, t: model.x[t + 1, 4] == pyo.atan(
#         pyo.sin(model.u[t, 1]) * model.x[t, 3] / model.car.wb))
#
#
# # check if car is on track
# def onTrackConstrs(model: Model, tSet):
#     # TODO: implementation
#     #   simple solution: check if the center of vehicle is on the track
#     #   quality solution: check if the whole car is within the track
#
#     # simple solution
#     # model.onTrackConstr0 = pyo.Constraint(tSet, rule=lambda model, t: model.x[t, 0] ** 2 + model.x[t, 1] ** 2 >= 500**2)
#     model.onTrackConstr1 = pyo.Constraint(tSet, rule=lambda model, t: model.x[t, 0] ** 2 + model.x[t, 1] ** 2 <= 520**2)
#     # model.onTrackConstrs0 = pyo.Constraint(tSet, rule=lambda model, t: -520 <= model.x[t, 0])
#     # model.onTrackConstrs1 = pyo.Constraint(tSet, rule=lambda model, t: -520 <= model.x[t, 1])
#     # model.onTrackConstrs2 = pyo.Constraint(tSet, rule=lambda model, t: model.x[t, 0] <= 520)
#     # model.onTrackConstrs3 = pyo.Constraint(tSet, rule=lambda model, t: model.x[t, 1] <= 520)
#     # model.onTrackConstrs4 = pyo.Constraint(tSet, rule=lambda model, t: model.x[t, 0] <= -500)
#     # model.onTrackConstrs5 = pyo.Constraint(tSet, rule=lambda model, t: model.x[t, 1] <= -500)
#     # model.onTrackConstrs6 = pyo.Constraint(tSet, rule=lambda model, t: 500 <= model.x[t, 0])
#     # model.onTrackConstrs7 = pyo.Constraint(tSet, rule=lambda model, t: 500 <= model.x[t, 1])
#     pass
#
#
# # check if the car still has traction
# def tractionConstrs(model: Model, tSet):
#     # TODO: implementation
#     #   simple solution: 1G all around(assume car is AWD)
#     #   quality solutions: check wheel spin(assume car is RWD)
#     #   additional feature?: add understeer and oversteer
#
#     # simple solution
#     rule = lambda model, t: model.u[t, 0] ** 2 + (model.x[t, 4] * model.x[t, 3]) ** 2 <= 1
#     model.tractionConstr0 = pyo.Constraint(tSet, rule=rule)
#     pass
#
#
# # check if the car can repeat the same lap
# # def sameStarting(model: Model, t: int) -> bool:
# #     # if the car is on near the starting location
# #     # but different speed or etc., return false
# #     # TODO: implementation
# #     pass
#
# def stateBounds(model, tSet):
#     # model.stateConstr0 = pyo.Constraint(tSet, rule=lambda model, t: (-pi, model.x[t, 2], pi))
#     model.stateConstr1 = pyo.Constraint(tSet, rule=lambda model, t: (0.1, model.x[t, 3], 100))
#
#
# # check if input is within bounds
# def inputBounds(model: Model, tSet):
#     # TODO: implementation
#     #   simple solution: fixed power(assume CVT)
#     #   quality solution: realistic powertrain
#
#     # simple solution
#     model.car.accelConstrs(model, tSet, 0, 3)
#     model.car.steerConstrs(model, tSet, 1)
#     # model.inputBounds2 = pyo.Constraint(tSet, rule=lambda model, t: (dtBounds[0], model.u[t, 2], dtBounds[1]))
#
#
# def startingCondition(model: Model) -> bool:
#     model.startConstrs0 = pyo.Constraint(expr=model.x[0, 0] == 0)
#     model.startConstrs1 = pyo.Constraint(expr=model.x[0, 1] == 510)
#     model.startConstrs2 = pyo.Constraint(expr=model.x[0, 2] == 0)
#     model.startConstrs3 = pyo.Constraint(expr=model.x[0, 3] == 10)
#     model.startConstrs4 = pyo.Constraint(expr=model.x[0, 4] == 0)
#
#
# def endingCondition(model: Model) -> bool:
#     # model.endingConstrs0 = pyo.Constraint(expr=model.x[model.N, 0] == 100)
#     # # model.endingConstrs1 = pyo.Constraint(expr=(500, model.x[model.N, 1], 520))
#     # model.endingConstrs1 = pyo.Constraint(expr=model.x[model.N, 1] == 0)
#     # model.endingConstrs2 = pyo.Constraint(expr=model.x[model.N, 2] == -1.6)
#     model.endingConstrs3 = pyo.Constraint(expr=1 <= model.x[model.N, 3])

def


def solve(N: int,
          car: Car,
          track: Track,
          costFunc: callable) -> (bool, np.ndarray, np.ndarray, float, Model):
    model = pyo.ConcreteModel()

    model.N = N
    model.car = car
    model.track = track

    model.xidx = pyo.Set(initialize=range(nx))
    model.uidx = pyo.Set(initialize=range(nu))
    model.tidx = pyo.Set(initialize=range(N + 1))
    model.tmidx = pyo.Set(initialize=range(N))

    model.x = pyo.Var(model.tidx, model.xidx, initialize=0.3)
    model.u = pyo.Var(model.tmidx, model.uidx, initialize=0.2)

    model.obj = pyo.Objective(rule=costFunc, sense=pyo.maximize)

    nextStateConstrs(model, model.tmidx)
    # model.dtConstr = pyo.Constraint(model.tmidx, rule = lambda model, t: model.x[t, 4] * model.x[t, 3])**2
    onTrackConstrs(model, model.tidx)
    tractionConstrs(model, model.tmidx)
    stateBounds(model, model.tidx)
    startingCondition(model)
    endingCondition(model)
    inputBounds(model, model.tmidx)

    # model.display()
    # print
    results = solver.solve(model, tee=True)

    feas = results.solver.termination_condition == TerminationCondition.optimal
    xOpt = np.asarray([[model.x[t, i]() for i in model.xidx] for t in model.tidx])
    uOpt = np.asarray([[model.u[t, i]() for i in model.uidx] for t in model.tmidx])
    JOpt = 0

    return feas, xOpt, uOpt, JOpt, model


N = 50
car = Car(m=1200, wb=2, P=300000, stL=1)
track = Track.simpleTrack()
results = solve(N, car, track, distance)
print(results[0])
xOpt = results[1].T
uOpt = results[2].T

timeLine = np.array(range(N + 1)) * dt
distanceLine = [0]

for i in range(N):
    distanceLine.append(distanceLine[-1] + xOpt[3][i] * dt)

fig = plt.figure()
plot1 = fig.add_subplot(221, aspect=1)
plot2 = fig.add_subplot(222)
plot3 = fig.add_subplot(223)
plot4 = fig.add_subplot(224)
plot1.plot(xOpt[0], xOpt[1])
plot2.plot(distanceLine[:-1], uOpt[0])
plot3.plot(distanceLine[:-1], uOpt[1])
plot4.plot(timeLine, xOpt[2])

plt.show()
