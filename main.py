import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
from pyomo.core.base.PyomoModel import Model
import numpy as np

# x contains: position_x, position_y, heading, speed
# u contains: acceleration, steering_angle

nx = 4
nu = 2
solver = pyo.SolverFactory('ipopt')


def solve(N: int,
          dt: float,
          distanceFunc: callable,
          nextStateFunc: callable,
          onTrackFunc: callable,
          tractionFunc: callable,
          repeatableFunc: callable,
          inputBoundsFunc: callable) -> (bool, np.ndarray, np.ndarray, float, Model):
    model = pyo.ConcreteModel()

    model.N = N

    model.xidx = pyo.Set(initialize=range(nx))
    model.uidx = pyo.Set(initialize=range(nu))
    model.tidx = pyo.Set(initialize=range(N + 1))
    model.tmidx = pyo.Set(initialize=range(N))

    model.x = pyo.Var(model.tidx, model.xidx)
    model.u = pyo.Var(model.tmidx, model.uidx)

    model.obj = pyo.Objective(rule=distanceFunc, sense=pyo.maximize)

    model.constraint0 = pyo.Constraint(model.tmidx, model.xidx, rule=nextStateFunc)
    model.constraint1 = pyo.Constraint(model.tidx, rule=onTrackFunc)
    model.constraint2 = pyo.Constraint(model.tidx, rule=tractionFunc)
    if repeatableFunc is not None:  # additional function
        model.constraint3 = pyo.Constraint(model.tidx, rule=repeatableFunc)
    model.constraint4 = pyo.Constraint(model.tmidx, model.uidx, rule= inputBoundsFunc)
    results = solver.solve(model)

    feas = results.solver.termination_condition == TerminationCondition.optimal
    xOpt = np.asarray([[model.x[t, i]() for i in model.xidx] for t in model.tidx])
    uOpt = np.asarray([[model.u[t, i]() for i in model.uidx] for t in model.tmidx])
    JOpt = model.cost()

    return feas, xOpt, uOpt, JOpt, model


# calculate distance traveled
def distance(model: Model) -> float:
    # TODO: implementation
    pass


# constraint check: x(k+1) = g(x(k), u(k))
def nextState(model: Model, t: int, i: int) -> bool:
    # TODO: implementation
    pass


# check if car is on track
def onTrack(model: Model, t: int) -> bool:
    # TODO: implementation
    pass


# check if the car still has traction
def traction(model: Model, t: int) -> bool:
    # TODO: implementation
    pass


# check if the car can repeat the same lap
def sameStarting(model: Model, t: int) -> bool:
    # if the car is on near the starting location
    # but different speed or etc., return false
    # TODO: implementation
    pass


# check if input is within bounds
def inputBounds(model: Model, t: int, i: int) -> bool:
    # TODO: implementation
    pass

if __name__ == '__main__':
    N = 100
    dt = 0.1

    results = solve(N, dt, distance, nextState, onTrack, traction, None, inputBounds)
    print(results[0])
