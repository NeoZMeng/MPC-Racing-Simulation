import pyomo.environ as pyo

maxAccel = 4  # (g)
minSpeed = 0.01  # (m/s)


class Car:
    # m: mass(kg)
    # wb: wheelbase(m)
    # J: rotational inertia
    # P: power(w)
    # stL: steering limit(rad)
    # bf: front wheel to center of mass
    # h: center of mass height
    def __init__(self,
                 m: float = None,
                 wb: float = None,
                 J: float = None,
                 P: float = None,
                 stL: float = None,
                 bf: float = None,
                 h: float = None
                 ):
        self.m = m
        self.wb = wb
        self.J = J
        self.P = P
        self.stL = stL
        self.bf = wb if bf is None else bf
        self.h = 0 if h is None else h

    @staticmethod
    def HP2W(hp: float) -> float:
        return 745.7 * hp

    def accelConstrs(self, model, tSet, accelIndex, speedIndex):
        model.accelConstr = pyo.Constraint(
            tSet, rule=lambda model, k: model.u[k, accelIndex] <= self.P / self.m / model.x[k, speedIndex])
        # model.accelConstr = pyo.Constraint(
        #     tSet, rule=lambda model, t: model.u[t, accelIndex] <= 0.5)

    def steerConstrs(self, model, tSet, steeringIndex: int):
        model.steeringConstr = pyo.Constraint(
            tSet, rule=lambda model, t: (-self.stL, model.u[t, steeringIndex], self.stL))
