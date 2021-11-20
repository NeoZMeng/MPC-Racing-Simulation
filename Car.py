class Car:
    # m: mass
    # wb: wheelbase
    # J: rotational inertia
    #
    def __init__(self,
                 m: float = None,
                 wb: float = None,
                 J: float = None):
        self.m = m
        self.wb = wb
        self.J = J

