from dataclasses import dataclass
from vector_operations import *
import math

G = 6.67e-11
dt = 0.01
N = 3


@dataclass
class Vector:
    x: float
    y: float

    def canonic(self):
        return [self.x, self.y]

    def magnitude(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)


@dataclass
class Solid:
    name: str
    m: float
    r: Vector
    v: Vector
    # g: Vector
    F: Vector
    a: Vector

    def canonic(self):
        return [self.name, self.m, self.r.canonic(), self.v.canonic(), self.F.canonic(), self.a.canonic()]


# class ODE_Solution():
    
#     def Euler_Integration(self)
        


if __name__ == "__main__":
    bodies = []
    for i in range(N):
        body = Solid(input(), float(input()), [float(i) for i in input().split()], [float(i) for i in input().split()], [0, 0], [0, 0], [0, 0])
        bodies.append(body)
        print(body)
    