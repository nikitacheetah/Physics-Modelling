from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import time
from random import randint

from fontTools.svgLib.path import PathBuilder
from urllib3.filepost import writer


class GravitationalSystem(object):

    def forces(self, N, G, r, m):  # A function that defines the nature of forces
        F_each = np.zeros((N, N, 2))
        for i in range(N):
            for j in range(i + 1, N):
                r_ij = r[j] - r[i]
                r_magn = np.linalg.norm(r_ij)
                F_each[i][j] = G * m[i] * m[j] / (r_magn ** 3) * r_ij
                F_each[j][i] = -F_each[i][j]
        F = [sum(F_each[i]) for i in range(N)]
        return F

    def __init__(self, N: int, G: float, dt: float, m, r, v, acc: str, R: float):
        self.G = G
        self.acc = acc  # accuracy of the numerical ODE solving method on a scale from 1 to 10
        self.N = N  # number of bodies
        self.m = m  # mass of every particle (kg)
        self.dt = dt  # finite time interval (s)
        self.radius = R  # radius of a particle (m)
        self.r = r  # positions of every particle (m)
        self.v = v  # velocities of every particle (m/s)
        self.F = self.forces(self.N, self.G, self.r,
                             self.m)  # the total force acting from the side of all particles on a given (N)
        self.a = np.array([self.F[i] / self.m[i] if self.m[i] != 0 else [0, 0] for i in
                           range(self.N)])  # acceleration of every particle (m/s^2)
        self.KE = 0.5 * sum([self.m[i] * (np.linalg.norm(self.v[i]) ** 2) for i in
                             range(self.N)])  # total kinetic energy of a system (J)
        self.PE = 0  # total potential energy of a system (J)
        self.momentum = np.dot(self.m, self.v)  # total momentum (kg*m/s)
        self.r_cm = np.dot(self.m, self.r) / sum(self.m)  # centre of mass of a system
        self.epsilon = 0.001  # a softening factor for potential energy

        for i in range(self.N):
            for j in range(i):
                r_ij = self.r[i] - self.r[j]
                r_magn = np.linalg.norm(r_ij)
                self.PE += -self.G * self.m[i] * self.m[j] / np.sqrt((r_magn) ** 2 + (self.epsilon) ** 2)

        self.energy = self.KE + self.PE  # total mechanical energy

        self.params = [[self.KE, self.PE, self.energy, self.momentum, self.r_cm]]  # sanity-check parameters
        self.pos = [self.r]
        self.vel = [self.v]

        self.t = self.dt

    def calculate_system_state(self):  # ODE numerical solution
        if self.t >= 0:
            match int(self.acc):
                case 1:
                    v_next = self.v + self.a * self.dt
                    r_next = self.r + v_next * self.dt
                    self.r, self.v = r_next, v_next
                case 2:
                    v_mid = self.v + self.a * self.dt / 2
                    v_next = self.v + self.a * self.dt
                    r_next = self.r + v_mid * self.dt
                    self.r, self.v = r_next, v_next
                case 3:
                    k1 = self.v
                    k2 = self.v + self.a * self.dt / 2
                    k3 = self.v + self.a * self.dt
                    r_next = self.r + (k1 + 4 * k2 + k3) * self.dt / 6
                    self.r, self.v = r_next, k3
                case 4:
                    r_next = self.r + self.v * self.dt + self.a * (self.dt ** 2) / 2
                    F_next = self.forces(self.N, self.G, r_next, self.m)
                    a_next = np.array([self.F[i] / self.m[i] if self.m[i] != 0 else [0, 0] for i in range(self.N)])
                    v_next = self.v + (a_next + self.a) / 2 * self.dt
                    self.r, self.v = r_next, v_next

            step = int(self.t // self.dt)

            # Uptading parameters of a system according to said calculations

            self.pos.append(self.r)
            self.vel.append(self.v)
            # dr = np.array(self.pos[-1] - self.pos[-2])
            # D = np.array(np.linalg.norm(dr[i]) for i in range(self.N))

            self.F = self.forces(self.N, self.G, self.r, self.m)
            self.a = np.array([self.F[i] / self.m[i] if self.m[i] != 0 else [0, 0] for i in range(self.N)])
            self.KE = 0.5 * sum([self.m[i] * np.dot(self.v[i], self.v[i]) for i in range(self.N)])
            # self.PE += sum([np.dot(self.F[i], dr[i]) for i in range(N)])
            self.PE = 0

            for i in range(self.N):
                for j in range(i):
                    r_ij = self.r[i] - self.r[j]
                    r_magn = np.linalg.norm(r_ij)
                    self.PE += -self.G * self.m[i] * self.m[j] / np.sqrt((r_magn) ** 2 + (self.epsilon) ** 2)

            # if abs(self.energy - (self.KE + self.PE)) > 1e3:
            #     self.dt = 0.0001
            # else:
            #     self.dt = 0.01

            self.energy = self.KE + self.PE
            self.momentum = np.dot(self.m, self.v)
            self.r_cm = np.dot(self.m, self.r) / sum(self.m)

            self.params.append([self.KE, self.PE, self.energy, self.momentum, self.r_cm])
            self.t += self.dt

            r_eff = self.radius

            for i in range(self.N):
                for j in range(i + 1, self.N):
                    if np.linalg.norm(self.r[i] - self.r[j]) < 2 * r_eff:
                        r_ij = self.r[i] - self.r[j]  # distance between particle i and particle j
                        v_ij = self.v[i] - self.v[j]  # relative velocity between said particles
                        self.v[i] -= r_ij.dot(v_ij) / r_ij.dot(r_ij) * r_ij  # update velocity of particle i
                        self.v[j] += r_ij.dot(v_ij) / r_ij.dot(r_ij) * r_ij  # update velocity of particle j


def Process(inpt1: str, inpt2: str, inpt3: str):
    N = 0
    G = 1  # gravitational constant
    time = 10
    dt = 0.01
    steps = int(time / dt)
    m = 0
    r = 0
    v = 0
    R = 0

    match inpt1.upper():
        case "R":
            N = 49
            gridsize = 10
            sq = int(np.floor(np.sqrt(N)))
            # m = np.array([100 + (-1) ** randint(1, 2) * randint(0, 1) for i in range(N)]) # masses are close to equal with little fluctuations according to sqrt(N) rule
            m = np.array([100 for i in range(N)])
            r = np.array([np.array([gridsize * j, gridsize * i]) for j in range(sq) for i in
                          range(sq)])  # particles are located in rows
            if N - sq ** 2 != 0:
                r = np.append(r, np.array([np.array([gridsize * sq, gridsize * i]) for i in range(N - sq ** 2)]),
                              axis=0)
            v = np.array([np.array([0, 0]) for _ in range(N)])  # in the start, all of the particles dont move
            R = 1.0
        case "G":
            N = 101
            m = np.array([10 if i != 0 else 10000 for i in range(N)])
            n = 20
            r = np.array([np.array([0.0, 0.0]) for _ in range(N)])
            Ri = 25
            d = 100
            theta = 2 * np.pi / n
            v = np.array([np.array([0.0, 0.0]) for _ in range(N)])
            for i in range(N - 1):
                k = i % n
                if k == 0:
                    Ri += d
                v_1 = int(np.sqrt(G * m[0] / Ri)) + 10
                r[i + 1] += np.array([Ri * np.cos(k * theta), Ri * np.sin(k * theta)])
                v[i + 1] += np.array([-v_1 * np.sin(k * theta), -v_1 * np.cos(k * theta)])
            R = 10.0

    acc = input("""Choose your numerical ODE solving method:

    1 - Euler
    2 - Midpoint rule
    3 - Runge-Kutta 3rd order
    4 - Vertlet

""")

    star_gas = GravitationalSystem(N, G, dt, m, r, v, acc, R)  # initializing our system
    print("Executing...")
    for _ in range(steps):  # the process has begun
        star_gas.calculate_system_state()
    T = [i * dt for i in range(steps)]  # time discretisation
    positions = np.array(star_gas.pos)  # positions of every particle at each time step
    motion_integrals = star_gas.params

    fig, ax = plt.subplots()

    match inpt2.upper():
        case "A":
            def animate(frame):
                ax.clear()
                for i in range(N):
                    x, y = positions[frame - 1][i][0], positions[frame - 1][i][1]
                    x_next, y_next = positions[frame][i][0], positions[frame][i][1]
                    circle = plt.Circle((x, y), R)
                    ax.add_artist(circle)
                    plt.plot([x, x_next], [y, y_next])

                t = round(T[frame], 3)

                # Relative error of every motion integral
                ax.plot([], [], ' ', label=f't = {t} s')
                ax.plot([], [], ' ',
                        label=f'dK/K0 = {(motion_integrals[frame - 1][0] - motion_integrals[0][0]) / (motion_integrals[0][0] + 0.001) * 100}%')
                ax.plot([], [], ' ',
                        label=f'dП/П0 = {(motion_integrals[frame - 1][1] - motion_integrals[0][1]) / (motion_integrals[0][1] + 0.001) * 100}%')
                ax.plot([], [], ' ',
                        label=f'dE/E0 = {(motion_integrals[frame - 1][2] - motion_integrals[0][2]) / (motion_integrals[0][2] + 0.001) * 100}%')
                ax.plot([], [], ' ',
                        label=f'dP/P0 = {(motion_integrals[frame - 1][3] - motion_integrals[0][3]) / (motion_integrals[0][3] + 0.001) * 100}%')
                ax.plot([], [], ' ', label=f'Centre of mass = {motion_integrals[frame - 1][4]}')

                ax.legend()

                ax.set_xlim(-30 * N, 30 * N)
                ax.set_ylim(-30 * N, 30 * N)

                plt.grid()

                plt.tight_layout()

            animation = anim.FuncAnimation(fig, animate, frames=steps, interval=dt)
            file_path = "/home/nick/Projects/Physics-Modelling/docs/result/images/galactical.gif"
            pillow_writer = anim.PillowWriter(fps=20)
            animation.save(file_path, writer=pillow_writer)
        case "P":
            motion_slice = motion_integrals[:steps]
            match inpt3.upper():
                case "E":
                    E = calculate_delta(motion_slice, 2)
                    ax.plot(T, E)
                    ax.set_xlabel("Time, s")
                    ax.set_ylabel("dE/E0")
                case "P":
                    lengths = [np.linalg.norm(motion_integrals[i][3]) for i in range(steps)]
                    ax.plot(T, [abs((lengths[i] - lengths[0]) / lengths[0]) for i in range(steps)])
                    ax.set_xlabel("Time, s")
                    ax.set_ylabel("dP/P0")
                case "C":
                    ax.plot(T, [row[4] for row in motion_slice])
                    ax.set_xlabel("Time, s")
                    ax.set_ylabel("C, m")
                    ax.legend(["Cx", "Cy"])

    # plt.grid()
    # plt.show()


def calculate_delta(motion_integral, component_idx):
    initial_value = motion_integral[0][component_idx]
    return [abs((row[component_idx] - initial_value) / initial_value) for row in motion_integral]


def do_main():
#     usr_1 = input("""Приветствуем в симуляции проблемы N тел! Выберите тип начального состояния системы из предложенных:
#     R - Прямоугольная, (R)ectangular
#     G - Галактическая, (G)alactical
#
# """)
#     usr_2 = input("""Начальные условия заданы. Теперь выберите, каким образом вы хотите визуализировать полученные данные:
#     A - Анимация, (A)nimation
#     P - График сохраняющейся величины, (P)lot
#
# """)
#     if usr_2.upper() == "P":
#         usr_3 = input("""Выберите саму величину(относится к системе):
#         E - Энергия
#         P - Импульс
#         C - Центр масс
#
# """)
#     else:
#         usr_3 = ''
    usr_1='g'
    usr_2='a'
    usr_3=''
    Process(usr_1, usr_2, usr_3)


if __name__ == '__main__':
    do_main()

# if __name__ == '__main__':
#     N = 49
#     G = 1 # gravitational constant
#     time = 10
#     dt = 0.01
#     steps = int(time/dt)

#     sq = int(np.floor(np.sqrt(N)))
#     m = np.array([100 for i in range(N)])
#     r = np.array([np.array([10 * j, 10 * i]) for j in range(sq) for i in range(sq)]) # particles are located in rows
#     if N - sq ** 2 != 0:
#         r = np.append(r, np.array([np.array([10 * sq, 10 * i]) for i in range(N - sq ** 2)]), axis=0)
#     v = np.zeros((N, 2)) # in the start, all of the particles dont move
#     R = 1.0

#     acc = input("""Choose your numerical ODE solving method:

#     - Euler
#     - Midpoint
#     - Runge-Kutta
#     - Vertlet

# """)

#     star_gas = GravitationalSystem(N, G, dt, m, r, v, acc, R) # initializing our system
#     for _ in range(steps): # the process has begun
#         star_gas.calculate_system_state()
#     T = [i * dt for i in range(steps)] # time discretisation
#     positions = np.array(star_gas.pos) # positions of every particle at each time step
#     motion_integrals = star_gas.params

#     fig, ax = plt.subplots()

#     def animate(frame):
#         ax.clear()
#         for i in range(N):
#             x, y = positions[frame-1][i][0], positions[frame-1][i][1]
#             x_next, y_next = positions[frame][i][0], positions[frame][i][1]
#             circle = plt.Circle((x, y), R)
#             ax.add_artist(circle)
#             plt.plot([x, x_next], [y, y_next])

#         t = round(T[frame], 3)

#         # Relative error of every motion integral
#         ax.plot([], [], ' ', label=f't = {t} s')
#         ax.plot([], [], ' ', label=f'dK/K0 = {(motion_integrals[frame-1][0]-motion_integrals[0][0])/(motion_integrals[0][0]+0.001)*100}%')
#         ax.plot([], [], ' ', label=f'dП/П0 = {(motion_integrals[frame-1][1]-motion_integrals[0][1])/(motion_integrals[0][1]+0.001)*100}%')
#         ax.plot([], [], ' ', label=f'dE/E0 = {(motion_integrals[frame-1][2]-motion_integrals[0][2])/(motion_integrals[0][2]+0.001)*100}%')
#         ax.plot([], [], ' ', label=f'dP/P0 = {(motion_integrals[frame-1][3]-motion_integrals[0][3])/(motion_integrals[0][3]+0.001)*100}%')
#         ax.plot([], [], ' ', label=f'Centre of mass = {motion_integrals[frame-1][4]}')

#         ax.legend()

#         ax.set_xlim(-3000, 3000)
#         ax.set_ylim(-3000, 3000)

#         plt.grid()

#         ax.set_facecolor('black')

#         plt.tight_layout()

#     animation = anim.FuncAnimation(fig, animate, frames=steps, interval=dt)

#     # ax.plot(T, [(motion_integrals[i][0]-motion_integrals[0][0])/(motion_integrals[0][0]+0.001)*100 for i in range(steps)])
#     # ax.plot(T, [(motion_integrals[i][1]-motion_integrals[0][1])/(motion_integrals[0][1]+0.001)*100 for i in range(steps)])
#     # ax.plot(T, [(motion_integrals[i][2]-motion_integrals[0][2])/(motion_integrals[0][2]+0.01)*100 for i in range(steps)])

# ax.plot(T, [motion_integrals[i][0] for i in range(steps)])
# ax.plot(T, [motion_integrals[i][1] for i in range(steps)])
# ax.plot(T, [motion_integrals[i][2] for i in range(steps)])
# ax.plot(T, [motion_integrals[i][3] for i in range(steps)])
# ax.plot(T, [motion_integrals[i][4] for i in range(steps)])

# ax.set_xlabel("Time, s")
# ax.set_ylabel("dE/E0, %")

# plt.grid()

# plt.show()

    # path = r"TRANSCEND/Gravity/travel.gif"
    # writergif = anim.PillowWriter(fps=20)
    # animation.save(path, writer=writergif)
