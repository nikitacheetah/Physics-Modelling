import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import time
from random import randint





class GravitationalSystem(object):



    def forces(self, N, r, m): # A function that defines the nature of forces
        F_each = np.zeros((N, N, 2))
        for i in range(N):
            for j in range(i+1, N):
                r_ij = r[i] - r[j]
                r_magn = np.linalg.norm(r_ij)
                F_each[i][j] = -G*m[i]*m[j]/(r_magn ** 3) * r_ij
                F_each[j][i] = -F_each[i][j]
        F = [sum(F_each[i]) for i in range(N)]
        return F

    def __init__(self, N: int, G: float, dt: float, m, r, v, acc: int):
        self.acc = acc # accuracy of the numerical ODE solving method on a scale from 1 to 10
        self.N = N # number of bodies
        self.m = m # mass of every particle (kg)
        self.dt = dt # finite time interval (s)
        self.r = r # positions of every particle (m)
        self.v = v # velocities of every particle (m/s)
        self.F = self.forces(self.N, self.r, self.m) # the total force acting from the side of all particles on a given (N)
        self.a = np.array([self.F[i] / self.m[i] if m[i] != 0 else [0, 0] for i in range(self.N)]) # acceleration of every particle (m/s^2)
        self.KE = 0.5 * sum([self.m[i] * (np.linalg.norm(self.v[i]) ** 2) for i in range(self.N)]) # total kinetic energy of a system (J)
        self.PE = 0 # total potential energy of a system (J)
        self.momentum = np.dot(self.m, self.v) # total momentum (kg*m/s)
        self.L = sum([np.cross(self.r[i], self.m[i] * self.v[i]) for i in range(self.N)]) # total angular momentum (kg*m^2/s)
        self.r_cm = np.dot(self.m, self.r) / sum(self.m) # centre of mass of a system
        self.epsilon = 0.001 # a softening factor for potential energy

        for i in range(self.N):
            for j in range(i):
                r_ij = self.r[i] - self.r[j]
                r_magn = np.linalg.norm(r_ij)
                self.PE += G*self.m[i]*self.m[j]/np.sqrt((r_magn)**2 + (self.epsilon)**2)
        
        self.energy = self.KE + self.PE # total mechanical energy

        self.params = [[self.KE, self.PE, self.momentum, self.L, self.r_cm]] # sanity-check parameters
        self.pos = [self.r]
        self.vel = [self.v]

        self.t = self.dt


    
    
    def calculate_system_state(self): # ODE numerical solution
        if self.t >= 0:
            match self.acc: # self.acc tells us about an accuracy of a method
                case 1: # Classical Euler integration
                    v_next = self.v + self.a * self.dt
                    r_next = self.r + v_next * self.dt
                    self.r, self.v = r_next, v_next
                case 2: # Midpoint rule
                    v_mid = self.v + self.a * self.dt / 2
                    v_next = self.v + self.a * self.dt
                    r_next = self.r + v_mid * self.dt
                    self.r, self.v = r_next, v_next
                case 3: # Runge-Kutta method (sorta)
                    k1 = self.v
                    k2 = self.v + self.a * self.dt / 2
                    k3 = self.v + self.a * self.dt
                    r_next = self.r + (k1 + 4 * k2 + k3) * self.dt / 6
                    self.r, self.v = r_next, k3

        
            step = int(self.t // self.dt)

            # Uptading parameters of a system according to said calculations

            self.pos.append(self.r)
            self.vel.append(self.v)

            self.F = self.forces(self.N, self.r, self.m)
            self.a = np.array([self.F[i] / self.m[i] if m[i] != 0 else [0, 0] for i in range(self.N)])
            self.KE = 0.5 * sum(self.m[i] * (np.linalg.norm(self.v[i]) ** 2) for i in range(self.N))
            self.PE = 0
            for i in range(self.N):
                for j in range(self.N):
                    if j != i:
                        r_ij = self.r[i] - self.r[j]
                        r_magn = np.linalg.norm(r_ij)
                        self.PE += G*self.m[i]*self.m[j]/(r_magn + self.epsilon)

            self.energy = self.KE + self.PE
            self.momentum = np.dot(self.m, self.v)
            self.L = sum([np.cross(self.r[i], self.m[i] * self.v[i]) for i in range(self.N)])
            self.r_cm = np.dot(self.m, self.r) / sum(self.m)
            
            self.params.append([self.KE, self.PE, self.momentum, self.L, self.r_cm])
            self.t += self.dt


    
if __name__ == '__main__':
    N = 10
    G = 1 # gravitational constant
    steps = 1000
    dt = 0.01

    m = np.array([100 + (-1) ** randint(1, 2) * randint(0, 10) for i in range(N)]) # masses are close to equal
    sq = int(np.floor(np.sqrt(N)))
    r = np.array([np.array([10 * j, 10 * i]) for j in range(sq) for i in range(sq)]) # particles are located in rows
    if N - sq ** 2 != 0:
        r = np.append(r, np.array([np.array([10 * sq, 10 * i]) for i in range(N - sq ** 2)]), axis=0)
    v = np.zeros((N, 2)) # in the start, all of the particles dont move

    acc = int(input("""Choose your numerical ODE solving method based on accuracy level:

    1 - Euler
    2 - Midpoint rule
    3 - Runge-Kutta 3rd order
    
"""))

    star_gas = GravitationalSystem(N, G, dt, m, r, v, acc) # initializing our system
    for _ in range(steps): # the process has begun
        star_gas.calculate_system_state()
    T = np.linspace(0, steps*dt, num=steps) # time discretisation
    positions = np.array(star_gas.pos) # positions of every particle at each time step
    motion_integrals = star_gas.params

    fig, ax = plt.subplots()

    t1 = time.time()

    def animate(frame):
        ax.clear()
        for i in range(N):
            x, y = positions[frame-1][i][0], positions[frame-1][i][1]
            x_next, y_next = positions[frame][i][0], positions[frame][i][1]
            circle = plt.Circle((x, y), 1)
            ax.add_artist(circle)
            plt.plot([x, x_next], [y, y_next])
            
        t2 = time.time()
        t = round(t2 - t1, 3)

        # Relative error of every motion integral
        ax.plot([], [], ' ', label=f't = {t}')
        ax.plot([], [], ' ', label=f'dK/K0 = {(motion_integrals[frame-1][0]-motion_integrals[0][0])/motion_integrals[1][0]*100}%')
        ax.plot([], [], ' ', label=f'dП/П0 = {(motion_integrals[frame-1][1]-motion_integrals[0][1])/motion_integrals[0][1]*100}%')
        ax.plot([], [], ' ', label=f'dP/P0 = {(motion_integrals[frame-1][2]-motion_integrals[0][2])/(motion_integrals[0][2]+0.001)*100}%')
        ax.plot([], [], ' ', label=f'dL/L0 = {(motion_integrals[frame-1][3]-motion_integrals[0][3])/(motion_integrals[0][3]+0.001)*100}%')
        ax.plot([], [], ' ', label=f'Centre of mass = {motion_integrals[frame-1][4]}')

        ax.legend()

        ax.set_xlim(-10*N, 10*N)
        ax.set_ylim(-10*N, 10*N)

        plt.grid()

        plt.tight_layout()

    animation = anim.FuncAnimation(fig, animate, frames=steps, interval=dt)

    plt.show()
    # path = r"/home/nick/Projects/Python/Learning/nick/modelling/black_hole.gif"
    # writergif = anim.PillowWriter(fps=20)
    # animation.save(path, writer=writergif)