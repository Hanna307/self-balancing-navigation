import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from sys import path
from pathlib import Path

path.append(str(Path(__file__).resolve().parents[1]))

from model.dynamics import nonlinear_robot_dynamics
from model.parameters import params


# Initial state
initial_state = [
    0.0,                  # x [m]
    0.0,                  # x_dot [m/s]
    np.deg2rad(5.0),      # theta [rad]
    0.0                   # theta_dot [rad/s]
]


# Free-fall simulation: no control torque
tau = 0.0


# Simulation time
t_span = (0.0, 5.0)


# Solve nonlinear dynamics
sol = solve_ivp(
    fun=lambda t, state: nonlinear_robot_dynamics(
        t,
        state,
        tau,
        params
    ),
    t_span=t_span,
    y0=initial_state,
    max_step=0.01
)


# Check result
print("Simulation successful:", sol.success)
print("Number of time points:", len(sol.t))
print("Final state:", sol.y[:, -1])
plt.figure()

plt.plot(sol.t, np.rad2deg(sol.y[2]))

plt.xlabel("Time [s]")
plt.ylabel("Theta [deg]")
plt.title("Free-fall simulation")

plt.grid()
plt.xlim(0, 1)
plt.show()