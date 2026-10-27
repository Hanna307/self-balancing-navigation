import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from sys import path
from pathlib import Path

path.append(str(Path(__file__).resolve().parents[1]))

from model.dynamics import nonlinear_robot_dynamics
from model.parameters import params


# ============================================================
# 1. Initial condition
# ============================================================

initial_state = [
    0.0,                  # x [m]
    0.0,                  # x_dot [m/s]
    np.deg2rad(1.0),      # theta [rad]
    0.0                   # theta_dot [rad/s]
]

tau = 0.0
t_span = (0.0, 0.1)


# ============================================================
# 2. Nonlinear model
# ============================================================

sol_nonlinear = solve_ivp(
    fun=lambda t, state: nonlinear_robot_dynamics(
        t,
        state,
        tau,
        params
    ),
    t_span=t_span,
    y0=initial_state,
    max_step=0.001
)


# ============================================================
# 3. Linear State-Space model
# ============================================================

m_B = params['m_B']
m_W = params['m_W']
I_B = params['I_B']
I_W = params['I_W']

l = params['l']
r = params['r']
b_W = params['b_W']
g = params['g']


M_x = m_B + 2*m_W + 2*I_W/(r**2)
M_xtheta = m_B*l
M_theta = m_B*l**2 + I_B
D_x = 2*b_W/(r**2)

Delta = M_x*M_theta - M_xtheta**2


a1 = -M_theta*D_x/Delta
a2 = -M_xtheta*m_B*g*l/Delta
a3 = M_xtheta*D_x/Delta
a4 = M_x*m_B*g*l/Delta

b1 = (M_theta/r + M_xtheta)/Delta
b2 = -(M_xtheta/r + M_x)/Delta


A = np.array([
    [0, 1, 0, 0],
    [0, a1, a2, 0],
    [0, 0, 0, 1],
    [0, a3, a4, 0]
])

B = np.array([
    [0],
    [b1],
    [0],
    [b2]
])


def linear_dynamics(t, state):
    return A @ state + B[:, 0] * tau


sol_linear = solve_ivp(
    fun=linear_dynamics,
    t_span=t_span,
    y0=initial_state,
    max_step=0.001
)


# ============================================================
# 4. Print final results
# ============================================================

theta_nonlinear = np.rad2deg(sol_nonlinear.y[2])
theta_linear = np.rad2deg(sol_linear.y[2])

print("=== T06.5 - Linear vs Nonlinear ===")
print()
print("Initial theta =", np.rad2deg(initial_state[2]), "deg")
print("Torque =", tau, "N.m")
print("Simulation time =", t_span[1], "s")
print()

print("Final nonlinear theta =",
      theta_nonlinear[-1], "deg")

print("Final linear theta =",
      theta_linear[-1], "deg")

print("Final absolute error =",
      abs(theta_nonlinear[-1] - theta_linear[-1]),
      "deg")


# ============================================================
# 5. Plot comparison
# ============================================================

plt.figure()

plt.plot(
    sol_nonlinear.t,
    theta_nonlinear,
    label="Nonlinear"
)

plt.plot(
    sol_linear.t,
    theta_linear,
    "--",
    label="Linear"
)

plt.xlabel("Time [s]")
plt.ylabel("Theta [deg]")
plt.title("T06.5 - Linear vs Nonlinear Model")

plt.grid()
plt.legend()

plt.show()