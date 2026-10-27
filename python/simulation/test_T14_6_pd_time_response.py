import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from sys import path
from pathlib import Path

path.append(str(Path(__file__).resolve().parents[1]))

from model.parameters import params


# ============================================================
# 1. System matrices from T09
# ============================================================

A = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, -634.20951051, -0.117913564, 20.59055556],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 5398.24190979, 57.32998454, -175.43190391]
])

B = np.array([
    [0.0],
    [3.11179212],
    [0.0],
    [-26.51252483]
])


# ============================================================
# 2. PD controller parameters
# ============================================================

Kp = 100.0
Kd = 5.0

V_max = params['V_max']


# ============================================================
# 3. Initial condition
# ============================================================

theta_0 = np.deg2rad(1.0)

x0 = np.array([
    0.0,        # x
    0.0,        # x_dot
    theta_0,    # theta
    0.0         # theta_dot
])


# ============================================================
# 4. Closed-loop dynamics
# ============================================================

def closed_loop_dynamics(t, state):

    x, x_dot, theta, theta_dot = state

    # Balance controller
    v_s = Kp * theta + Kd * theta_dot

    # Voltage saturation
    v_s = np.clip(v_s, -V_max, V_max)

    state_dot = A @ state + B.flatten() * v_s

    return state_dot


# ============================================================
# 5. Simulation
# ============================================================

t_start = 0.0
t_end = 2.0

t_eval = np.linspace(t_start, t_end, 2001)

solution = solve_ivp(
    closed_loop_dynamics,
    (t_start, t_end),
    x0,
    t_eval=t_eval
)


# ============================================================
# 6. Check simulation
# ============================================================

print("Simulation successful:", solution.success)

print("\nController parameters:")
print("Kp =", Kp)
print("Kd =", Kd)
print("V_max =", V_max)

print("\nInitial theta:")
print(np.rad2deg(theta_0), "deg")

print("\nFinal state:")
print(solution.y[:, -1])

print("\nFinal theta:")
print(np.rad2deg(solution.y[2, -1]), "deg")

print("\nMaximum absolute theta:")
print(
    np.rad2deg(
        np.max(np.abs(solution.y[2, :]))
    ),
    "deg"
)


# ============================================================
# 7. Plot
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    solution.t,
    np.rad2deg(solution.y[2, :])
)

plt.axhline(
    0.0,
    linestyle='--'
)

plt.xlabel("Time [s]")
plt.ylabel("Theta [deg]")
plt.title("T14.6 - PD Balance Controller")

plt.grid(True)

plt.tight_layout()

plt.show()