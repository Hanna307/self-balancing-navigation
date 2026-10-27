import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from sys import path
from pathlib import Path

path.append(str(Path(__file__).resolve().parents[1]))

from model.parameters import params


# ============================================================
# T09 - Electromechanical State-Space Model
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
# PID parameters - same as T14.9
# ============================================================

Kp = 2000.0
Kd = 0.1
Ki = 1.0

V_max = params['V_max']

theta_0 = np.deg2rad(2.0)


# ============================================================
# Initial state
# ============================================================

# [x, x_dot, theta, theta_dot, integral_error]

x0 = np.array([
    0.0,
    0.0,
    theta_0,
    0.0,
    0.0
])


# ============================================================
# Storage for PID control signals
# ============================================================

time_history = []
v_unsaturated_history = []
v_s_history = []


# ============================================================
# Closed-loop dynamics
# ============================================================

def closed_loop_dynamics(t, state):

    x, x_dot, theta, theta_dot, integral_error = state

    theta_target = 0.0

    # Angle error
    error = theta_target - theta

    # Error derivative
    error_dot = -theta_dot

    # PID terms
    P = Kp * error
    I = Ki * integral_error
    D = Kd * error_dot

    # PID output before saturation
    v_unsaturated = -(P + I + D)

    # Voltage saturation
    v_s = np.clip(
        v_unsaturated,
        -V_max,
        V_max
    )

    # Simple anti-windup
    if abs(v_unsaturated) < V_max:
        integral_error_dot = error
    else:
        integral_error_dot = 0.0

    # Robot dynamics
    robot_state = np.array([
        x,
        x_dot,
        theta,
        theta_dot
    ])

    robot_state_dot = (
        A @ robot_state
        + B.flatten() * v_s
    )

    # Save controller signals
    time_history.append(t)
    v_unsaturated_history.append(v_unsaturated)
    v_s_history.append(v_s)

    return np.array([
        robot_state_dot[0],
        robot_state_dot[1],
        robot_state_dot[2],
        robot_state_dot[3],
        integral_error_dot
    ])


# ============================================================
# Simulation
# ============================================================

t_start = 0.0
t_end = 2.0

t_eval = np.linspace(
    t_start,
    t_end,
    2001
)

solution = solve_ivp(
    closed_loop_dynamics,
    (t_start, t_end),
    x0,
    t_eval=t_eval
)


# ============================================================
# Convert angle to degree
# ============================================================

theta_deg = np.rad2deg(
    solution.y[2, :]
)


# ============================================================
# Convert controller history to arrays
# ============================================================

time_history = np.array(time_history)
v_unsaturated_history = np.array(
    v_unsaturated_history
)
v_s_history = np.array(
    v_s_history
)


# ============================================================
# Saturation statistics - evaluated on uniform t_eval points
# ============================================================

# Recalculate PID control effort at the actual simulation points
theta_history = solution.y[2, :]
theta_dot_history = solution.y[3, :]
integral_history = solution.y[4, :]

error_history = -theta_history
error_dot_history = -theta_dot_history

v_unsaturated_eval = -(
    Kp * error_history
    + Kd * error_dot_history
    + Ki * integral_history
)

v_s_eval = np.clip(
    v_unsaturated_eval,
    -V_max,
    V_max
)

saturated = (
    np.abs(v_unsaturated_eval) >= V_max
)

saturation_percentage = (
    np.sum(saturated)
    / len(saturated)
    * 100
)

saturation_time = (
    np.sum(saturated)
    * (t_eval[1] - t_eval[0])
)

max_abs_v_unsaturated = np.max(
    np.abs(v_unsaturated_eval)
)

max_abs_v_unsaturated = np.max(
    np.abs(v_unsaturated_history)
)


# ============================================================
# Print results
# ============================================================

print("T14.10 - PID Control Effort & Saturation")
print("=" * 70)

print("\nController parameters:")
print("Kp =", Kp)
print("Kd =", Kd)
print("Ki =", Ki)
print("V_max =", V_max)

print("\nInitial theta:")
print(np.rad2deg(theta_0), "deg")

print("\nFinal theta:")
print(theta_deg[-1], "deg")

print("\nMaximum absolute theta:")
print(np.max(np.abs(theta_deg)), "deg")

print("\nMaximum absolute unsaturated voltage:")
print(max_abs_v_unsaturated, "V")

print("\nSaturation percentage:")
print(saturation_percentage, "%")


# ============================================================
# Plot control effort
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    time_history,
    v_unsaturated_history,
    label="V_unsaturated"
)

plt.plot(
    time_history,
    v_s_history,
    label="V_motor"
)

plt.axhline(
    V_max,
    linestyle="--"
)

plt.axhline(
    -V_max,
    linestyle="--"
)

plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")

plt.title(
    "T14.10 - PID Control Effort and Saturation"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()