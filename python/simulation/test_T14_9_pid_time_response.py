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
# 2. PID parameters
# ============================================================

Kp = 2000.0
Kd = 0.1
Ki = 1.0

V_max = params['V_max']


# ============================================================
# 3. Initial condition
# ============================================================

theta_0 = np.deg2rad(2.0)

x0 = np.array([
    0.0,        # x
    0.0,        # x_dot
    theta_0,    # theta
    0.0,        # theta_dot
    0.0         # integral error
])


# ============================================================
# 4. PID closed-loop dynamics
# ============================================================

def closed_loop_dynamics(t, state):

    x, x_dot, theta, theta_dot, integral_error = state

    # Reference
    theta_target = 0.0

    # Error
    error = theta_target - theta

    # Derivative of error
    error_dot = -theta_dot

    # PID terms
    P = Kp * error
    I = Ki * integral_error
    D = Kd * error_dot

    # Controller sign according to T14.1
    v_unsaturated = -(P + I + D)

    # Voltage saturation
    v_s = np.clip(
        v_unsaturated,
        -V_max,
        V_max
    )

    # ========================================================
    # Anti-windup
    # ========================================================

    if abs(v_unsaturated) < V_max:

        integral_error_dot = error

    else:

        # Stop integral accumulation while saturated
        integral_error_dot = 0.0

    # Robot state derivative
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

    return np.array([
        robot_state_dot[0],
        robot_state_dot[1],
        robot_state_dot[2],
        robot_state_dot[3],
        integral_error_dot
    ])


# ============================================================
# 5. Simulation
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
# 6. Results
# ============================================================

theta_deg = np.rad2deg(
    solution.y[2, :]
)

print("T14.9 - PID Time Response")
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

print("\nFinal integral error:")
print(solution.y[4, -1])


# ============================================================
# 7. Plot
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    solution.t,
    theta_deg
)

plt.axhline(
    0.0,
    linestyle="--"
)

plt.xlabel("Time [s]")
plt.ylabel("Theta [deg]")

plt.title(
    "T14.9 - PID Balance Controller"
)

plt.grid(True)

plt.tight_layout()

plt.show()