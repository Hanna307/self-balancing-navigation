import numpy as np

from sys import path
from pathlib import Path

path.append(str(Path(__file__).resolve().parents[1]))

from model.dynamics import nonlinear_robot_dynamics
from model.parameters import params


# =========================
# Initial state
# =========================

state = [
    0.0,                  # x [m]
    0.0,                  # x_dot [m/s]
    0.0,                  # theta [rad]
    0.0                   # theta_dot [rad/s]
]


# =========================
# Positive torque
# =========================

tau_positive = 0.1       # [N.m]


result_positive = nonlinear_robot_dynamics(
    0.0,
    state,
    tau_positive,
    params
)


# =========================
# Negative torque
# =========================

tau_negative = -0.1      # [N.m]


result_negative = nonlinear_robot_dynamics(
    0.0,
    state,
    tau_negative,
    params
)


# =========================
# Print results
# =========================

print("Positive torque:")
print("tau =", tau_positive, "N.m")
print("state derivative =", result_positive)

print()

print("Negative torque:")
print("tau =", tau_negative, "N.m")
print("state derivative =", result_negative)