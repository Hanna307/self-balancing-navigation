import numpy as np

from sys import path
from pathlib import Path

path.append(str(Path(__file__).resolve().parents[1]))

from model.dynamics import nonlinear_robot_dynamics
from model.parameters import params


# =========================
# Test state
# =========================

state = [
    0.0,                  # x [m]
    0.0,                  # x_dot [m/s]
    0.0,                  # theta [rad]
    0.0                   # theta_dot [rad/s]
]


# =========================
# Input torque
# =========================

tau = 0.1                # [N.m]


# =========================
# Get acceleration from model
# =========================

result = nonlinear_robot_dynamics(
    0.0,
    state,
    tau,
    params
)


x_dot = result[0]
x_ddot = result[1]

theta_dot = result[2]
theta_ddot = result[3]


# =========================
# Parameters
# =========================

m_B = params['m_B']
m_W = params['m_W']
I_B = params['I_B']
I_W = params['I_W']

l = params['l']
r = params['r']
b_W = params['b_W']
g = params['g']


theta = state[2]


# =========================
# Equation (2.12)
# =========================

lhs_x = (
    (m_B + 2*m_W + 2*I_W/(r**2)) * x_ddot
    + 2*b_W/(r**2) * x_dot
    + m_B*l*theta_ddot*np.cos(theta)
    - m_B*l*(theta_dot**2)*np.sin(theta)
)

rhs_x = tau / r


error_x = lhs_x - rhs_x


# =========================
# Equation (2.13)
# =========================

lhs_theta = (
    m_B*l*x_ddot*np.cos(theta)
    + (m_B*l**2 + I_B)*theta_ddot
    - m_B*g*l*np.sin(theta)
)

rhs_theta = -tau


error_theta = lhs_theta - rhs_theta


# =========================
# Print results
# =========================

print("=== T05 - Equation Check ===")

print()

print("Acceleration from dynamics.py:")
print("x_ddot     =", x_ddot, "m/s^2")
print("theta_ddot =", theta_ddot, "rad/s^2")

print()

print("Equation (2.12):")
print("LHS =", lhs_x)
print("RHS =", rhs_x)
print("Error =", error_x)

print()

print("Equation (2.13):")
print("LHS =", lhs_theta)
print("RHS =", rhs_theta)
print("Error =", error_theta)