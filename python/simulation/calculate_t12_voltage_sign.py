import numpy as np

from sys import path
from pathlib import Path

path.append(str(Path(__file__).resolve().parents[1]))

from model.parameters import params


# ============================================================
# 1. Read parameters
# ============================================================

m_B = params['m_B']
m_W = params['m_W']

I_B = params['I_B']
I_W = params['I_W']

l = params['l']
r = params['r']

b_W = params['b_W']
g = params['g']

R = params['R']
K = params['K']
n = params['n']

I_M = params['I_M']
I_G = params['I_G']

b_M = params['b_M']
b_G = params['b_G']


# ============================================================
# 2. Equivalent actuator parameters
# ============================================================

J_eq = I_M * n**2 + I_G

B_eq = (
    (b_M + K**2 / R) * n**2
    + b_G
)


# ============================================================
# 3. Build E, F, G, H
# ============================================================

E = np.array([
    [
        m_B + 2*m_W
        + 2*I_W/(r**2)
        + 2*J_eq/(r**2),

        m_B*l - 2*J_eq/r
    ],

    [
        m_B*l - 2*J_eq/r,

        m_B*l**2 + I_B + 2*J_eq
    ]
])


F = np.array([
    [
        2*b_W/(r**2)
        + 2*B_eq/(r**2),

        -2*B_eq/r
    ],

    [
        -2*B_eq/r,

        2*B_eq
    ]
])


G = np.array([
    [0.0, 0.0],
    [0.0, -m_B*g*l]
])


H = np.array([
    [2*K*n/(R*r)],
    [-2*K*n/R]
])


# ============================================================
# 4. Calculate state-space matrices
# ============================================================

E_inv_F = np.linalg.solve(E, F)
E_inv_G = np.linalg.solve(E, G)
E_inv_H = np.linalg.solve(E, H)


A = np.array([
    [0.0, 1.0, 0.0, 0.0],

    [
        -E_inv_G[0, 0],
        -E_inv_F[0, 0],
        -E_inv_G[0, 1],
        -E_inv_F[0, 1]
    ],

    [0.0, 0.0, 0.0, 1.0],

    [
        -E_inv_G[1, 0],
        -E_inv_F[1, 0],
        -E_inv_G[1, 1],
        -E_inv_F[1, 1]
    ]
])


B = np.array([
    [0.0],
    [E_inv_H[0, 0]],
    [0.0],
    [E_inv_H[1, 0]]
])


# ============================================================
# 5. Equilibrium state
# ============================================================

state = np.array([
    0.0,   # x
    0.0,   # x_dot
    0.0,   # theta
    0.0    # theta_dot
])


# ============================================================
# 6. Positive voltage
# ============================================================

v_positive = 1.0

state_dot_positive = (
    A @ state
    + B[:, 0] * v_positive
)


# ============================================================
# 7. Negative voltage
# ============================================================

v_negative = -1.0

state_dot_negative = (
    A @ state
    + B[:, 0] * v_negative
)


# ============================================================
# 8. Print results
# ============================================================

print("=== T12 - Motor Voltage Sign Check ===")
print()

print("Initial state:")
print(state)
print()

print("Positive voltage:")
print("v_s =", v_positive, "V")
print("state derivative =", state_dot_positive)
print()

print("Negative voltage:")
print("v_s =", v_negative, "V")
print("state derivative =", state_dot_negative)
print()

print("Expected sign:")
print("Positive voltage -> x_ddot > 0, theta_ddot < 0")
print("Negative voltage -> x_ddot < 0, theta_ddot > 0")