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
# 4. Calculate E^-1 F, E^-1 G, E^-1 H
# ============================================================

E_inv_F = np.linalg.solve(E, F)
E_inv_G = np.linalg.solve(E, G)
E_inv_H = np.linalg.solve(E, H)


# ============================================================
# 5. State-space matrices
#
# State:
# x = [position, velocity, angle, angular velocity]^T
#
# x_dot = A x + B v_s
# ============================================================

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
# 6. Output matrices
#
# y = [theta, x]^T
# ============================================================

C = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0]
])


D = np.array([
    [0.0],
    [0.0]
])


# ============================================================
# 7. Print results
# ============================================================

print("=== T09 - Electromechanical State-Space Model ===")
print()

print("E^-1 F:")
print(E_inv_F)
print()

print("E^-1 G:")
print(E_inv_G)
print()

print("E^-1 H:")
print(E_inv_H)
print()

print("Matrix A:")
print(A)
print()

print("Matrix B:")
print(B)
print()

print("Matrix C:")
print(C)
print()

print("Matrix D:")
print(D)
print()

print("Matrix dimensions:")
print("A:", A.shape)
print("B:", B.shape)
print("C:", C.shape)
print("D:", D.shape)