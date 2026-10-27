import numpy as np

from sys import path
from pathlib import Path

path.append(str(Path(__file__).resolve().parents[1]))

from model.parameters import params


# ============================================================
# 1. Read mechanical parameters
# ============================================================

m_B = params['m_B']
m_W = params['m_W']

I_B = params['I_B']
I_W = params['I_W']

l = params['l']
r = params['r']

b_W = params['b_W']
g = params['g']


# ============================================================
# 2. Read actuator parameters
# ============================================================

R = params['R']
K = params['K']
n = params['n']

I_M = params['I_M']
I_G = params['I_G']

b_M = params['b_M']
b_G = params['b_G']


# ============================================================
# 3. Equivalent actuator parameters
# ============================================================

J_eq = I_M * n**2 + I_G

B_eq = (
    (b_M + K**2 / R) * n**2
    + b_G
)


# ============================================================
# 4. Matrix E
# ============================================================

E11 = (
    m_B
    + 2*m_W
    + 2*I_W/(r**2)
    + 2*J_eq/(r**2)
)

E12 = (
    m_B*l
    - 2*J_eq/r
)

E21 = E12

E22 = (
    m_B*l**2
    + I_B
    + 2*J_eq
)

E = np.array([
    [E11, E12],
    [E21, E22]
])


# ============================================================
# 5. Matrix F
# ============================================================

F11 = (
    2*b_W/(r**2)
    + 2*B_eq/(r**2)
)

F12 = -2*B_eq/r

F21 = F12

F22 = 2*B_eq

F = np.array([
    [F11, F12],
    [F21, F22]
])


# ============================================================
# 6. Matrix G
# ============================================================

G = np.array([
    [0.0, 0.0],
    [0.0, -m_B*g*l]
])


# ============================================================
# 7. Matrix H
# ============================================================

H = np.array([
    [2*K*n/(R*r)],
    [-2*K*n/R]
])


# ============================================================
# 8. Print results
# ============================================================

print("=== T08 - Electromechanical Matrices E, F, G, H ===")
print()

print("Equivalent actuator parameters:")
print("J_eq =", J_eq, "kg.m^2")
print("B_eq =", B_eq, "N.m.s/rad")
print()

print("Matrix E:")
print(E)
print()

print("Matrix F:")
print(F)
print()

print("Matrix G:")
print(G)
print()

print("Matrix H:")
print(H)
print()

print("Matrix dimensions:")
print("E:", E.shape)
print("F:", F.shape)
print("G:", G.shape)
print("H:", H.shape)