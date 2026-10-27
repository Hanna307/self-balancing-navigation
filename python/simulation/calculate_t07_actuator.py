from sys import path
from pathlib import Path

path.append(str(Path(__file__).resolve().parents[1]))

from model.parameters import params


# ============================================================
# W5.2 - Actuator equivalent parameters
# ============================================================

R = params['R']
K = params['K']
n = params['n']

I_M = params['I_M']
I_G = params['I_G']

b_M = params['b_M']
b_G = params['b_G']


# Equivalent inertia referred to wheel side
J_eq = I_M * n**2 + I_G


# Equivalent damping referred to wheel side
B_eq = (b_M + K**2 / R) * n**2 + b_G


# ============================================================
# Print results
# ============================================================

print("=== T07 - Actuator Equivalent Parameters ===")
print()

print("Motor parameters:")
print("R   =", R, "Ohm")
print("K   =", K)
print("n   =", n)
print("I_M =", I_M, "kg.m^2")
print("I_G =", I_G, "kg.m^2")
print("b_M =", b_M, "N.m.s/rad")
print("b_G =", b_G, "N.m.s/rad")
print()

print("Equivalent parameters at wheel side:")
print("J_eq =", J_eq, "kg.m^2")
print("B_eq =", B_eq, "N.m.s/rad")