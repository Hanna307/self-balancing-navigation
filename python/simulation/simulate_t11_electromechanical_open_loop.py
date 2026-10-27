import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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
# 5. Initial condition
# ============================================================

initial_state = [
    0.0,                  # x [m]
    0.0,                  # x_dot [m/s]
    np.deg2rad(1.0),      # theta [rad]
    0.0                   # theta_dot [rad/s]
]


# ============================================================
# 6. Input voltage
# ============================================================

v_s = 0.0

t_span = (0.0, 0.1)


# ============================================================
# 7. State-space dynamics
# ============================================================

def state_space_dynamics(t, state):

    state = np.asarray(state)

    state_dot = A @ state + B[:, 0] * v_s

    return state_dot


# ============================================================
# 8. Run simulation
# ============================================================

sol = solve_ivp(
    fun=state_space_dynamics,
    t_span=t_span,
    y0=initial_state,
    max_step=0.001
)


# ============================================================
# 9. Print results
# ============================================================

print("=== T11 - Electromechanical State-Space Open-Loop ===")
print()

print("Initial state:")
print(initial_state)
print()

print("Motor voltage:")
print("v_s =", v_s, "V")
print()

print("Simulation successful:", sol.success)
print("Number of time points:", len(sol.t))
print()

print("Final state:")
print(sol.y[:, -1])
print()

print("Final position:")
print(sol.y[0, -1], "m")

print("Final velocity:")
print(sol.y[1, -1], "m/s")

print("Final angle:")
print(np.rad2deg(sol.y[2, -1]), "deg")

print("Final angular velocity:")
print(sol.y[3, -1], "rad/s")


# ============================================================
# 10. Plot tilt angle
# ============================================================

plt.figure()

plt.plot(
    sol.t,
    np.rad2deg(sol.y[2])
)

plt.xlabel("Time [s]")
plt.ylabel("Theta [deg]")
plt.title("T11 - Electromechanical Open-Loop Response")

plt.grid()
plt.show()