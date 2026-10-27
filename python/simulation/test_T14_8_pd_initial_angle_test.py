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
# 2. PD controller
# ============================================================

Kp = 2000.0
Kd = 0.1

V_max = params['V_max']


# ============================================================
# 3. Initial angles to test
# ============================================================

initial_angles_deg = [
    0.5,
    1.0,
    2.0,
    5.0
]


# ============================================================
# 4. Simulation settings
# ============================================================

t_start = 0.0
t_end = 2.0

t_eval = np.linspace(
    t_start,
    t_end,
    2001
)


# ============================================================
# 5. Simulation function
# ============================================================

def simulate_pd(theta_initial_deg):

    theta_initial = np.deg2rad(theta_initial_deg)

    x0 = np.array([
        0.0,
        0.0,
        theta_initial,
        0.0
    ])

    def closed_loop_dynamics(t, state):

        x, x_dot, theta, theta_dot = state

        # PD controller
        v_s = Kp * theta + Kd * theta_dot

        # Voltage saturation
        v_s = np.clip(
            v_s,
            -V_max,
            V_max
        )

        state_dot = (
            A @ state
            + B.flatten() * v_s
        )

        return state_dot

    solution = solve_ivp(
        closed_loop_dynamics,
        (t_start, t_end),
        x0,
        t_eval=t_eval
    )

    theta_deg = np.rad2deg(
        solution.y[2, :]
    )

    final_theta = theta_deg[-1]

    max_abs_theta = np.max(
        np.abs(theta_deg)
    )

    return (
        solution,
        final_theta,
        max_abs_theta
    )


# ============================================================
# 6. Run all tests
# ============================================================

print("T14.8 - PD Initial Angle Robustness Test")
print("=" * 70)

print("\nController parameters:")
print("Kp =", Kp)
print("Kd =", Kd)
print("V_max =", V_max)

print("\nResults:")
print("-" * 70)

results = []

for angle in initial_angles_deg:

    solution, final_theta, max_abs_theta = (
        simulate_pd(angle)
    )

    results.append([
        angle,
        final_theta,
        max_abs_theta
    ])

    print(
        f"Initial theta = {angle:4.1f} deg | "
        f"Final theta = {final_theta:12.6f} deg | "
        f"Max |theta| = {max_abs_theta:12.6f} deg"
    )


# ============================================================
# 7. Plot
# ============================================================

plt.figure(figsize=(10, 6))

for angle in initial_angles_deg:

    solution, final_theta, max_abs_theta = (
        simulate_pd(angle)
    )

    theta_deg = np.rad2deg(
        solution.y[2, :]
    )

    plt.plot(
        solution.t,
        theta_deg,
        label=f"Initial = {angle} deg"
    )


plt.axhline(
    0.0,
    linestyle="--"
)

plt.xlabel("Time [s]")
plt.ylabel("Theta [deg]")

plt.title(
    "T14.8 - PD Initial Angle Robustness Test"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()