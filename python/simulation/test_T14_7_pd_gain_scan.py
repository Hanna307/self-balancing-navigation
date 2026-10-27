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
# 2. Controller / simulation parameters
# ============================================================

V_max = params['V_max']

theta_0 = np.deg2rad(1.0)

t_start = 0.0
t_end = 2.0

t_eval = np.linspace(t_start, t_end, 1001)


# ============================================================
# 3. Gain ranges
# ============================================================

Kp_values = [
    500,
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    8000,
    10000
]

Kd_values = [
    0.1,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0
]


# ============================================================
# 4. Simulation function
# ============================================================

def simulate_pd(Kp, Kd):

    x0 = np.array([
        0.0,
        0.0,
        theta_0,
        0.0
    ])

    def closed_loop_dynamics(t, state):

        x, x_dot, theta, theta_dot = state

        # PD controller
        v_s = Kp * theta + Kd * theta_dot

        # Voltage saturation
        v_s = np.clip(v_s, -V_max, V_max)

        state_dot = A @ state + B.flatten() * v_s

        return state_dot

    solution = solve_ivp(
        closed_loop_dynamics,
        (t_start, t_end),
        x0,
        t_eval=t_eval
    )

    theta_deg = np.rad2deg(solution.y[2, :])

    final_theta = abs(theta_deg[-1])
    max_theta = np.max(np.abs(theta_deg))

    return solution.success, final_theta, max_theta


# ============================================================
# 5. Gain scan
# ============================================================

results = []

print("T14.7 - PD Gain Scan")
print("=" * 70)

for Kp in Kp_values:

    for Kd in Kd_values:

        success, final_theta, max_theta = simulate_pd(Kp, Kd)

        results.append([
            Kp,
            Kd,
            final_theta,
            max_theta
        ])

        print(
            f"Kp={Kp:5.0f}, "
            f"Kd={Kd:4.1f}, "
            f"Final |theta|={final_theta:10.6f} deg, "
            f"Max |theta|={max_theta:10.6f} deg"
        )


# ============================================================
# 6. Find candidate gains
# ============================================================

print("\n" + "=" * 70)
print("Candidate stable gains")
print("=" * 70)

candidates = []

for Kp, Kd, final_theta, max_theta in results:

    if final_theta < 0.1 and max_theta < 5.0:

        candidates.append([
            Kp,
            Kd,
            final_theta,
            max_theta
        ])

if len(candidates) == 0:

    print("No candidate gain found.")

else:

    for item in candidates:

        print(
            f"Kp={item[0]:5.0f}, "
            f"Kd={item[1]:4.1f}, "
            f"Final |theta|={item[2]:.6f} deg, "
            f"Max |theta|={item[3]:.6f} deg"
        )


# ============================================================
# 7. Plot final theta for each gain
# ============================================================

plt.figure(figsize=(10, 6))

for Kp, Kd, final_theta, max_theta in results:

    success, _, _ = simulate_pd(Kp, Kd)

    if success:

        # Simulate again for plotting
        x0 = np.array([
            0.0,
            0.0,
            theta_0,
            0.0
        ])

        def dynamics(t, state):

            x, x_dot, theta, theta_dot = state

            v_s = Kp * theta + Kd * theta_dot

            v_s = np.clip(v_s, -V_max, V_max)

            return A @ state + B.flatten() * v_s

        solution = solve_ivp(
            dynamics,
            (t_start, t_end),
            x0,
            t_eval=t_eval
        )

        plt.plot(
            solution.t,
            np.rad2deg(solution.y[2, :]),
            label=f"Kp={Kp}, Kd={Kd}"
        )


plt.axhline(
    0.0,
    linestyle='--'
)

plt.xlabel("Time [s]")
plt.ylabel("Theta [deg]")
plt.title("T14.7 - PD Gain Scan")

plt.grid(True)

plt.legend(
    fontsize=7,
    ncol=2
)

plt.tight_layout()

plt.show()