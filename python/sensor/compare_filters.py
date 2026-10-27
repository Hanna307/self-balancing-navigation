
import numpy as np
import matplotlib.pyplot as plt

# 1. Thiet lap thoi gian mo phong
T = 10.0
dt = 0.01
t = np.arange(0, T + dt, dt)

# 2. Tao goc nghieng thuc
goc_bien_do_deg = 10.0
tan_so = 0.2

theta_true = np.deg2rad(
    goc_bien_do_deg * np.sin(2 * np.pi * tan_so * t)
)

# 3. Tao du lieu accelerometer
g = 9.81
sigma_accel = 0.05
rng = np.random.default_rng(42)

noise_ax = rng.normal(0, sigma_accel, size=len(t))
noise_az = rng.normal(0, sigma_accel, size=len(t))

ax_raw = -g * np.sin(theta_true) + noise_ax
az_raw = g * np.cos(theta_true) + noise_az

theta_accel = np.arctan2(-ax_raw, az_raw)

# 4. Tao du lieu gyroscope
omega_true = np.gradient(theta_true, dt)

bias_gyro = 0.02
sigma_gyro = 0.005

noise_gyro = rng.normal(0, sigma_gyro, size=len(t))
omega_measured = omega_true + bias_gyro + noise_gyro

print("So mau:", len(t))
print("Thoi gian mo phong:", t[-1], "s")

# 5. Complementary Filter

alpha = 0.90
theta_comp = np.zeros(len(t))
theta_comp[0] = theta_accel[0]

for k in range(1, len(t)):
    theta_pred = (
        theta_comp[k - 1]
        + omega_measured[k] * dt
    )

    theta_comp[k] = (
        alpha * theta_pred
        + (1 - alpha) * theta_accel[k]
    )

error_comp_deg = np.rad2deg(theta_comp - theta_true)
rmse_comp = np.sqrt(np.mean(error_comp_deg**2))

print("\n--- COMPLEMENTARY FILTER ---")
print("Alpha:", alpha)
print("RMSE:", rmse_comp, "deg")

# 6. Kalman Filter

Q = 0.001
R = 0.01

theta_kalman = np.zeros(len(t))
theta_kalman[0] = theta_accel[0]

P = 1.0
K_history = np.zeros(len(t))

for k in range(1, len(t)):
    # Prediction
    theta_pred = (
        theta_kalman[k - 1]
        + omega_measured[k] * dt
    )

    P_pred = P + Q

    # Update
    K = P_pred / (P_pred + R)

    theta_kalman[k] = (
        theta_pred
        + K * (theta_accel[k] - theta_pred)
    )

    P = (1 - K) * P_pred
    K_history[k] = K

error_kalman_deg = np.rad2deg(
    theta_kalman - theta_true
)

rmse_kalman = np.sqrt(
    np.mean(error_kalman_deg**2)
)

print("\n--- KALMAN FILTER ---")
print("Q:", Q)
print("R:", R)
print("Kalman Gain cuoi:", K_history[-1])
print("RMSE:", rmse_kalman, "deg")

# 7. Ve do thi so sanh hai bo loc

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, np.rad2deg(theta_true), label="True angle")
plt.plot(t, np.rad2deg(theta_comp), label="Complementary Filter")
plt.plot(t, np.rad2deg(theta_kalman), label="Kalman Filter")

plt.ylabel("Angle (deg)")
plt.title("W6 - Complementary Filter vs Kalman Filter")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, error_comp_deg, label="Complementary error")
plt.plot(t, error_kalman_deg, label="Kalman error")

plt.xlabel("Time (s)")
plt.ylabel("Angle error (deg)")
plt.legend()
plt.grid(True)

plt.tight_layout()

output_path = "python/sensor/W6_filter_comparison.png"
plt.savefig(output_path, dpi=300)
plt.show()

print("Da luu do thi:", output_path)
