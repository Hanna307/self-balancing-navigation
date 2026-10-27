
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

# 3. Mo phong accelerometer co nhieu
g = 9.81
sigma_accel = 0.05

rng = np.random.default_rng(42)

noise_ax = rng.normal(0, sigma_accel, size=len(t))
noise_az = rng.normal(0, sigma_accel, size=len(t))

ax_raw = -g * np.sin(theta_true) + noise_ax
az_raw = g * np.cos(theta_true) + noise_az

# 4. Uoc luong goc tu accelerometer
theta_accel = np.arctan2(-ax_raw, az_raw)

# 5. Mo phong gyroscope co bias va nhieu
omega_true = np.gradient(theta_true, dt)

bias_gyro = 0.02
sigma_gyro = 0.005

noise_gyro = rng.normal(
    0, sigma_gyro, size=len(t)
)

omega_measured = (
    omega_true + bias_gyro + noise_gyro
)

# 6. Kiem tra du lieu dau vao
print("So mau:", len(t))
print("Thoi gian mo phong:", t[-1], "s")
print("Goc thuc tai mau dau:", np.rad2deg(theta_true[0]), "deg")
print("Goc accel tai mau dau:", np.rad2deg(theta_accel[0]), "deg")
print("Gyro tai mau dau:", omega_measured[0], "rad/s")
# 7. Khoi tao Kalman Filter

theta_est = np.zeros(len(t))

# Khoi tao goc uoc luong bang accelerometer
theta_est[0] = theta_accel[0]

# Do bat dinh ban dau cua goc uoc luong
P = 1.0

# Nhieu mo hinh va nhieu phep do
Q = 0.001
R = 0.01

# Mang luu Kalman Gain
K_history = np.zeros(len(t))

# 8. Vong lap Kalman Filter

for k in range(1, len(t)):

    # Buoc 1: Prediction - du doan goc tu gyroscope
    theta_pred = (
        theta_est[k - 1]
        + omega_measured[k] * dt
    )

    # Du doan do bat dinh
    P_pred = P + Q

    # Buoc 2: Update - tinh Kalman Gain
    K = P_pred / (P_pred + R)

    # Hieu chinh goc bang accelerometer
    theta_est[k] = (
        theta_pred
        + K * (theta_accel[k] - theta_pred)
    )

    # Cap nhat do bat dinh
    P = (1 - K) * P_pred

    # Luu Kalman Gain
    K_history[k] = K
# 9. Danh gia Kalman Filter

error_kalman_deg = np.rad2deg(
    theta_est - theta_true
)

rmse_kalman = np.sqrt(
    np.mean(error_kalman_deg**2)
)

print("\n--- E06: KALMAN FILTER ---")
print("Q:", Q)
print("R:", R)
print("Goc Kalman cuoi:", np.rad2deg(theta_est[-1]), "deg")
print("Kalman Gain cuoi:", K_history[-1])
print("RMSE Kalman:", rmse_kalman, "deg")
# 10. Ve do thi Kalman Filter

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, np.rad2deg(theta_true), label="True angle")
plt.plot(t, np.rad2deg(theta_accel),
         alpha=0.6, label="Accelerometer")
plt.plot(t, np.rad2deg(theta_est),
         label="Kalman Filter")

plt.ylabel("Angle (deg)")
plt.title("W6-E06 - Kalman Filter")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, error_kalman_deg, label="Kalman error")

plt.xlabel("Time (s)")
plt.ylabel("Angle error (deg)")
plt.legend()
plt.grid(True)

plt.tight_layout()

output_path = "python/sensor/W6_E06_kalman_filter.png"
plt.savefig(output_path, dpi=300)
plt.show()

print("Da luu do thi:", output_path)