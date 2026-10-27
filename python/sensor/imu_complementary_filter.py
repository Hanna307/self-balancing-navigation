
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

# 3. Kiem tra du lieu
print("So mau:", len(t))
print("Thoi gian dau:", t[0], "s")
print("Thoi gian cuoi:", t[-1], "s")
print("Goc lon nhat:", np.rad2deg(np.max(theta_true)), "deg")
print("Goc nho nhat:", np.rad2deg(np.min(theta_true)), "deg")

# 4. Mo phong accelerometer co nhieu
g = 9.81
sigma_accel = 0.05

rng = np.random.default_rng(42)

noise_ax = rng.normal(0, sigma_accel, size=len(t))
noise_az = rng.normal(0, sigma_accel, size=len(t))

ax_raw = -g * np.sin(theta_true) + noise_ax
az_raw = g * np.cos(theta_true) + noise_az

# 5. Uoc luong goc tu accelerometer
theta_accel = np.arctan2(-ax_raw, az_raw)

# 6. Kiem tra du lieu
print("\n--- E05: ACCELEROMETER ---")
print("Gia toc trong truong g:", g, "m/s^2")
print("Do lech chuan nhieu:", sigma_accel, "m/s^2")
print("ax trung binh:", np.mean(ax_raw), "m/s^2")
print("az trung binh:", np.mean(az_raw), "m/s^2")
print(
    "Goc accel tai t=0:",
    np.rad2deg(theta_accel[0]), "deg"
)

error_accel_deg = np.rad2deg(theta_accel - theta_true)
rmse_accel = np.sqrt(np.mean(error_accel_deg**2))

print("Sai so RMSE:", rmse_accel, "deg")

# 7. Mo phong gyroscope co bias va nhieu
omega_true = np.gradient(theta_true, dt)

bias_gyro = 0.02
sigma_gyro = 0.005

noise_gyro = rng.normal(
    0, sigma_gyro, size=len(t)
)

omega_measured = (
    omega_true + bias_gyro + noise_gyro
)

# 8. Tich phan gyroscope
theta_gyro = np.zeros(len(t))

for k in range(1, len(t)):
    theta_gyro[k] = (
        theta_gyro[k - 1]
        + omega_measured[k] * dt
    )

# 9. Danh gia sai so gyroscope
error_gyro_deg = np.rad2deg(
    theta_gyro - theta_true
)

rmse_gyro = np.sqrt(
    np.mean(error_gyro_deg**2)
)

print("\n--- E05: GYROSCOPE ---")
print("Bias gyro:", bias_gyro, "rad/s")
print("Do lech chuan nhieu:", sigma_gyro, "rad/s")
print(
    "Goc gyro cuoi:",
    np.rad2deg(theta_gyro[-1]), "deg"
)
print("RMSE gyro:", rmse_gyro, "deg")

# 10. Complementary Filter
alpha = 0.90
theta_complementary = np.zeros(len(t))

# Khoi tao goc tai mau dau tien
theta_complementary[0] = theta_accel[0]

for k in range(1, len(t)):
    # Du doan goc tu gyroscope
    theta_pred = (
        theta_complementary[k - 1]
        + omega_measured[k] * dt
    )

    # Ket hop gyroscope va accelerometer
    theta_complementary[k] = (
        alpha * theta_pred
        + (1 - alpha) * theta_accel[k]
    )

# 11. Danh gia sai so Complementary Filter
error_complementary_deg = np.rad2deg(
    theta_complementary - theta_true
)

rmse_complementary = np.sqrt(
    np.mean(error_complementary_deg**2)
)

print("\n--- E05: COMPLEMENTARY FILTER ---")
print("Alpha:", alpha)
print(
    "Goc loc cuoi:",
    np.rad2deg(theta_complementary[-1]), "deg"
)
print(
    "RMSE Complementary Filter:",
    rmse_complementary, "deg"
)

# 12. Ve do thi so sanh cac phuong phap
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, np.rad2deg(theta_true), label="True angle")
plt.plot(t, np.rad2deg(theta_accel),
         alpha=0.7, label="Accelerometer")
plt.plot(t, np.rad2deg(theta_gyro),
         label="Gyro integration")
plt.plot(t, np.rad2deg(theta_complementary),
         label="Complementary Filter")

plt.ylabel("Angle (deg)")
plt.title("W6-E05 - Complementary Filter")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, error_accel_deg, label="Accelerometer error")
plt.plot(t, error_gyro_deg, label="Gyro error")
plt.plot(t, error_complementary_deg,
         label="Complementary Filter error")

plt.xlabel("Time (s)")
plt.ylabel("Angle error (deg)")
plt.legend()
plt.grid(True)

plt.tight_layout()

output_path = "python/sensor/W6_E05_complementary_filter.png"
plt.savefig(output_path, dpi=300)
plt.show()

print("Da luu do thi:", output_path)
