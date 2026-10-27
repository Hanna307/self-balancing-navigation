
import numpy as np
import matplotlib.pyplot as plt

# 1. Thiet lap thoi gian mo phong
T = 10.0       # Tong thoi gian mo phong (s)
dt = 0.01      # Chu ky lay mau (s)
t = np.arange(0, T + dt, dt)

# 2. Thong so chuyen dong
goc_bien_do_deg = 10.0
tan_so = 0.2   # Hz
print("So mau:", len(t))
print("Thoi diem dau:", t[0], "s")
print("Thoi diem cuoi:", t[-1], "s")

# 3. Tao goc thuc (rad)
theta_true = np.deg2rad(
    goc_bien_do_deg * np.sin(2 * np.pi * tan_so * t)
)

# 4. Tinh van toc goc thuc (rad/s)
omega_true = np.gradient(theta_true, dt)

print("Goc thuc tai t=0:", np.rad2deg(theta_true[0]), "deg")
print("Van toc goc tai t=0:", omega_true[0], "rad/s")
print("Goc thuc lon nhat:", np.rad2deg(np.max(theta_true)), "deg")
print("Goc thuc nho nhat:", np.rad2deg(np.min(theta_true)), "deg")

# 5. Mo phong gyroscope
rng = np.random.default_rng(42)

bias_gyro = 0.02       # Bias (rad/s)
sigma_gyro = 0.005     # Do lech chuan nhieu (rad/s)

# Gyroscope ly tuong
omega_ideal = omega_true.copy()

# Gyroscope co bias va nhieu
noise_gyro = rng.normal(0, sigma_gyro, size=len(t))
omega_measured = omega_true + bias_gyro + noise_gyro

print("Bias gyroscope:", bias_gyro, "rad/s")
print("Nhieu gyro - do lech chuan:", sigma_gyro, "rad/s")
print("Van toc goc thuc tai t=0:", omega_true[0], "rad/s")
print("Van toc gyro do tai t=0:", omega_measured[0], "rad/s")

# 6. Tich phan van toc goc de uoc luong goc
theta_gyro_ideal = np.zeros(len(t))
theta_gyro_measured = np.zeros(len(t))

for k in range(1, len(t)):
    theta_gyro_ideal[k] = (
        theta_gyro_ideal[k - 1] + omega_ideal[k] * dt
    )

    theta_gyro_measured[k] = (
        theta_gyro_measured[k - 1] + omega_measured[k] * dt
    )

# 7. Tinh sai so goc (do)
error_ideal_deg = np.rad2deg(theta_gyro_ideal - theta_true)
error_measured_deg = np.rad2deg(theta_gyro_measured - theta_true)

print("Goc thuc cuoi:", np.rad2deg(theta_true[-1]), "deg")
print("Goc gyro ly tuong cuoi:", np.rad2deg(theta_gyro_ideal[-1]), "deg")
print("Goc gyro co sai so cuoi:", np.rad2deg(theta_gyro_measured[-1]), "deg")
print("Sai so gyro ly tuong cuoi:", error_ideal_deg[-1], "deg")
print("Sai so gyro co sai so cuoi:", error_measured_deg[-1], "deg")

# 8. Ve do thi so sanh goc
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, np.rad2deg(theta_true), label="True angle")
plt.plot(t, np.rad2deg(theta_gyro_ideal), "--",
         label="Ideal gyro integration")
plt.plot(t, np.rad2deg(theta_gyro_measured),
         label="Gyro with bias and noise")
plt.ylabel("Angle (deg)")
plt.title("W6-E04 - Gyroscope Integration")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, error_ideal_deg, label="Ideal integration error")
plt.plot(t, error_measured_deg,
         label="Integration error with bias and noise")
plt.xlabel("Time (s)")
plt.ylabel("Angle error (deg)")
plt.legend()
plt.grid(True)

plt.tight_layout()

# Luu hinh vao thu muc sensor
output_path = "python/sensor/W6_E04_gyro_integration.png"
plt.savefig(output_path, dpi=300)
plt.show()

print("Da luu do thi:", output_path)
