
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# W6-E02: IMU NOISE, BIAS AND ANGLE DRIFT
# Muc dich: Mo phong nhieu, bias va do troi cua IMU
#
# LUU Y:
# Cac gia tri noise va bias trong bai test la gia dinh mo phong,
# khong phai so lieu do tu MPU6050 that.
# ============================================================


# 1. THIET LAP MO PHONG
g = 9.81                   # Gia toc trong truong (m/s^2)
duration = 10.0             # Thoi gian mo phong (s)
dt = 0.01                   # Chu ky lay mau (s), tuong ung 100 Hz

t = np.arange(0.0, duration + dt, dt)

theta_true = 0.0            # Goc nghieng thuc (rad)
theta_dot_true = 0.0        # Toc do goc thuc (rad/s)

# Cac tham so gia dinh de mo phong
sigma_accel = 0.05          # Do lech chuan noise accelerometer (m/s^2)
sigma_gyro = 0.005           # Do lech chuan noise gyroscope (rad/s)
b_gyro = 0.025               # Bias gyroscope (rad/s)

# Co dinh seed de cac lan chay cho ket qua giong nhau
rng = np.random.default_rng(42)


# ============================================================
# 2. MO PHONG DU LIEU ACCELEROMETER
# ============================================================

# Gia tri gia toc do trong luc khi robot dung thang
ax_ideal = -g * np.sin(theta_true)
az_ideal = g * np.cos(theta_true)

# Them noise ngau nhien
ax_raw = ax_ideal + rng.normal(
    loc=0.0,
    scale=sigma_accel,
    size=t.size
)

az_raw = az_ideal + rng.normal(
    loc=0.0,
    scale=sigma_accel,
    size=t.size
)

# Tinh goc tu accelerometer, don vi radian
theta_accel = np.arctan2(-ax_raw, az_raw)


# ============================================================
# 3. MO PHONG DU LIEU GYROSCOPE
# ============================================================

# Toc do goc do duoc = toc do goc thuc + bias + noise
gyro_raw = (
    theta_dot_true
    + b_gyro
    + rng.normal(
        loc=0.0,
        scale=sigma_gyro,
        size=t.size
    )
)


# ============================================================
# 4. TICH PHAN GYROSCOPE DE UOC LUONG GOC
# ============================================================

theta_gyro = np.zeros_like(t)

for k in range(1, len(t)):
    theta_gyro[k] = (
        theta_gyro[k - 1]
        + gyro_raw[k] * dt
    )


# ============================================================
# 5. THONG KE KET QUA
# ============================================================

print("W6-E02 - IMU NOISE AND BIAS TEST")
print("=" * 42)

print(f"Thoi gian mo phong : {duration:.2f} s")
print(f"Chu ky lay mau     : {dt:.3f} s")
print(f"Tan so lay mau     : {1 / dt:.1f} Hz")
print(f"So mau              : {len(t)}")

print("\nCAC THAM SO MO PHONG")
print("-" * 42)
print(f"Accelerometer noise : {sigma_accel:.5f} m/s^2")
print(f"Gyroscope noise     : {sigma_gyro:.5f} rad/s")
print(f"Gyroscope bias      : {b_gyro:.5f} rad/s")

print("\nACCELEROMETER")
print("-" * 42)
print(f"ax mean             : {np.mean(ax_raw):.5f} m/s^2")
print(f"ax std              : {np.std(ax_raw):.5f} m/s^2")
print(f"az mean             : {np.mean(az_raw):.5f} m/s^2")
print(f"az std              : {np.std(az_raw):.5f} m/s^2")
print(
    f"Goc accel mean      : "
    f"{np.rad2deg(np.mean(theta_accel)):.4f} deg"
)
print(
    f"Goc accel std       : "
    f"{np.rad2deg(np.std(theta_accel)):.4f} deg"
)

print("\nGYROSCOPE")
print("-" * 42)
print(f"Gyro mean           : {np.mean(gyro_raw):.5f} rad/s")
print(f"Gyro std            : {np.std(gyro_raw):.5f} rad/s")
print(
    f"Goc gyro cuoi       : "
    f"{np.rad2deg(theta_gyro[-1]):.4f} deg"
)

expected_drift_deg = np.rad2deg(b_gyro * duration)
print(
    f"Do troi du kien     : "
    f"{expected_drift_deg:.4f} deg sau {duration:.1f} s"
)


# ============================================================
# 6. VE DO THI
# ============================================================

fig, axes = plt.subplots(
    3, 1,
    figsize=(11, 9),
    sharex=True
)

# DO THI 1: GOC TU ACCELEROMETER
axes[0].plot(
    t,
    np.rad2deg(theta_accel),
    color="tab:blue",
    linewidth=0.9,
    label="Accelerometer angle"
)

axes[0].axhline(
    np.rad2deg(theta_true),
    color="black",
    linestyle="--",
    linewidth=1.3,
    label="True angle"
)

axes[0].set_ylabel("Angle (deg)")
axes[0].set_title("Accelerometer: Measurement Noise")
axes[0].grid(True, linestyle=":", alpha=0.6)
axes[0].legend(loc="upper right")


# DO THI 2: TOC DO GOC TU GYROSCOPE
axes[1].plot(
    t,
    gyro_raw,
    color="tab:blue",
    linewidth=0.9,
    label="Raw gyro"
)

axes[1].axhline(
    b_gyro,
    color="tab:red",
    linestyle="--",
    linewidth=1.5,
    label="Bias level"
)

axes[1].axhline(
    theta_dot_true,
    color="black",
    linestyle=":",
    linewidth=1.5,
    label="True angular rate"
)

axes[1].set_ylabel("Angular rate (rad/s)")
axes[1].set_title("Gyroscope: Noise and Bias")
axes[1].grid(True, linestyle=":", alpha=0.6)
axes[1].legend(loc="upper right")


# DO THI 3: GOC SAU KHI TICH PHAN GYROSCOPE
axes[2].plot(
    t,
    np.rad2deg(theta_gyro),
    color="tab:green",
    linewidth=1.5,
    label="Integrated gyro angle"
)

axes[2].axhline(
    np.rad2deg(theta_true),
    color="black",
    linestyle="--",
    linewidth=1.3,
    label="True angle"
)

axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Angle (deg)")
axes[2].set_title("Gyroscope Integration: Drift")
axes[2].grid(True, linestyle=":", alpha=0.6)
axes[2].legend(loc="upper left")


# TIEU DE CHUNG
fig.suptitle(
    "W6-E02: IMU Noise, Bias, and Angle Drift",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.97])


# ============================================================
# 7. LUU DO THI
# ============================================================

output_path = (
    Path(__file__).resolve().with_name(
        "W6_E02_imu_noise_bias.png"
    )
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print("\n" + "=" * 42)
print(f"Da luu do thi tai:\n{output_path}")

plt.show()
