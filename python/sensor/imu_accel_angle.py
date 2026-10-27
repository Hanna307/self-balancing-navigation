
import numpy as np
import matplotlib.pyplot as plt

# W6-E03: Accelerometer angle estimation

# 1. Simulation configuration
g = 9.81
dt = 0.01
T = 10.0
sigma_accel = 0.05

t = np.arange(0, T + dt, dt)
rng = np.random.default_rng(42)

# 2. True tilt angle: amplitude 10 degrees
theta_true = np.deg2rad(
    10 * np.sin(2 * np.pi * 0.2 * t)
)

# 3. Ideal accelerometer measurements
ax_ideal = -g * np.sin(theta_true)
az_ideal = g * np.cos(theta_true)

# 4. Add measurement noise
ax_raw = ax_ideal + rng.normal(
    0, sigma_accel, len(t)
)
az_raw = az_ideal + rng.normal(
    0, sigma_accel, len(t)
)

# 5. Estimate tilt from accelerometer
theta_accel = np.arctan2(-ax_raw, az_raw)

# 6. Calculate estimation error
error = theta_accel - theta_true
error_deg = np.rad2deg(error)

print("W6-E03 - ACCELEROMETER ANGLE TEST")
print(f"Simulation time  : {T:.2f} s")
print(f"Sampling period  : {dt:.3f} s")
print(f"Sampling rate    : {1/dt:.1f} Hz")
print(f"Number of samples: {len(t)}")
print(f"Noise sigma      : {sigma_accel:.3f} m/s^2")
print(f"Error mean       : {error_deg.mean():.4f} deg")
print(f"Error std        : {error_deg.std(ddof=1):.4f} deg")
print(f"Error RMS        : {np.sqrt(np.mean(error_deg**2)):.4f} deg")

# 7. Plot true and estimated angles
plt.figure(figsize=(10, 6))
plt.plot(t, np.rad2deg(theta_true), label="True angle")
plt.plot(
    t,
    np.rad2deg(theta_accel),
    label="Accelerometer estimate"
)
plt.xlabel("Time (s)")
plt.ylabel("Angle (deg)")
plt.title("W6-E03 - Accelerometer Angle Estimation")
plt.grid(True)
plt.legend()
plt.tight_layout()

# 8. Save figure
plt.savefig(
    "python/sensor/W6_E03_accel_angle.png",
    dpi=300
)
plt.show()
