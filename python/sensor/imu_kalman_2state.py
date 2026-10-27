import numpy as np
import matplotlib.pyplot as plt
# 1. Thong so mo phong
T = 10.0
dt = 0.01
t = np.arange(0, T + dt, dt)

N = len(t)

print("So mau:", N)
print("Thoi gian mo phong:", t[-1], "s")
print("Tan so lay mau:", 1 / dt, "Hz")
# 2. Ma tran mo hinh trang thai
A = np.array([
    [1.0, -dt],
    [0.0,  1.0]
])

B = np.array([
    [dt],
    [0.0]
])

H = np.array([
    [1.0, 0.0]
])

print("\nMa tran A:")
print(A)

print("\nMa tran B:")
print(B)

print("\nMa tran H:")
print(H)

print("\nKich thuoc:")
print("A:", A.shape)
print("B:", B.shape)
print("H:", H.shape)
# 3. Khoi tao Q, R, P va trang thai

# Ma tran nhieu qua trinh
Q = np.array([
    [1e-5, 0.0],
    [0.0, 1e-7]
])

# Ma tran nhieu phep do goc accelerometer
R = np.array([
    [0.1]
])

# Ma tran hiep phuong sai ban dau
P = np.array([
    [1.0, 0.0],
    [0.0, 1.0]
])

# Trang thai ban dau: [goc nghieng, gyro bias]
x = np.array([
    [0.0],
    [0.0]
])

print("\nMa tran Q:")
print(Q)

print("\nMa tran R:")
print(R)

print("\nMa tran P ban dau:")
print(P)

print("\nTrang thai ban dau x:")
print(x)

print("\nKich thuoc:")
print("Q:", Q.shape)
print("R:", R.shape)
print("P:", P.shape)
print("x:", x.shape)
# 4. Tao du lieu IMU mo phong

# Goc nghieng thuc te: bien thien hinh sin 10 do, tan so 0.2 Hz
theta_true = np.deg2rad(
    10 * np.sin(2 * np.pi * 0.2 * t)
)

# Mo phong accelerometer
g = 9.81
sigma_accel = 0.05
rng = np.random.default_rng(42)

noise_ax = rng.normal(0, sigma_accel, size=N)
noise_az = rng.normal(0, sigma_accel, size=N)

ax_raw = -g * np.sin(theta_true) + noise_ax
az_raw = g * np.cos(theta_true) + noise_az

# Tinh goc tu accelerometer
theta_accel = np.arctan2(-ax_raw, az_raw)

# Mo phong gyroscope
omega_true = np.gradient(theta_true, dt)

bias_gyro_true = 0.02
sigma_gyro = 0.005

noise_gyro = rng.normal(0, sigma_gyro, size=N)

omega_measured = (
    omega_true
    + bias_gyro_true
    + noise_gyro
)

print("\nDu lieu IMU mo phong:")
print("Goc that ban dau:", np.rad2deg(theta_true[0]), "deg")
print("Goc accelerometer ban dau:",
      np.rad2deg(theta_accel[0]), "deg")
print("Gyro do ban dau:", omega_measured[0], "rad/s")
print("Gyro bias that:", bias_gyro_true, "rad/s")
# 5. Luu ket qua Kalman Filter

x_history = np.zeros((2, N))
P_history = np.zeros((2, 2, N))
K_history = np.zeros((2, N))

# Khoi tao trang thai tu goc accelerometer dau tien
x[0, 0] = theta_accel[0]
x[1, 0] = 0.0

x_history[:, 0] = x[:, 0]
P_history[:, :, 0] = P
# 6. Kalman Filter: Prediction va Update

for k in range(1, N):

    # PREDICTION
    x_pred = A @ x + B * omega_measured[k]

    P_pred = A @ P @ A.T + Q
        # UPDATE

    # Sai lech giua goc accelerometer va goc du doan
    y = np.array([[theta_accel[k]]]) - H @ x_pred

    # Kalman Gain
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    # Cap nhat trang thai
    x = x_pred + K @ y

    # Cap nhat ma tran P
    I = np.eye(2)
    P = (I - K @ H) @ P_pred

    # Luu ket qua
    x_history[:, k] = x[:, 0]
    P_history[:, :, k] = P
    K_history[:, k] = K[:, 0]
    # 7. Danh gia ket qua

theta_est = x_history[0, :]
bias_est = x_history[1, :]

error_deg = np.rad2deg(theta_est - theta_true)
rmse = np.sqrt(np.mean(error_deg**2))

print("\n--- KALMAN FILTER 2 BIEN ---")
print("RMSE goc:", rmse, "deg")
print("Bias gyro that:", bias_gyro_true, "rad/s")
print("Bias gyro uoc luong cuoi:", bias_est[-1], "rad/s")
print("Kalman Gain goc cuoi:", K_history[0, -1])
print("Kalman Gain bias cuoi:", K_history[1, -1])
# 8. Ve do thi ket qua

plt.figure(figsize=(10, 5))
plt.plot(t, np.rad2deg(theta_true), label="True angle")
plt.plot(t, np.rad2deg(theta_est), label="Kalman 2-state")
plt.plot(t, np.rad2deg(theta_accel), alpha=0.5, label="Accelerometer angle")

plt.xlabel("Time (s)")
plt.ylabel("Angle (deg)")
plt.title("2-state Kalman Filter - Angle Estimation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(t, np.full(N, bias_gyro_true), label="True gyro bias")
plt.plot(t, bias_est, label="Estimated gyro bias")

plt.xlabel("Time (s)")
plt.ylabel("Gyro bias (rad/s)")
plt.title("2-state Kalman Filter - Gyro Bias Estimation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()