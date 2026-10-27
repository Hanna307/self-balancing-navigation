import numpy as np


# =========================
# W6-E01: RAW IMU MODEL
# =========================

g = 9.81  # m/s^2


def simulate_imu_rest():
    """
    Mô phỏng dữ liệu IMU khi robot đứng yên,
    ở tư thế thẳng đứng.
    """

    # Accelerometer
    ax = 0.0
    ay = 0.0
    az = g

    # Gyroscope
    gx = 0.0
    gy = 0.0
    gz = 0.0

    acceleration = np.array([ax, ay, az])
    angular_velocity = np.array([gx, gy, gz])

    return acceleration, angular_velocity


if __name__ == "__main__":

    acceleration, angular_velocity = simulate_imu_rest()

    print("W6-E01 - RAW IMU TEST")
    print("----------------------")

    print("Accelerometer:")
    print(f"ax = {acceleration[0]:.3f} m/s^2")
    print(f"ay = {acceleration[1]:.3f} m/s^2")
    print(f"az = {acceleration[2]:.3f} m/s^2")

    print()

    print("Gyroscope:")
    print(f"gx = {angular_velocity[0]:.3f} rad/s")
    print(f"gy = {angular_velocity[1]:.3f} rad/s")
    print(f"gz = {angular_velocity[2]:.3f} rad/s")