import numpy as np


# ==========================================
# W5-E05
# Encoder Full Chain
# Count -> Angle -> Angular Velocity
#        -> Linear Velocity -> Distance
# ==========================================

cpr = 100
wheel_radius = 0.0325
delta_time = 0.1

counts = [0, 10, 20, 30, 40, 50]


print("W5-E05: Encoder Full Chain")
print("---------------------------")


previous_count = counts[0]
previous_time = 0.0

total_distance = 0.0


for i, current_count in enumerate(counts):

    current_time = i * delta_time

    # --------------------------------------
    # E01: Count -> Angle
    # --------------------------------------

    current_angle = (
        2 * np.pi * current_count / cpr
    )

    # --------------------------------------
    # First sample
    # --------------------------------------

    if i == 0:

        angular_velocity = 0.0
        linear_velocity = 0.0

    else:

        previous_angle = (
            2 * np.pi * previous_count / cpr
        )

        delta_angle = current_angle - previous_angle
        delta_t = current_time - previous_time

        # ----------------------------------
        # E02: Angle -> Angular Velocity
        # ----------------------------------

        angular_velocity = delta_angle / delta_t

        # ----------------------------------
        # E03: Angular Velocity
        #      -> Linear Velocity
        # ----------------------------------

        linear_velocity = (
            wheel_radius * angular_velocity
        )

        # ----------------------------------
        # E04: Linear Velocity -> Distance
        # ----------------------------------

        distance_increment = (
            linear_velocity * delta_t
        )

        total_distance += distance_increment

    print(
        f"Time = {current_time:.1f} s | "
        f"Count = {current_count:2d} | "
        f"Angle = {current_angle:.6f} rad | "
        f"Angular velocity = {angular_velocity:.6f} rad/s | "
        f"Linear velocity = {linear_velocity:.6f} m/s | "
        f"Distance = {total_distance:.6f} m"
    )

    previous_count = current_count
    previous_time = current_time


print()
print(
    f"Final distance = "
    f"{total_distance:.6f} m"
)