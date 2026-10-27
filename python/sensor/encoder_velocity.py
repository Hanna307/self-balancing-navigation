import numpy as np


def encoder_count_to_angle(count, cpr):
    """
    Convert encoder count to wheel angle.

    Parameters
    ----------
    count : float
        Encoder count.

    cpr : float
        Counts per revolution.

    Returns
    -------
    float
        Wheel angle in radians.
    """

    return 2 * np.pi * count / cpr


def wheel_angular_velocity(
    previous_count,
    current_count,
    previous_time,
    current_time,
    cpr
):
    """
    Calculate wheel angular velocity from encoder counts.

    Returns
    -------
    float
        Wheel angular velocity in rad/s.
    """

    previous_angle = encoder_count_to_angle(
        previous_count,
        cpr
    )

    current_angle = encoder_count_to_angle(
        current_count,
        cpr
    )

    delta_angle = current_angle - previous_angle
    delta_time = current_time - previous_time

    angular_velocity = delta_angle / delta_time

    return angular_velocity


# ==========================================
# W5-E02
# Wheel Angle -> Angular Velocity
# ==========================================

cpr = 100

tests = [
    (0, 25, 0.0, 0.1),
    (0, 50, 0.0, 0.1),
    (0, 25, 0.0, 0.2)
]


print("W5-E02: Wheel Angle -> Angular Velocity")
print("----------------------------------------")


for previous_count, current_count, previous_time, current_time in tests:

    previous_angle = encoder_count_to_angle(
        previous_count,
        cpr
    )

    current_angle = encoder_count_to_angle(
        current_count,
        cpr
    )

    delta_angle = current_angle - previous_angle

    angular_velocity = wheel_angular_velocity(
        previous_count,
        current_count,
        previous_time,
        current_time,
        cpr
    )

    print(
        f"Count: {previous_count:3d} -> {current_count:3d} | "
        f"Delta angle = {delta_angle:.6f} rad | "
        f"Angular velocity = {angular_velocity:.6f} rad/s"
    )