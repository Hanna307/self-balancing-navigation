def wheel_angular_to_linear_velocity(
    angular_velocity,
    wheel_radius
):
    """
    Convert wheel angular velocity to
    linear velocity.

    Parameters
    ----------
    angular_velocity : float
        Wheel angular velocity in rad/s.

    wheel_radius : float
        Wheel radius in meters.

    Returns
    -------
    float
        Linear velocity in m/s.
    """

    linear_velocity = wheel_radius * angular_velocity

    return linear_velocity


# ==========================================
# W5-E03
# Angular Velocity -> Linear Velocity
# ==========================================

wheel_radius = 0.0325

angular_velocities = [
    0,
    10,
    20,
    30
]


print("W5-E03: Angular Velocity -> Linear Velocity")
print("---------------------------------------------")


for angular_velocity in angular_velocities:

    linear_velocity = wheel_angular_to_linear_velocity(
        angular_velocity,
        wheel_radius
    )

    print(
        f"Angular velocity = "
        f"{angular_velocity:5.1f} rad/s | "
        f"Linear velocity = "
        f"{linear_velocity:.6f} m/s"
    )