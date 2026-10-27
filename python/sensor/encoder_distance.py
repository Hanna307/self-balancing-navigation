def calculate_distance(
    linear_velocity,
    delta_time
):
    """
    Calculate distance traveled
    during a time interval.

    Parameters
    ----------
    linear_velocity : float
        Linear velocity in m/s.

    delta_time : float
        Time interval in seconds.

    Returns
    -------
    float
        Distance traveled in meters.
    """

    distance = linear_velocity * delta_time

    return distance


# ==========================================
# W5-E04
# Linear Velocity -> Distance
# ==========================================

linear_velocity = 0.65

time_intervals = [
    0,
    1,
    2,
    5
]


print("W5-E04: Linear Velocity -> Distance")
print("------------------------------------")


for delta_time in time_intervals:

    distance = calculate_distance(
        linear_velocity,
        delta_time
    )

    print(
        f"Time = {delta_time:3.1f} s | "
        f"Linear velocity = {linear_velocity:.2f} m/s | "
        f"Distance = {distance:.6f} m"
    )