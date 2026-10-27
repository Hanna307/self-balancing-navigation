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

    angle = 2 * np.pi * count / cpr

    return angle


# ==========================================
# W5-E01
# Encoder Count -> Wheel Angle
# ==========================================

cpr = 100

counts = [0, 25, 50, 75, 100, 200]


print("W5-E01: Encoder Count -> Wheel Angle")
print("--------------------------------------")


for count in counts:

    angle = encoder_count_to_angle(count, cpr)

    revolutions = count / cpr

    print(
        f"Count = {count:3d} | "
        f"Angle = {angle:.6f} rad | "
        f"Revolutions = {revolutions:.2f}"
    )