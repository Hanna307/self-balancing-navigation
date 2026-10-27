import numpy as np

A = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, -634.20951051, -0.117913564, 20.59055556],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 5398.24190979, 57.32998454, -175.431904]
])

B = np.array([
    [0.0],
    [3.11179212],
    [0.0],
    [-26.51252483]
])

C_theta = np.array([
    [0.0, 0.0, 1.0, 0.0]
])

C_theta_dot = np.array([
    [0.0, 0.0, 0.0, 1.0]
])

Kp = 10.0
Kd_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
for Kd in Kd_values:

    feedback = Kp * C_theta + Kd * C_theta_dot

    Acl = A + B @ feedback

    eigenvalues = np.linalg.eigvals(Acl)

    max_real = np.max(np.real(eigenvalues))

    print(f"\nKp = {Kp}, Kd = {Kd}")
    print("Eigenvalues:")

    for i, lam in enumerate(eigenvalues, start=1):
        print(f"  lambda_{i} = {lam}")

    print(f"Maximum real part = {max_real}")