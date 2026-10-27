import numpy as np

from sys import path
from pathlib import Path

path.append(str(Path(__file__).resolve().parents[1]))

from model.parameters import params
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
Kp = 1.0
Acl = A + B @ (Kp * C_theta)

eigenvalues = np.linalg.eigvals(Acl) #tính các eigenvalue của ma trận A_cl

print("Kp =", Kp)
print("Closed-loop eigenvalues:")

for i, lam in enumerate(eigenvalues, start=1): # in từng eigenvalue ra màn hình
    print(f"lambda_{i} = {lam}") #hiển thị rõ lambda_1,lambda_2,lambda_3,lambda_4