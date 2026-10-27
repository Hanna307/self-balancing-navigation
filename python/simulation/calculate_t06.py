import numpy as np

m_B = 0.9
m_W = 0.035
I_B = 0.0045
I_W = 0.0000185

l = 0.075
r = 0.0325
b_W = 0.001
g = 9.81

M_x = m_B + 2*m_W + 2*I_W/(r**2)
M_xtheta = m_B*l
M_theta = m_B*l**2 + I_B
D_x = 2*b_W/(r**2)

Delta = M_x*M_theta - M_xtheta**2

a1 = -M_theta*D_x/Delta
a2 = -M_xtheta*m_B*g*l/Delta
a3 = M_xtheta*D_x/Delta
a4 = M_x*m_B*g*l/Delta

b1 = (M_theta/r + M_xtheta)/Delta
b2 = -(M_xtheta/r + M_x)/Delta

print("=== T06 - State-Space Coefficients ===")
print()
print("a1 =", a1)
print("a2 =", a2)
print("a3 =", a3)
print("a4 =", a4)
print()
print("b1 =", b1)
print("b2 =", b2)
A = np.array([
    [0, 1, 0, 0],
    [0, a1, a2, 0],
    [0, 0, 0, 1],
    [0, a3, a4, 0]
])

B = np.array([
    [0],
    [b1],
    [0],
    [b2]
])

eigenvalues = np.linalg.eigvals(A)

print()
print("=== T06.3 - Eigenvalues of A ===")
for i, eigenvalue in enumerate(eigenvalues):
    print(f"lambda_{i+1} =", eigenvalue)
C_control = np.hstack([
    B,
    A @ B,
    A @ A @ B,
    A @ A @ A @ B
])

rank_control = np.linalg.matrix_rank(C_control)

print()
print("=== T06.4 - Controllability Check ===")
print("Controllability matrix:")
print(C_control)
print()
print("Rank =", rank_control)