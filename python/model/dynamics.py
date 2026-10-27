import numpy as np #Thư viện Numpy để tạo ma trận, vecto và thực hiện tính toán số
from .parameters import params #lấy bộ thông số robot ở file parameters.py
def nonlinear_robot_dynamics(t, state, tau, params=params): #hàm chính của mô hình với t: thời gian, state: trạng thái robot, tau: tổng mô men tác động, params: thông số robot
        x, x_dot, theta, theta_dot = state
        m_B = params['m_B'] #lấy các giá trị từ parameters.py
        m_W = params['m_W']
        I_B = params['I_B']
        I_W = params['I_W']
        l   = params['l']
        r   = params['r']
        b_W = params['b_W']
        g   = params['g']
        M11 = m_B + 2*m_W + 2*I_W/(r**2) #tạo ma trận M(q)
        M12 = m_B*l*np.cos(theta)
        M21 = M12
        M22 = m_B*l**2 + I_B

        M = np.array([
        [M11, M12],
        [M21, M22]
    ])
        C = np.array([ #tạo vecto phi tuyến C
        2*b_W/(r**2)*x_dot
        - m_B*l*(theta_dot**2)*np.sin(theta),
        0.0
    ])
        G = np.array([ #tạo vecto  trọng lực G(q)
        0.0,
        -m_B*g*l*np.sin(theta)
    ])
        B_tau = np.array([ #tạo vecto mô men tác động
        tau/r,
        -tau
    ])
        q_ddot = np.linalg.solve( #giải phương trình để tìm gia tốc
        M,
        B_tau - C - G
    )
        return [ #trả về đạo hàm trạng thái
        x_dot, 
        q_ddot[0],
        theta_dot,
        q_ddot[1]
    ]