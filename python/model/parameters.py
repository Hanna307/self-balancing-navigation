# Physical parameters of the self-balancing robot

params = {

    # Mass [kg]
    'm_B': 0.9,
    'm_W': 0.035,

    # Moment of inertia [kg.m^2]
    'I_B': 0.0045,
    'I_W': 0.0000185,

    # Geometry [m]
    'l': 0.075,
    'r': 0.0325,

    # Wheel damping coefficient
    'b_W': 0.001,

    # Gravity [m/s^2]
    'g': 9.81,
    # ========================================================
    # 2. Actuator Subsystem
    # Assumed parameters for simulation
    # ========================================================

    # Armature resistance [Ohm]
    'R': 2.5,

    # Motor constant: Kt = Ke
    'K': 0.012,

    # Gearbox ratio [-]
    'n': 30,

    # Motor rotor inertia [kg.m^2]
    'I_M': 1.0e-6,

    # Gearbox inertia [kg.m^2]
    'I_G': 1.0e-4,

    # Motor viscous damping [N.m.s/rad]
    'b_M': 0.001,

    # Gearbox viscous damping [N.m.s/rad]
    'b_G': 0.001,

    # Maximum supply voltage [V]
    'V_max': 12.0
}