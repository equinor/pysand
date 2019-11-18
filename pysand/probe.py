from .erosion import probes


def er_sand_rate(E_meas, v_m, rho_m, D, d_p, alpha=60):
    """
    ER probe sand rate calculation, model reference to DNVGL RP-O501, August 2015.
    This approach involve uncertainty, particularly at low bulk flow velocities; Should not be used when v_m < 5 m/s.
    :param E_meas: Measured erosion rate from ER probe [mm/year]
    :param v_m: Upstream mix velocity [m/s]
    :param rho_m: Mix density [kg/m3]
    :param D: Branch pipe diameter [m]
    :param d_p: Particle diameter [mm]
    :param alpha: particle impact angle [degrees], default = 60
    :return: Sand rate [g/s]
    """

    E_theor = probes(v_m, rho_m, 1, D, d_p, alpha)
    Q_s = E_meas / E_theor  # (4.63)

    return Q_s