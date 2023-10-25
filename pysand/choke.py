def critical_velocity(p1: float, p2: float, rho_1: float, rho_2: float) -> float:
    """
    Throttling velocity in the choke
    DNVGL RP-O501, August 2015 - chapter 4.12.3
    :param p1: Choke upstream pressure [bar]
    :param p2: Choke downstream pressure [bar]
    :param rho_1: Fluid mixture density upstream choke [kg/m3]
    :param rho_2: Fluid mixture density downstream choke [kg/m3]
    :return: Throttling velocity [m/s]
    """
    v_c = (4e5 * (p1 - p2) / (rho_1 + rho_2)) ** (1/2)
    return v_c


def min_choke(Qs: float, v_c: float) -> float:
    """
    Recommended minimum choke opening for plug/cage and cage/sleeve types of chokes
    DNVGL RP-O501, August 2015 - chapter 4.12.3
    :param Qs: Sand rate [g/s]
    :param v_c: Throttling velocity [m/s]
    :return: Minimum relative Cv [-]
    """
    if Qs < 0.01:
        if v_c < 100:
            cvr_min = 0.05
        else:
            cvr_min = 0.1
    elif 0.01 <= Qs < 0.1:
        if v_c < 50:
            cvr_min = 0.05
        elif 50 <= v_c < 100:
            cvr_min = 0.1
        else:
            cvr_min = 0.2
    else:
        if v_c < 50:
            cvr_min = 0.1
        else:
            cvr_min = 0.2
    return cvr_min
