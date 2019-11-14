import numpy as np

# Models from DNVGL RP-O501, equation references in parenthesis


def mix_velocity(P, T, Qo, Qw, Qg, Z, D):
    '''
    Blackoil mixture velocity model, based on DNVGL RP-O501, August 2015 edition
    :param P: Pressure [bar]
    :param T: Temperature [deg C]
    :param Qo: Oil rate [Sm3/d]
    :param Qw: Water rate [Sm3/d]
    :param Qg: Gas rate [Sm3/d]
    :param Z: Gas compressibility factor [-]
    :param D: Cross sectional diameter [m]
    :return: Mix velocity [m/s]
    '''
    T = T + 273.15
    # Constants
    P0 = 1.01325  # Pressure at std conditions [bar]
    T0 = 289.0  # Temperature at std conditions [K]

    # Calculations
    v_m = (Qo + Qw + Qg * Z * P0 * T / (P * T0)) / (np.pi / 4 * D ** 2) / 24 / 3600  # (4.14, 4.15, 4.16)
    return np.round(v_m, 2)


def mix_density(P, T, Qo, Qw, Qg, rho_o, rho_w, MW, Z):
    '''
    Blackoil mixture density calculator, based on DNVGL RP-O501, August 2015 edition
    :param P: Pressure [bar]
    :param T: Temperature [deg C]
    :param Qo: Oil rate [Sm3/d]
    :param Qw: Water rate [Sm3/d]
    :param Qg: Gas rate [Sm3/d]
    :param rho_o: Oil density at std conditions [kg/m3]
    :param rho_w: Water density at std conditions [kg/m3]
    :param MW: Gas molecular weight [kg/kmol]
    :param Z: Gas compressibility factor [-]
    :return: Mix density [kg/m3]
    '''
    T = T + 273.15
    # Constants
    P0 = 1.01325  # Pressure at std conditions [bar]
    T0 = 289.0  # Temperature at std conditions [K]
    R = 8314.0  # Universal gas constant [J/kgK]

    # Calculations
    rho_m = (Qo * rho_o + Qw * rho_w + Qg * P0 * MW / (R * T0) * 1e5) / (Qo + Qw + Qg * Z * P0 * T / (P * T0))  # (4.17)
    return np.round(rho_m, 2)


def mix_viscosity(P, T, Qo, Qw, Qg, mu_o, mu_w, mu_g, Z):
    '''
    Blackoil mixture viscosity calculator, based on DNVGL RP-O501, August 2015 edition
    Output viscosity units equals input units
    :param P: Pressure [bar]
    :param T: Temperature [deg C]
    :param Qo: Oil rate [Sm3/d]
    :param Qw: Water rate [Sm3/d]
    :param Qg: Gas rate [Sm3/d]
    :param mu_o: Oil viscosity
    :param mu_w: Water viscosity
    :param mu_g: Gas viscosity
    :param Z: Gas compressibility factor [-]
    :return: Mixture viscosity
    '''
    T = T + 273.15
    # Constants
    P0 = 1.01325  # Pressure at std conditions [bar]
    T0 = 289.0  # Temperature at std conditions [K]

    # Calculations
    mu_m = (mu_o + Qw/Qo * mu_w + Qg/Qo*P0*T*Z / (P * T0) * mu_g) / (1 + Qw/Qo + Qg/Qo*P0*T*Z / (P * T0))
    return np.round(mu_m, 6)