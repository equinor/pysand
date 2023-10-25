import numpy as np
import logging
import pysand.exceptions as exc

logger = logging.getLogger(__name__)
# Models from DNVGL RP-O501, equation references in parenthesis

def validate_fluid_props(**kwargs):
    """
        Validation of all input parameters that go into fluid properties models
    """
    for i in ['P', 'T', 'Qo', 'Qw', 'Qg', 'Z', 'D', 'rho_o', 'rho_w', 'MW', 'mu_o', 'mu_w', 'mu_g']:
        if i in kwargs:
            if kwargs[i] is None:
                raise exc.FunctionInputFail('No fluid properties are calculated due to missing {}'.format(i))
            if not isinstance(kwargs[i], (float, int)):
                raise exc.FunctionInputFail('{} is not a number'.format(i))
            if not kwargs[i] >= 0:
                logger.warning('The model has got negative value(s) of {} and returned nan.'.format(i))
                return True

def mix_velocity(P: float, T: float, Qo: float, Qw: float, Qg: float, Z: float, D: float) -> float:
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

    kwargs = {'P': P, 'T': T, 'Qo': Qo, 'Qw': Qw, 'Qg': Qg, 'Z': Z, 'D': D}
    if validate_fluid_props(**kwargs):
        return np.nan

    T = T + 273.15
    # Constants
    P0 = 1.01325  # Pressure at std conditions [bar]
    T0 = 289.0  # Temperature at std conditions [K]

    # Calculations
    v_m = (Qo + Qw + Qg * Z * P0 * T / (P * T0)) / (np.pi / 4 * D ** 2) / 24 / 3600  # (4.14, 4.15, 4.16)
    return np.round(v_m, 2)


def mix_density(P: float, T: float, Qo: float, Qw: float, Qg: float, 
                rho_o: float, rho_w: float, MW: float, Z: float) -> float:
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

    kwargs = {'P': P, 'T': T, 'Qo': Qo, 'Qw': Qw, 'Qg': Qg, 'rho_o': rho_o, 'rho_w': rho_w, 'MW': MW, 'Z': Z}
    if validate_fluid_props(**kwargs):
        return np.nan

    T = T + 273.15
    # Constants
    P0 = 1.01325  # Pressure at std conditions [bar]
    T0 = 289.0  # Temperature at std conditions [K]
    R = 8314.0  # Universal gas constant [J/kgK]

    # Calculations
    rho_m = (Qo * rho_o + Qw * rho_w + Qg * P0 * MW / (R * T0) * 1e5) / (Qo + Qw + Qg * Z * P0 * T / (P * T0))  # (4.17)
    return np.round(rho_m, 2)


def mix_viscosity(P: float, T: float, Qo: float, Qw: float, Qg: float, mu_o: float, mu_w: float, mu_g: float, Z: float) -> float:
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

    kwargs = {'P': P, 'T': T, 'Qo': Qo, 'Qw': Qw, 'Qg': Qg, 'mu_o': mu_o, 'mu_w': mu_w, 'mu_g': mu_g, 'Z': Z}
    if validate_fluid_props(**kwargs):
        return np.nan

    T = T + 273.15
    # Constants
    P0 = 1.01325  # Pressure at std conditions [bar]
    T0 = 289.0  # Temperature at std conditions [K]

    # Calculations
    mu_m = (mu_o + Qw/Qo * mu_w + Qg/Qo*P0*T*Z / (P * T0) * mu_g) / (1 + Qw/Qo + Qg/Qo*P0*T*Z / (P * T0))
    return np.round(mu_m, 6)


def liq_density(Qo: float, Qw: float, rho_o: float, rho_w: float) -> float:
    """
    Weighted average of oil and water densities
    :param Qo: Oil rate [Sm3/d]
    :param Qw: Water rate [Sm3/d]
    :param rho_o: Oil density at std conditions [kg/m3]
    :param rho_w: Water density at std conditions [kg/m3]
    :return: Liquid density
    """

    kwargs = {'Qo': Qo, 'Qw': Qw, 'rho_o': rho_o, 'rho_w': rho_w}
    if validate_fluid_props(**kwargs):
        return np.nan

    rho_l = (rho_o * Qo + rho_w * Qw) / (Qo + Qw)

    return rho_l


def liq_viscosity(Qo: float, Qw: float, mu_o: float, mu_w: float) -> float:
    """
    Weighted average of oil and water viscosities
    :param Qo: Oil rate [Sm3/d]
    :param Qw: Water rate [Sm3/d]
    :param mu_o: Oil viscosity
    :param mu_w: Water viscosity
    :return: Liquid viscosity
    """

    kwargs = {'Qo': Qo, 'Qw': Qw, 'mu_o': mu_o, 'mu_w': mu_w}
    if validate_fluid_props(**kwargs):
        return np.nan

    mu_l = (mu_o * Qo + mu_w * Qw) / (Qo + Qw)

    return mu_l

