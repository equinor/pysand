import numpy as np
import pandas as pd
import logging
import pysand.exceptions as exc

logger = logging.getLogger(__name__)
# Models from DNVGL RP-O501, equation references in parenthesis

def validate_inputs(**kwargs):
    """
    Validation of all input parameters that go into erosion models;
    Besides validating for illegal data input, model parameters are limited within RP-O501 boundaries:
    -------------------------------------------------------------------
    Model parameter         ---   Lower boundary   ---   Upper boundary
    -------------------------------------------------------------------
    Particle diameter       ---        0.02        ---        5     ---
    Pipe inner diameter(D)  ---        0.01        ---        1     ---
    Particle impact angle   ---        0           ---        90    ---
    Bend radius             ---        0.5         ---        50    ---
    Manifold diameter       ---        D           ---              ---
    Heoght of the weld      ---        0           ---        D     ---
    -------------------------------------------------------------------
    Geometry factor can only be 1, 2, 3 or 4
    """

    for i in ['v_m', 'rho_m', 'mu_m', 'Q_s']:
        if i in kwargs:
            if not isinstance(kwargs[i], (float, int, np.ndarray, pd.Series)) or np.isnan(kwargs[i]):
                raise exc.FunctionInputFail('{} is not a number or pandas series'.format(i))

    for j in ['R', 'GF', 'D', 'd_p', 'h', 'Dm', 'D1', 'D2']:
        if j in kwargs:
            if not isinstance(kwargs[j], (int, float)) or np.isnan(kwargs[j]):
                raise exc.FunctionInputFail('{} is not a number'.format(j))

    for k in ['D', 'D1', 'D2']:
        if k in kwargs:
            if (kwargs[k] < 0.01) or (kwargs[k] > 1):
                logger.warning('Pipe inner diameter, {}, is outside RP-O501 model boundaries.'.format(k))

    if 'd_p' in kwargs:
        if (kwargs['d_p'] < 0.02) or (kwargs['d_p'] > 5):
            logger.warning('Particle diameter, d_p, is outside RP-O501 model boundaries.')
    if 'GF' in kwargs:
        if kwargs['GF'] not in [1, 2, 3, 4]:
            logger.warning('Geometry factor, GF, can only be 1, 2, 3 or 4')
    if 'alpha' in kwargs:
        if (kwargs['alpha'] < 0) or (kwargs['alpha'] > 90):
            logger.warning('Particle impact angle [degrees], alpha, is outside RP-O501 model boundaries.')

    # bend
    if 'R' in kwargs:
        if (kwargs['R'] < 0.5) or (kwargs['R'] > 50):
            logger.warning('Bend radius, R, is outside RP-O501 model boundaries.')

    # manifold
    if 'Dm' in kwargs:
        if kwargs['Dm'] < kwargs['D']:
            logger.warning('Manifold diameter, Dm, is expected to be bigger than branch pipe diameter, D')

    # welded joint
    if 'h' in kwargs:
        if (kwargs['h'] < 0) or (kwargs['h'] > kwargs['D']):
            logger.warning('Height of the weld, h, must positive number not exceeding pipe inner diameter size, D')


def bend(v_m, rho_m, mu_m, Q_s, R, GF, D, d_p, K=2e-9, n=2.6, rho_t=7850, rho_p=2650):
    '''
    Particle erosion in bends, model reference to DNVGL RP-O501, August 2015
    :param v_m: Mix velocity [m/s]
    :param rho_m: Mix density [kg/m3]
    :param mu_m: Mix viscosity [kg/ms]
    :param Q_s: Sand production rate [g/s]
    :param R: Bend-radius [# ID's]
    :param GF: Geometry factor [-]
    :param D: Pipe diameter [m]
    :param d_p: Particle diameter [mm]
    :param K: Material erosion constant, default = 2e-9 (duplex steel)
    :param n: Velocity exponent, default = 2.6 (duplex steel)
    :param rho_t: Target material density [kg/m3], default = 7850 (duplex steel)
    :param rho_p: Particle density [kg/m3], default = 2650 (quartz)
    :return: E = Erosion rate [mm/y]
    '''

    # Input validation
    kwargs = {'v_m': v_m, 'rho_m': rho_m, 'mu_m': mu_m, 'Q_s': Q_s, 'GF': GF, 'D': D, 'd_p': d_p}
    validate_inputs(**kwargs)

    # Constants:
    C1 = 2.5  # Model geometry factor for pipe bends

    # Calculations
    a_rad = np.arctan(1 / (2 * R) ** (0.5))  # Pipe bend impact angle [radians] (4.28)
    Apipe = np.pi / 4 * D ** 2  # Pipe cross section area [m2] (4.15)
    At = Apipe / np.sin(a_rad)  # Area exposed to erosion [m2] (4.23)
    gamma = d_p / 1000 / D  # Ratio of particle diameter to geometrical diameter (4.30)
    A = rho_m ** 2 * np.tan(a_rad) * v_m * D / (rho_p * mu_m)  # Dimensionless parameter group (4.29)
    beta = rho_p / rho_m # Dimensionless parameter group (4.29)
    gamma_c = 1 / (beta * (1.88 * np.log(A) - 6.04))  # Relative critical particle diameter (4.30)
    if gamma_c <= 0 or gamma_c > 0.1:
        gamma_c = 0.1
    if gamma < gamma_c: # Particle size correction function (4.31)
        G = gamma / gamma_c
    else:
        G = 1
    # Calculate Erosion rate and return data
    E = K * F(a_rad) * v_m ** n / (rho_t * At) * G * C1 * GF * Q_s * 3600 * 24 * 365.25  # Erosion rate [mm/y] (4.34)
    return E


def tee(v_m, rho_m, mu_m, Q_s, GF, D, d_p, K=2e-9, n=2.6, rho_t=7850, rho_p=2650):
    '''
    Particle erosion in blinded tees, model reference to DNVGL RP-O501, August 2015
    :param v_m: Mix velocity [m/s]
    :param rho_m: Mix density [kg/m3]
    :param mu_m: Mix viscosity [kg/ms]
    :param Q_s: Sand production rate [g/s]
    :param GF: Geometry factor [-]
    :param D: Pipe diameter [m]
    :param d_p: Particle diameter [mm]
    :param K: Material erosion constant, default = 2e-9 (duplex steel)
    :param n: Velocity exponent, default = 2.6 (duplex steel)
    :param rho_t: Target material density [kg/m3], default = 7850 (duplex steel)
    :param rho_p: Particle density [kg/m3], default = 2650 (quartz)
    :return: E: Erosion rate [mm/y]
    '''

    # Input validation
    kwargs = {'v_m': v_m, 'rho_m': rho_m, 'mu_m': mu_m, 'Q_s': Q_s, 'GF': GF, 'D': D, 'd_p': d_p}
    validate_inputs(**kwargs)

    # Calculations
    gamma = d_p / 1000 / D  # Ratio of particle diameter to geometrical diameter (4.37)
    beta = rho_p / rho_m  # Ratio of particle to fluid density (4.38)
    Re = v_m * D * rho_m / mu_m  # Reynolds number (4.39)
    if beta < 40:
        gamma_c = .14 / beta  # Normalised critical particle diameter (4.40)
        if gamma < gamma_c:  # (4.41)
            c = 19 / np.log(Re)
        else:
            c = 0
        C1 = 3 / beta**.3  # Model factor (4.42)
    else:
        b = (np.log(Re/10000+1)+1)**-.6 - 1.2
        gamma_c = 0.0035*(beta/40)**b
        if gamma < gamma_c:  # (4.43)
            c = 19 / np.log(Re)
        else:
            c = -.3*(1-1.01**(40-beta))
        C1 = 1
    G = (gamma/gamma_c)**c  # Particle size correction factor (4.44)
    At = np.pi / 4 * D ** 2  # Characteristic particle impact area [m2] (4.45)
    C_unit = 3600 * 24 * 365.25  # (4.46) 1e-3 lower due to g instead of kg
    E = K * v_m**n / (rho_t * At) * G * C1 * GF * Q_s * C_unit  # Erosion rate [mm/y] (4.48)
    return E


def straight_pipe(v_m, Q_s, D):
    '''
    Particle erosion in smooth and straight pipes, model reference to DNVGL RP-O501, August 2015
    :param v_m: Mix velocity [m/s]
    :param Q_s: Sand production rate [g/s]
    :param D: Pipe diameter [m]
    :return: E: Erosion rate [mm/y]
    '''

    # Input validation
    kwargs = {'v_m': v_m, 'Q_s': Q_s, 'D': D}
    validate_inputs(**kwargs)

    E = 2.5e-5 * v_m**2.6 * D**(-2) * Q_s / 1000
    return E


def welded_joint(v_m, rho_m, Q_s, D, d_p, h, alpha=60, K=2e-9, n=2.6, rho_t=7850):
    '''
    Particle erosion in welded joints, model reference to DNVGL RP-O501, August 2015
    :param v_m: Mix velocity [m/s]
    :param rho_m: Mix density [kg/m3]
    :param Q_s: Sand production rate [g/s]
    :param D: Pipe diameter [m]
    :param d_p: Particle diameter [mm]
    :param h: height of the weld [m]
    :param alpha: particle impact angle [degrees], default = 60
    :param K: Material erosion constant, default = 2e-9 (duplex steel)
    :param n: Velocity exponent, default = 2.6 (duplex steel)
    :param rho_t: Target material density [kg/m3], default = 7850 (duplex steel)
    :return: E_up: Erosion rate flow facing part of weld [mm/year]
    :return: E_down: Erosion rate downstream of weld [mm/year]
    '''

    # Input validation
    kwargs = {'v_m': v_m, 'rho_m': rho_m, 'Q_s': Q_s, 'D': D, 'd_p': d_p, 'h': h}
    validate_inputs(**kwargs)

    A_pipe = np.pi * D**2 / 4
    a_rad = np.deg2rad(alpha)
    At = A_pipe / np.sin(a_rad)  # Area exposed to erosion (4.23)
    C_unit = 3.15e10  # Conversion factor from m/s to mm/year (4.24)
    C2 = 10**6 * d_p / 1000 / (30 * rho_m**.5)  # Particle size and fluid density correction factor (4.25)
    if C2 >= 1:
        C2 = 1
    E_up = K * F(a_rad) * v_m**n * np.sin(a_rad) / (rho_t * A_pipe) * C2 * C_unit * Q_s / 1000
    E_down = 3.3e-2 * (7.5e-4 + h) * v_m**n * D**(-2) * Q_s / 1000
    return E_up, E_down


def F(a_rad):
    '''
    Angle dependency function for ductile materials, reference to DNVGL RP-O501, August 2015 (3.3)
    :param a_rad: impact angle [radians]
    :return: angle dependency
    '''
    A, B, C, k = .6, 7.2, 20, .6
    return A * (np.sin(a_rad) + B * (np.sin(a_rad) - np.sin(a_rad) ** 2))**k * (1 - np.exp(-C * a_rad))


def manifold(v_m, rho_m, mu_m, Q_s, GF, D, d_p, Dm):
    '''
    Manifold model, pending inclusion in DNVGL RP-O501. Velocity and fluid properties in branch line.
    :param v_m: Mix velocity [m/s]
    :param rho_m: Branch line mix density [kg/m3]
    :param mu_m: Branch line mix viscosity [kg/ms]
    :param Q_s: Sand production rate [g/s]
    :param GF: Geometry factor [-]
    :param D: Branch pipe diameter [m]
    :param d_p: Particle diameter [mm]
    :param Dm: Manifold diameter [m]
    :return: Manifold erosion rate [mm/year]
    '''

    # Input validation
    kwargs = {'v_m': v_m, 'rho_m': rho_m, 'mu_m': mu_m, 'Q_s': Q_s, 'GF': GF, 'D': D, 'd_p': d_p, 'Dm': Dm}
    validate_inputs(**kwargs)

    R = Dm / D - 0.5  # Synthetic bend radius
    return bend(v_m, rho_m, mu_m, Q_s, R, GF, D, d_p)


def reducer(v_m, rho_m, Q_s, D1, D2, d_p, GF=2, alpha=60, K=2e-9, n=2.6, rho_t=7850):
    '''
    Particle erosion in reducers, model reference to DNVGL RP-O501, August 2015
    :param v_m: Upstream mix velocity [m/s]
    :param rho_m: Mix density [kg/m3]
    :param Q_s: Sand production rate [g/s]
    :param D1: Upstream pipe diameter [m]
    :param D2: Downstream pipe diameter [m]
    :param d_p: Particle diameter [mm]
    :param GF: Geometry factor [-], default = 2
    :param alpha: particle impact angle [degrees], default = 60 (worst case scenario)
    :param K: Material erosion constant, default = 2e-9 (duplex steel)
    :param n: Velocity exponent, default = 2.6 (duplex steel)
    :param rho_t: Target material density [kg/m3], default = 7850 (duplex steel)
    :return: Reducer erosion rate [mm/year]
    '''

    # Input validation
    kwargs = {'v_m': v_m, 'rho_m': rho_m, 'Q_s': Q_s, 'D1': D1, 'D2': D2, 'GF': GF, 'd_p': d_p}
    validate_inputs(**kwargs)

    a_rad = np.deg2rad(alpha)
    At = np.pi/(4 * np.sin(a_rad))*(D1**2-D2**2)  # Characteristic particle impact area [m2] (4.50)
    Aratio = 1 - (D2/D1)**2  # Area aspect ratio (4.51)
    Up = v_m * (D1/D2)**2  # Characteristic particle velocity [m/s] (4.52)
    C2 = 10 ** 6 * d_p / 1000 / (30 * rho_m ** .5)  # Particle size and fluid density correction factor (4.53)
    if C2 >= 1:
        C2 = 1
    C_unit = 3.15e10  # Conversion factor from m/s to mm/year (4.54)
    E = K * F(a_rad) * Up**n / (rho_t * At) * Aratio * C2 * GF * Q_s / 1000 * C_unit
    return E


def probes(v_m, rho_m, Q_s, D, d_p, alpha=60, K=2e-9, n=2.6, rho_t=7850):
    '''
    Particle erosion for intrusive erosion probes, model reference to DNVGL RP-O501, August 2015
    :param v_m: Upstream mix velocity [m/s]
    :param rho_m: Mix density [kg/m3]
    :param Q_s: Sand production rate [g/s]
    :param D: Branch pipe diameter [m]
    :param d_p: Particle diameter [mm]
    :param alpha: particle impact angle [degrees], default = 60 (worst case scenario)
    :param K: Material erosion constant, default = 2e-9 (duplex steel)
    :param n: Velocity exponent, default = 2.6 (duplex steel)
    :param rho_t: Target material density [kg/m3], default = 7850 (duplex steel)
    :return:
    '''

    # Input validation
    kwargs = {'v_m': v_m, 'rho_m': rho_m, 'Q_s': Q_s, 'D': D, 'd_p': d_p, 'alpha': alpha}
    validate_inputs(**kwargs)

    a_rad = np.deg2rad(alpha)
    At = np.pi / 4 * D**2 * 1 / np.sin(a_rad)  # Eqv particle impact area for homogeneously distributed particles (4.58)
    C2 = 10 ** 6 * d_p / 1000 / (30 * rho_m ** .5)  # Particle size and fluid density correction factor (4.59)
    if C2 >= 1:
        C2 = 1
    C_unit = 3.15e10  # Conversion factor from m/s to mm/year (4.60)
    E = K * F(a_rad) * v_m ** n / (rho_t * At) * C2 * Q_s / 1000 * C_unit
    return E