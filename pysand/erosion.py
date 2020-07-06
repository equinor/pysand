import numpy as np
import logging
import pysand.exceptions as exc

logger = logging.getLogger(__name__)
# Models from DNVGL RP-O501, equation references in parenthesis

def validate_inputs(**kwargs):
    """
    Validation of all input parameters that go into erosion models;
    Besides validating for illegal data input, model parameters are limited within RP-O501 boundaries:
    -------------------------------------------------------------------------
    Model parameter               ---   Lower boundary   ---   Upper boundary
    -------------------------------------------------------------------------
    Mix velocity                  ---        0           ---        200   ---
    Mix density                   ---        1           ---        1500  ---
    Mix viscosity                 ---        1e-6        ---        1e-2  ---
    Particle concentration [ppmV] ---        0           ---        500   ---
    Particle diameter             ---        0.02        ---        5     ---
    Pipe inner diameter(D)        ---        0.01        ---        1     ---
    Particle impact angle         ---        0           ---        90    ---
    Bend radius                   ---        0.5         ---        50    ---
    Manifold diameter             ---        D           ---              ---
    Height of the weld            ---        0           ---        D     ---
    Radius of choke gallery (Rc)  ---        0           ---              ---
    Gap cage and choke body (gap) ---        0           ---        Rc    ---
    Height of gallery (H)         ---        0           ---              ---
    -------------------------------------------------------------------------
    Geometry factor can only be 1, 2, 3 or 4
    """

    if not 'rho_p' in kwargs:
        kwargs['rho_p'] = 2650

    for i in ['v_m', 'rho_m', 'mu_m', 'Q_s']:
        if i in kwargs:
            if not isinstance(kwargs[i], (float, int)) or np.isnan(kwargs[i]):
                raise exc.FunctionInputFail('{} is not a number'.format(i))
            if not kwargs[i] >= 0:
                logger.warning('The model has got negative value(s) of {} and returned nan.'.format(i))
                return True

    if 'v_m' in kwargs:
        if kwargs['v_m'] > 200:
            logger.warning('Mix velocity, v_m, is outside RP-O501 model boundaries (0-200 m/s).')
    if 'rho_m' in kwargs:
        if (kwargs['rho_m'] < 1) or (kwargs['rho_m'] > 1500):
            logger.warning('Mix density, rho_m, is outside RP-O501 model boundaries (1-1500 kg/m3).')
    if 'mu_m' in kwargs:
        if (kwargs['mu_m'] < 1e-6) or (kwargs['mu_m'] > 1e-2):
            logger.warning(
                'Mix viscosity, mu_m, is outside RP-O501 model boundaries (1e-6 - 1e-2 kg/ms).')

    if ('Q_s' in kwargs) and ('rho_p' in kwargs) and ('v_m' in kwargs) and ('D' in kwargs):
        ppmV = kwargs['Q_s'] / (kwargs['rho_p'] * kwargs['v_m'] * np.pi/4*kwargs['D']**2) * 1e3  # (4.20 in RP-O501)
        if (ppmV < 0) or (ppmV > 500):
            logger.warning('The particle concentration is outside RP-O501 model boundaries ( 0-500 ppmV).')

    for j in ['R', 'GF', 'D', 'd_p', 'h', 'Dm', 'D1', 'D2', 'R_c', 'gap', 'H', 'alpha', 'At']:
        if j in kwargs:
            if not isinstance(kwargs[j], (int, float)) or np.isnan(kwargs[j]):
                raise exc.FunctionInputFail('{} is not a number'.format(j))

    for k in ['D', 'D1', 'D2']:
        if k in kwargs:
            if (kwargs[k] < 0.01) or (kwargs[k] > 1):
                logger.warning('Pipe inner diameter, {}, is outside RP-O501 model boundaries (0.01 - 1 m).'.format(k))
            if not kwargs[k] > 0:
                raise exc.FunctionInputFail(' Pipe inner diameter, {}, must be positive'.format(k))

    if 'd_p' in kwargs:
        if (kwargs['d_p'] < 0.02) or (kwargs['d_p'] > 5):
            logger.warning('Particle diameter, d_p, is outside RP-O501 model boundaries (0.02 - 5 mm).')
        if kwargs['d_p'] < 0:
            exc.FunctionInputFail('Particle diameter cannot be negative')
    if 'GF' in kwargs:
        if kwargs['GF'] not in [1, 2, 3, 4]:
            logger.warning('Geometry factor, GF, can only be 1, 2, 3 or 4')

    # bend/choke gallery
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

    # choke gallery
    for l in ['R_c', 'gap', 'H']:
        if l in kwargs:
            if not kwargs[l] > 0:
                raise exc.FunctionInputFail('{} has to be larger than 0'.format(l))
    if 'R_c' in kwargs and 'gap' in kwargs:
        if kwargs['gap'] > kwargs['R_c']:
            raise exc.FunctionInputFail('The gap between the cage and choke body is larger than the radius')

    # Nozzlevalve wall
    if 'model' in kwargs:
        if kwargs['model'] == 'nozzlevalve_wall' and kwargs['d_p'] > 0.6:
            logger.warning('Particle diameter, d_p, is higher than CFD-study boundary (0.6 mm).')


def bend(v_m, rho_m, mu_m, R, GF, D, d_p, material='duplex', rho_p=2650):
    '''
    Particle erosion in bends, model reference to DNVGL RP-O501, August 2015
    :param v_m: Mix velocity [m/s]
    :param rho_m: Mix density [kg/m3]
    :param mu_m: Mix viscosity [kg/ms]
    :param R: Bend-radius [# ID's]
    :param GF: Geometry factor [-]
    :param D: Pipe diameter [m]
    :param d_p: Particle diameter [mm]
    :param material: Material exposed to erosion, default = 'duplex' (duplex steel). For others, run: materials()
    :param rho_p: Particle density [kg/m3], default = 2650 (quartz)
    :return: Relative erosion rate [mm/ton]
    '''

    # Input validation
    kwargs = {'v_m': v_m, 'rho_m': rho_m, 'mu_m': mu_m, 'R': R, 'GF': GF, 'D': D, 'd_p': d_p}
    if validate_inputs(**kwargs):
        return np.nan

    # Constants:
    C1 = 2.5  # Model geometry factor for pipe bends

    rho_t, K, n, ad = material_properties(material)

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
    # Calculate Relative surface thickness loss [mm/t] (4.34)
    E_rel = K * F(a_rad, ad) * v_m ** n / (rho_t * At) * G * C1 * GF * 10 ** 6

    return E_rel


def tee(v_m, rho_m, mu_m, GF, D, d_p, material='duplex', rho_p=2650):
    '''
    Particle erosion in blinded tees, model reference to DNVGL RP-O501, August 2015
    :param v_m: Mix velocity [m/s]
    :param rho_m: Mix density [kg/m3]
    :param mu_m: Mix viscosity [kg/ms]
    :param GF: Geometry factor [-]
    :param D: Pipe diameter [m]
    :param d_p: Particle diameter [mm]
    :param material: Material exposed to erosion, default = 'duplex' (duplex steel). For others, run: materials()
    :param rho_p: Particle density [kg/m3], default = 2650 (quartz)
    :return: E: Relative erosion rate [mm/ton]
    '''

    # Input validation
    kwargs = {'v_m': v_m, 'rho_m': rho_m, 'mu_m': mu_m, 'GF': GF, 'D': D, 'd_p': d_p}
    if validate_inputs(**kwargs):
        return np.nan

    rho_t, K, n, _ = material_properties(material)

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
    E_rel = K * v_m**n / (rho_t * At) * G * C1 * GF * 10 ** 6  # Relative surface thickness loss [mm/t] (4.34)
    return E_rel


def straight_pipe(v_m, D):
    '''
    Particle erosion in smooth and straight pipes, model reference to DNVGL RP-O501, August 2015
    :param v_m: Mix velocity [m/s]
    :param D: Pipe diameter [m]
    :return: E: Relative erosion [mm/ton]
    '''

    # Input validation
    kwargs = {'v_m': v_m, 'D': D}
    if validate_inputs(**kwargs):
        return np.nan
    C_unit = 1000 * 3600 * 24 * 365.25
    E = 2.5e-5 * v_m**2.6 * D**(-2) * (1e6/C_unit)
    return E


def welded_joint(v_m, rho_m, D, d_p, h, alpha=60, location='downstream', material='duplex'):
    '''
    Particle erosion in welded joints, model reference to DNVGL RP-O501, August 2015
    :param v_m: Mix velocity [m/s]
    :param rho_m: Mix density [kg/m3]
    :param D: Pipe diameter [m]
    :param d_p: Particle diameter [mm]
    :param h: height of the weld [m]
    :param alpha: particle impact angle [degrees], default = 60
    :param location: Erosion calculation locations 'downstream' or 'upstream' of weld, default = 'downstream'
    :param material: Material exposed to erosion, default = 'duplex' (duplex steel). For others, run: materials()
    :return: E_up: Relative erosion at flow facing part of weld [mm/ton]
    :return: E_down: Relative erosion downstream of weld [mm/ton]
    '''

    # Input validation
    kwargs = {'v_m': v_m, 'rho_m': rho_m, 'D': D, 'd_p': d_p, 'h': h, 'alpha': alpha}
    if validate_inputs(**kwargs):
        return np.nan

    if (alpha < 0) or (alpha > 90):
            logger.warning('Particle impact angle [degrees], alpha, is outside RP-O501 model boundaries (0-90 deg).')

    rho_t, K, n, ad = material_properties(material)

    A_pipe = np.pi * D**2 / 4
    a_rad = np.deg2rad(alpha)
    C_unit = 3.15e10  # Conversion factor from m/s to mm/year (4.24)
    C2 = 10**6 * d_p / 1000 / (30 * rho_m**.5)  # Particle size and fluid density correction factor (4.25)
    if C2 >= 1:
        C2 = 1
    if location == 'downstream':
        E_down = 3.3e-2 * (7.5e-4 + h) * v_m**n * D**(-2) * (1e6/C_unit)
        return E_down
    elif location == 'upstream':
        E_up = K * F(a_rad, ad) * v_m ** n * np.sin(a_rad) / (rho_t * A_pipe) * C2 * 10**6
        return E_up
    else:
        raise exc.FunctionInputFail('Location must be either downstream or upstream. {} is passed.'.format(location))


def manifold(v_m, rho_m, mu_m, GF, D, d_p, Dm, material='duplex'):
    '''
    Manifold model, pending inclusion in DNVGL RP-O501. Velocity and fluid properties in branch line.
    :param v_m: Mix velocity [m/s]
    :param rho_m: Branch line mix density [kg/m3]
    :param mu_m: Branch line mix viscosity [kg/ms]
    :param GF: Geometry factor [-]
    :param D: Branch pipe diameter [m]
    :param d_p: Particle diameter [mm]
    :param Dm: Manifold diameter [m]
    :param material: Material exposed to erosion, default = 'duplex' (duplex steel). For others, run: materials()
    :return: Manifold relative erosion rate [mm/ton]
    '''

    # Input validation
    kwargs = {'v_m': v_m, 'rho_m': rho_m, 'mu_m': mu_m, 'GF': GF, 'D': D, 'd_p': d_p, 'Dm': Dm}
    if validate_inputs(**kwargs):
        return np.nan

    R = Dm / D - 0.5  # Synthetic bend radius
    return bend(v_m, rho_m, mu_m, R, GF, D, d_p, material=material)  # Relative surface thickness loss [mm/t]


def reducer(v_m, rho_m, D1, D2, d_p, GF=2, alpha=60, material='duplex'):
    '''
    Particle erosion in reducers, model reference to DNVGL RP-O501, August 2015
    :param v_m: Upstream mix velocity [m/s]
    :param rho_m: Mix density [kg/m3]
    :param D1: Upstream pipe diameter [m]
    :param D2: Downstream pipe diameter [m]
    :param d_p: Particle diameter [mm]
    :param GF: Geometry factor [-], default = 2
    :param alpha: particle impact angle [degrees], default = 60 (worst case scenario)
    :param material: Material exposed to erosion, default = 'duplex' (duplex steel). For others, run: materials()
    :return: Reducer relative erosion rate [mm/ton]
    '''

    # Input validation
    kwargs = {'v_m': v_m, 'rho_m': rho_m, 'D1': D1, 'D2': D2, 'GF': GF, 'd_p': d_p, 'alpha': alpha}
    if validate_inputs(**kwargs):
        return np.nan

    if (alpha < 10) or (alpha > 80):
            logger.warning('Particle impact angle [degrees], alpha, is outside RP-O501 model boundaries (10-80 deg).')

    rho_t, K, n, ad = material_properties(material)

    a_rad = np.deg2rad(alpha)
    At = np.pi/(4 * np.sin(a_rad))*(D1**2-D2**2)  # Characteristic particle impact area [m2] (4.50)
    Aratio = 1 - (D2/D1)**2  # Area aspect ratio (4.51)
    Up = v_m * (D1/D2)**2  # Characteristic particle velocity [m/s] (4.52)
    C2 = 10 ** 6 * d_p / 1000 / (30 * rho_m ** .5)  # Particle size and fluid density correction factor (4.53)
    if C2 >= 1:
        C2 = 1

    E_rel = K * F(a_rad, ad) * Up**n / (rho_t * At) * Aratio * C2 * GF * 10**6  # Relative surface thickness loss [mm/t]
    return E_rel


def probes(v_m, rho_m, D, d_p, alpha=60, material='duplex'):
    '''
    Particle erosion for intrusive erosion probes, model reference to DNVGL RP-O501, August 2015
    :param v_m: Upstream mix velocity [m/s]
    :param rho_m: Mix density [kg/m3]
    :param D: Branch pipe diameter [m]
    :param d_p: Particle diameter [mm]
    :param alpha: particle impact angle [degrees], default = 60 (worst case scenario)
    :param material: Material exposed to erosion, default = 'duplex' (duplex steel). For others, run: materials()
    :return: Relative erosion rate [mm/ton]
    '''

    # Input validation
    kwargs = {'v_m': v_m, 'rho_m': rho_m, 'D': D, 'd_p': d_p, 'alpha': alpha}
    if validate_inputs(**kwargs):
        return np.nan

    if (alpha < 10) or (alpha > 90):
            logger.warning('Particle impact angle [degrees], alpha, is outside RP-O501 model boundaries (10-90 deg).')

    rho_t, K, n, ad = material_properties(material)

    a_rad = np.deg2rad(alpha)
    At = np.pi / 4 * D**2 * 1 / np.sin(a_rad)  # Eqv particle impact area for homogeneously distributed particles (4.58)
    C2 = 10 ** 6 * d_p / 1000 / (30 * rho_m ** .5)  # Particle size and fluid density correction factor (4.59)
    if C2 >= 1:
        C2 = 1

    E_rel = K * F(a_rad, ad) * v_m ** n / (rho_t * At) * C2 * 10**6  # Relative surface thickness loss [mm/t]
    return E_rel


def flexible(v_m, rho_m, mu_m, mbr, D, d_p, material='duplex'):
    """
    Particle erosion for flexible pipes with interlock carcass, model reference to DNVGL RP-O501, August 2015
    :param v_m: Mix velocity [m/s]
    :param rho_m: Mix density [kg/m3]
    :param mu_m: Mix viscosity [kg/ms]
    :param mbr: Minimum Bending Radius in operation [# ID's]
    :param D: Minimum internal diameter for the interlock carcass [m]
    :param d_p: Particle diameter [mm]
    :param material: Material exposed to erosion, default = 'duplex' (duplex steel). For others, run: materials()
    :return: Relative erosion rate [mm/ton]
    """

    GF = 2
    E_rel = bend(v_m, rho_m, mu_m, mbr, GF, D, d_p, material=material)  # Relative surface thickness loss [mm/t]
    return E_rel


def choke_gallery(v_m, rho_m, mu_m, GF, D, d_p, R_c, gap, H, material='cr_37_tungsten'):
    """
    Particle erosion for angle style choke gallery, model reference to DNVGL RP-O501, August 2015
    :param v_m: Upstream mix velocity [m/s]
    :param rho_m: Mix density [kg/m3]
    :param mu_m: Mix viscosity [kg/ms]
    :param GF: Geometry factor [-]
    :param D: Upstream pipe diameter [m]
    :param d_p: Particle diameter [mm]
    :param R_c: Radius of the choke gallery [m]
    :param gap: Gap between the cage and choke body [m]
    :param H: Height (effective) of gallery [m]
    :param material: Material exposed to erosion, default = 'cr-37_tungsten' (CR-37 Tungsten Carbide). For others, run: materials()
    :return: Relative erosion rate [mm/ton]
    """

    kwargs = {'R_c': R_c, 'gap': gap, 'H': H}
    if validate_inputs(**kwargs):
        return np.nan

    Ag = 2 * H * gap  # Effective gallery area (table 4-5)
    C1_bend = 2.5  # Model geometry factor for pipe bends
    C1_choke = 1.25  # Model geometry factor for choke gallery (table 4-5)
    Q = v_m * np.pi / 4 * D**2  # Actual flow [m3/s]
    v_c = 3/4 * Q / Ag  # Velocity [m/s] (table 4-5)
    R = R_c/gap  # Checked with DNVGL on e-mail 23.08.17

    # Relative surface thickness loss [mm/t]
    E_rel = bend(v_c, rho_m, mu_m, R, GF, gap, d_p, material=material) / C1_bend * C1_choke

    return E_rel

def nozzlevalve_wall(v_m, d_p, GF, At, material='duplex'):
    """
    Particle valve wall erosion for non-slam nozzle type check-valve. Based on DNVGL CFD-study of Johan Sverdrup Phase 1 check-valves (13.01.2020)
    Report No: 2019-1237 Rev.1, Document No: 547341
    :param v_m: Internal valve mix velocity [m/s]. Use mixture velocity through minimum flow area of the valve.
    :param GF: Geometry factor [-]
    :param At: Target area [m²]. Set to minimum flow area of the valve. 
    :param d_p: Particle diameter [mm]
    :param material: Material exposed to erosion, default = 'duplex'. For others, run: materials()
    :return: Relative erosion rate [mm/ton]
    """
    # Input validation
    kwargs = {'v_m': v_m, 'd_p': d_p, 'GF': GF, 'At': At, 'model': 'nozzlevalve_wall'}
    if validate_inputs(**kwargs):
        return np.nan
    
    C1 = 8.33 * d_p**3 - 29.2 * d_p**2 + 22.8 * d_p + 1 # Model geometry factor
    rho_t, K, n, _ = material_properties(material) # Material properties

    E_rel = K * v_m ** n / (2 * rho_t * At) * C1 * GF * 10 ** 6 # Relative surface thickness loss [mm/t]

    return E_rel

def material_properties(material):
    """
    Function to deal with material properties, reference to table 3-1 in DNVGL RP-O501, August 2015
    :param material: Material. For a full list of materials run: materials()
    :return: rho_t (material density), K (material constant), n (material exponent), angle_dependency
    """

    properties = {'carbon_steel': (7800, 2e-9, 2.6, 'ductile'),
                  'duplex': (7850, 2e-9, 2.6, 'ductile'),
                  'ss316': (8000, 2e-9, 2.6, 'ductile'),
                  'inconel': (8440, 2e-9, 2.6, 'ductile'),
                  'grp_epoxy': (1800, 3e-10, 3.6, 'ductile'),
                  'grp_vinyl_ester': (1800, 6e-10, 3.6, 'ductile'),
                  'hdpe': (1150, 3.5e-9, 2.9, 'ductile'),
                  'aluminium': (2700, 5.8e-9, 2.3, 'ductile'),
                  'dc_05_tungsten': (15250, 1.1e-10, 2.3, 'brittle'),
                  'cs_10_tungsten': (14800, 3.2e-10, 2.2, 'brittle'),
                  'cr_37_tungsten': (14600, 8.8e-11, 2.5, 'brittle'),
                  '95_alu_oxide': (3700, 6.8e-8, 2, 'brittle'),
                  '99_alu_oxide': (3700, 9.5e-7, 1.2, 'brittle'),
                  'psz_ceramic_zirconia': (5700, 4.1e-9, 2.5, 'brittle'),
                  'ZrO2-Y3_ceramic_zirconia': (6070, 4e-11, 2.7, 'brittle'),
                  'SiC_silicon_carbide': (3100, 6.5e-9, 1.9, 'brittle'),
                  'Si3N4_silicon_nitride': (3200, 2e-10, 2, 'brittle'),
                  'TiB2_titanium_diboride': (4250, 9.3e-9, 1.9, 'brittle'),
                  'B4C_boron_carbide': (2500, 3e-8, .9, 'brittle'),
                  'SiSiC_ceramic_carbide': (3100, 7.4e-11, 2.7, 'brittle')}

    if material == 'list':
        return list(properties.keys())

    if material not in properties:
        raise exc.FunctionInputFail('The material {} is not defined. For a full list of materials run materials()'
                                    .format(material))

    rho_t = properties[material][0]
    K = properties[material][1]
    n = properties[material][2]
    angle_dependency = properties[material][3]

    return rho_t, K, n, angle_dependency


def F(a_rad, angle_dependency):
    """
    Angle dependency function, reference to DNVGL RP-O501, August 2015 (3.3)
    :param a_rad: impact angle [radians]
    :param angle_dependency: angle dependency (ductile or brittle)
    :return: ductility factor
    """
    if angle_dependency == 'ductile':
        A, B, C, k = .6, 7.2, 20, .6
        return A * (np.sin(a_rad) + B * (np.sin(a_rad) - np.sin(a_rad) ** 2))**k * (1 - np.exp(-C * a_rad))
    elif angle_dependency == 'brittle':
        return 2 * a_rad / np.pi
    else:
        raise exc.FunctionInputFail('Angle dependency {} is not defined.'.format(angle_dependency))

def materials():
    print('--------------------')
    print('Available materials:')
    print('--------------------')
    return material_properties('list')


def erosion_rate(E_rel, Q_s):
    """
    :param E_rel: Relative Erosion [mm/ton]
    :param Q_s: Sand production rate [g/s]
    :return: Erosion Rate [mm/y]
    """
    kwargs = {'Q_s': Q_s}
    if validate_inputs(**kwargs):
        return np.nan

    C_unit = 1000 * 3600 * 24 * 365.25
    E = (E_rel / 10**6) * (Q_s / 1000) * C_unit  # Annual surface thickness loss [mm/y] (4.35)
    return E