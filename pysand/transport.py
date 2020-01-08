from scipy import optimize
import numpy as np

# Constants
g = 9.80665  # Standard gravity [m/s^2]


def hydro(D, rho_l, mu_l, d_p, e=5e-5, rho_p=2650):
    """
    Equinor sand transport model for horizontal pipelines
    Based on T. Søntvedt (1995) and R. Schulkes (2002) work in Hydro
    :param D: Pipe diameter [m]
    :param rho_l: Liquid density [kg/m3]
    :param mu_l: Dynamic liquid viscosity [Pa.s]
    :param d_p: Sand grain diameter [mm]
    :param e: Roughness of pipe [m] Optional. Default value: 5e-5 m
    :param rho_p: Particle density [kg/m3], default = 2650 (quartz sand)
    :return: list of critical velocities for transitions from stationary to moving bed
    and from moving bed to dispersed transport [m/s]
    """
    d_p = d_p / 1000  # from mm to m
    nu = mu_l / rho_l  # Liquid kinematic viscotity [m2/s]
    result = []

    def moody(ed, Re):
        '''
        Finds darcy friction factor by solving Colebrook equation (Moody diagram).
        Laminar and turbulent flow are correctly accounted for.
        :param ed: relative roughness = epsilon/diameter
        :param Re: Reynolds number
        :return: friction factor
        '''

        if Re < 0:
            print('Reynolds number =', Re, '. Cannot be negative')
        if Re < 2000:
            f = 64 / Re  # Laminar flow
            return f
        if ed > 0.05:
            print('epsilon/diameter ratio =', ed, 'is not on Moody chart')
        if Re < 4000:
            print('Re = ', Re, ' in transition range')

        # Use fzero to find f from the Colebrook equation.
        # colebrook is a function to evaluate F(f,e/d,Re)
        # fzero returns the value of f such that F(f,e/d/Re) = 0 (approximately)
        # fi = initial guess from Haaland equation
        # Iterations of fzero are terminated when f is known to within +/- 5e-6

        def colebrook(f, ed, Re):
            cb = 1.0 / f ** 0.5 + 2.0 * np.log10(ed / 3.7 + 2.51 / (Re * f ** 0.5))
            return cb

        fi = 1 / (1.8 * np.log10(6.9 / Re + (ed / 3.7) ** 1.11)) ** 2  # initial guess at f
        fzero = optimize.newton(colebrook, fi, args=(ed, Re), tol=5e-6)

        # sanity check:
        if fzero < 0:
            print('Friction factor = ', fzero, ', but cannot be negative')
        return fzero

    def critical_u(u, m):
        Re = (u * D) / nu  # Reynolds number
        f_turb = 0.25 * moody(e / D, Re)  # Fanning friction factor
        t_w = .5 * f_turb * rho_l * u ** 2  # Wall shear stress [Pa]
        uf = (t_w / rho_l) ** 0.5  # Friction velocity
        ds = d_p * (uf / nu)  # Dimensionless particle diameter
        SH = t_w / ((rho_p - rho_l) * g * d_p)  # Shield number
        SH_bed = 0.42 / ((ds) ** 0.49)  # Critical Shield number between fixed bed and scouring
        SH_scouring = 1.12 / ((ds) ** 0.37)  # Critical Shield number between scouring and dispersed
        if m == 'bed':
            return SH - SH_bed
        if m == 'scouring':
            return SH - SH_scouring

    for i in ['bed', 'scouring']:
        vel = optimize.newton(critical_u, 1, args=(i,), tol=5e-6)
        result.append(np.round(vel, 2))

    return result


def stokes(rho_m, mu_m, d_p, angle, rho_p=2650):
    """
    Sand lifting rate calculation model for vertical or deviated (up to ~80 deg) inclination
    Based on Stokes law for settling velocities, but corrected for turbulent flow around sand grains
    Mixture fluid properties and inclination at point of interest (e.g. down hole in a well)
    :param rho_m: Mixture density [kg/m3]
    :param mu_m: Mixture viscosity [kg/ms]
    :param d_p: Particle diameter [mm]
    :param angle: Inclination from vertical [deg]
    :param rho_p: Particle density [kg/m3], default = 2650 (quartz sand)
    :return: Particle settling velocity at point of interest [m/s]
    """
    def f(v_est):
        Re = rho_m * d_p/1000 / mu_m * v_est
        if Re > 1899:
            f = .4
        else:
            f = 10 ** (1.413 - 0.923 * np.log10(Re) + 0.113 * np.log10(Re)**2)  # Polynomial friction factor
        A1 = 4 * g * d_p/1000 * (rho_p - rho_m) / (3 * rho_m * np.cos(np.deg2rad(angle)))
        v_c = (A1 / f) ** 0.5  # Critical velocity
        return abs(v_est - v_c)
    result = optimize.minimize_scalar(f, method='bounded', bounds=([0.001, 100]))
    if result.success:
        return np.round(result.x, 2)
