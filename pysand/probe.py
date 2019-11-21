from pysand.erosion import probes
import logging
import pysand.exceptions as exc
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)


def validate_inputs(**kwargs):
    """
    Validation of all input parameters that go into probe models;
    Besides validating for illegal data input, model parameters are limited within RP-O501 boundaries:
    ----------------------------------------------------------------------------
    Model parameter                  ---   Lower boundary   ---   Upper boundary
    ----------------------------------------------------------------------------
    Measured erosion rate (E_meas)   ---        0           ---              ---
    Mix velocity (v_m)               ---        5           ---              ---
    ----------------------------------------------------------------------------
    """

    for i in kwargs:
        if i in ['E_meas', 'v_m']:
            if not isinstance(kwargs[i], (float, int, np.ndarray, pd.Series)) or np.isnan(kwargs[i]):
                raise exc.FunctionInputFail('{} is not a number or pandas series'.format(i))
            if kwargs[i] < 0:
                raise exc.FunctionInputFail('{} cannot be negative.'.format(i))
            if i == 'v_m':
                if kwargs[i] < 5:
                    logger.warning('Mix velocity, v_m = {} m/s, is too low (< 5 m/s) '
                                   'to trust the quantification model.'.format(kwargs[i]))


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

    # Input validation
    kwargs = {'E_meas': E_meas, 'v_m': v_m, 'rho_m': rho_m, 'D': D, 'd_p': d_p, 'alpha': alpha}
    validate_inputs(**kwargs)

    E_theor = probes(v_m, rho_m, 1, D, d_p, alpha)
    Q_s = E_meas / E_theor  # (4.63)

    return Q_s
