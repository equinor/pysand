from scipy import interpolate
import numpy as np
import pysand.exceptions as exc
import logging

logger = logging.getLogger(__name__)

def validate_asd(**kwargs):
    """
    Validation of all input parameters that go into std_step_clampon, std_step_emerson and sand_rate functions;
    """
    for i in ['v_m', 'GLR', 'GOR']:
        if i in kwargs:
            if kwargs[i] is None:
                raise exc.FunctionInputFail('No calculation is done due to missing {}'.format(i))
            if not isinstance(kwargs[i], (float, int)):
                raise exc.FunctionInputFail('{} is not a number'.format(i))
            if not kwargs[i] >= 0:
                logger.warning('The model has got negative value(s) of {} and returned nan.'.format(i))
                return True

def std_step_clampon(v_m, GLR):
    '''
    Standard step calculation for Clampon ASD's
    :param v_m: fluid mix velocity at ASD [m/s]
    :param GLR: Gas liquid ratio [Sm3/Sm3]
    :return: Standard step value
    '''

    kwargs = {'v_m': v_m, 'GLR': GLR}
    if validate_asd(**kwargs):
        return np.nan

    # Standard Step values from Equinor wiki, Jan 18:
    step_v_m = [0, 1, 2, 3, 4, 6, 8, 12, 16, 22, 30]
    step_oil = [0, 425, 1500, 2500, 3100, 4500, 5500, 8100, 11300, 16100, 22500]
    step_gas = [0, 525, 1500, 2500, 3200, 4800, 6500, 12320, 20740, 38196, 64980]
    # Linear interpolation functions for STEP
    f_oil = interpolate.interp1d(step_v_m, step_oil, fill_value='extrapolate')
    f_gas = interpolate.interp1d(step_v_m, step_gas, fill_value='extrapolate')

    if GLR < 150:
        step = f_oil(v_m)
    else:
        step = f_gas(v_m)

    if step < 0:
        logger.warning('Negative step')
        step = None

    return step


def std_step_emerson(v_m, GOR):
    '''
    Standard step calculation for Emerson ASD's
    :param v_m: fluid mix velocity at ASD [m/s]
    :param GOR: Gas oil ratio [Sm3/Sm3]
    :return: Standard step value
    '''

    kwargs = {'v_m': v_m, 'GOR': GOR}
    if validate_asd(**kwargs):
        return np.nan

    # Standard Step values from Equinor wiki, Jan 18:
    if GOR > 150:
        E, F, G, H = 3.2, 149.5, 4486.9, 360.8
    else:
        E, F, G, H = 50, 48, 200, 1000

    step = E * v_m ** 3 + F * v_m ** 2 + G * v_m + H

    if step < 0:
        logger.warning('Negative step')
        step = None

    return step


def sand_rate(raw, zero, step):
    '''
    ASD sand rate calculation
    :param raw: raw value from ASD
    :param zero: Background noise zero value (no sand production)
    :param step: Sand noise
    :return: Sand rate [g/s]
    '''

    for key, value in {'raw': raw, 'zero': zero, 'step': step}.items():
        if value is None:
            raise exc.FunctionInputFail('No calculation is done due to missing {}'.format(key))

    if raw > zero:
        try:
            Qs = (raw - zero) / step
        except ZeroDivisionError:
            logger.warning('Step value equal to zero, calculation not possible')
            Qs = None
        else:
            if Qs < 0:
                logger.warning('Negative step')
                Qs = None
    else:
        Qs = 0

    return Qs