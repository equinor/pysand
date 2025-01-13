from scipy import interpolate
import numpy as np
import pysand.exceptions as exc
import logging

logger = logging.getLogger(__name__)

def validate_asd(**kwargs: float) -> bool:
    """
    Validation of all input parameters that go into std_step_clampon, std_step_emerson and sand_rate functions;
    """
    for i in ['v_m', 'GLR', 'GOR']:
        if i in kwargs:
            if kwargs[i] is None:
                raise exc.FunctionInputFail(f'No calculation is done due to missing {i}')
            if not isinstance(kwargs[i], (float, int, np.integer)):
                raise exc.FunctionInputFail(f'{i} is not a number')
            if not kwargs[i] >= 0:
                logger.warning(f'The model has got negative value(s) of {i} and returned nan.')
                return True

def std_step_clampon(v_m: float, GLR: float) -> float:
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
    step_v_m = [0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 36, 40, 45, 50]
    step_oil = [0, 425, 1500, 2500, 3100, 4500, 5500, 8100, 11300, 14500, 17700, 20900, 24100, 27300, 30500, 34500, 38500]
    step_gas = [0, 525, 1500, 2500, 3200, 4800, 6500, 12320, 20740, 31500, 44892, 58284, 74664, 94032, 113400, 137610, 161820]
    # Linear interpolation functions for STEP
    f_oil = interpolate.interp1d(step_v_m, step_oil, fill_value='extrapolate')
    f_gas = interpolate.interp1d(step_v_m, step_gas, fill_value='extrapolate')

    if GLR < 150:
        step = f_oil(v_m)
    else:
        step = f_gas(v_m)

    return step


def std_step_emerson(v_m: float, GOR: float) -> float:
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

    return step


def sand_rate(raw: float, zero: float, step: float, exp: float=1) -> float:
    '''
    ASD sand rate calculation
    :param raw: raw value from ASD
    :param zero: Background noise zero value (no sand production)
    :param step: Sand noise
    :param exp: Linearity exponent (calibration constant)
    :return: Sand rate [g/s]
    '''

    for key, value in {'raw': raw, 'zero': zero, 'step': step, 'exp': exp}.items():
        if value is None:
            raise exc.FunctionInputFail(f'No calculation is done due to missing {key}')

    if raw > zero:
        try:
            Qs = (raw - zero)**exp / step
        except ZeroDivisionError:
            logger.warning('Step value equal to zero, sand rate set to NaN')
            Qs = np.nan
        else:
            if Qs < 0:
                logger.warning('Negative step. Sand rate set to NaN')
                Qs = np.nan
            elif zero == 0:
                logger.warning('Zero value is 0. Sand rate set to NaN')
                Qs = np.nan

    else:
        Qs = 0

    return Qs