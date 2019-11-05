from scipy import interpolate


def std_step_clampon(v_mix, GLR):
    '''
    Standard step calculation for Clampon ASD's
    :param v_mix: fluid mix velocity at ASD [m/s]
    :param GLR: Gas liquid ratio [Sm3/Sm3]
    :return: Standard step value
    '''
    # Standard Step values from Equinor wiki, Jan 18:
    step_v_mix = [0, 1, 2, 3, 4, 6, 8, 12, 16, 22, 30]
    step_oil = [0, 425, 1500, 2500, 3100, 4500, 5500, 8100, 11300, 16100, 22500]
    step_gas = [0, 525, 1500, 2500, 3200, 4800, 6500, 12320, 20740, 38196, 64980]
    # Linear interpolation functions for STEP
    f_oil = interpolate.interp1d(step_v_mix, step_oil, fill_value='extrapolate')
    f_gas = interpolate.interp1d(step_v_mix, step_gas, fill_value='extrapolate')

    if GLR < 150:
        step = f_oil(v_mix)
    else:
        step = f_gas(v_mix)

    if step < 0:
        print('Negative step')
        step = None

    return step


def std_step_emerson(v_mix, GOR):
    '''
    Standard step calculation for Emerson ASD's
    :param v_mix: fluid mix velocity at ASD [m/s]
    :param GOR: Gas oil ratio [Sm3/Sm3]
    :return: Standard step value
    '''

    # Standard Step values from Equinor wiki, Jan 18:
    if GOR > 150:
        E, F, G, H = 3.2, 149.5, 4486.9, 360.8
    else:
        E, F, G, H = 50, 48, 200, 1000

    step = E * v_mix ** 3 + F * v_mix ** 2 + G * v_mix + H

    if step < 0:
        print('Negative step')
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

    if raw > zero:
        try:
            Qs = (raw - zero) / step
        except ZeroDivisionError:
            print('Step value equal to zero, calculation not possible')
            Qs = None
        else:
            if Qs < 0:
                print('Negative step')
                Qs = None
    else:
        Qs = 0

    return Qs