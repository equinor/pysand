import pytest
import numpy as np
import logging
import pysand.exceptions as exc
from pysand.asd import std_step_clampon, std_step_emerson, sand_rate, validate_asd


# Testing illegal input throws exception
def test_validate_asd(caplog):
    v_m = 29
    GLR = 100
    GOR = 800

    kwargs = {'v_m': v_m, 'GLR': GLR, 'GOR': GOR}
    for inp in ['v_m', 'GLR', 'GOR']:
        for non_number in [None, 'string']:
            kwargs[inp] = non_number
            with pytest.raises(exc.FunctionInputFail) as excinfo:
                validate_asd(**kwargs)

        kwargs[inp] = -1
        with caplog.at_level(logging.WARNING):
            validate_asd(**kwargs)
        assert "The model has got negative value(s) of {} and returned nan.".format(inp) in str(caplog.records)

        kwargs['v_m'] = v_m
        kwargs['GLR'] = GLR
        kwargs['GOR'] = GOR

clampon_validation = [(0, 50, 0),
                      (20, 70, 14500),
                      (50, 90, 38500),
                      (-10, 400, None),
                      (22, 400, 38196),
                      (70, 500, 198900)]
@pytest.mark.parametrize('v_mix, GLR, E', clampon_validation)
def test_clampon(v_mix, GLR, E):
    assert std_step_clampon(v_mix, GLR) == E


emerson_validation = [(0, 50, 1000),
                    (20, 70, 424200),
                    (50, 90, 6381000),
                    (-10, 400, None),
                    (20, 400, 175498.8),
                    (70, 500, 2144593.8)]
@pytest.mark.parametrize('v_mix, GOR, E', emerson_validation)
def test_emerson(v_mix, GOR, E):
    assert std_step_emerson(v_mix, GOR) == E


sand_rate_validation = [(5000, 5000, 2000, 1, 0),
                        (4000, 2000, 1000, 1, 2),
                        (2000, 4000, 500, 1, 0),
                        (2000, 1000, 0, 1, None),
                        (2000, 1000, -5000, 1, None),
                        (4000, 1500, 8000, 1.6, pytest.approx(34.17, rel=1e-2))]
@pytest.mark.parametrize('raw, zero, step, exp, E', sand_rate_validation)
def test_sand_rate(raw, zero, step, exp, E):
    assert sand_rate(raw, zero, step, exp=exp) == E


def test_sand_rate_None_input():
    raw = 5000
    zero = 3000
    step = 500
    exp = 1
    kwargs = {'raw': raw, 'zero': zero, 'step': step, 'exp': exp}
    for inp in ['raw', 'zero', 'step', 'exp']:
        kwargs[inp] = None
        with pytest.raises(exc.FunctionInputFail) as excinfo:
            sand_rate(**kwargs)

        kwargs['raw'] = raw
        kwargs['zero'] = zero
        kwargs['step'] = step
        kwargs['exp'] = exp
