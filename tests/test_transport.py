import pytest
import numpy as np
import logging
from pysand.transport import hydro, stokes, validate_inputs
import pysand.exceptions as exc

hydro_validation = [(0.1, 700, 1e-3, .1, [0.52, 0.86]),
                    (.01, 500, 1e-2, .1, [0.2, 0.41])]
@pytest.mark.parametrize('D, rho_l, mu_l, d_p, E', hydro_validation)
def test_hydro(D, rho_l, mu_l, d_p, E):
    assert hydro(D, rho_l, mu_l, d_p) == E

    num = 1
    kwargs = {'D': num, 'rho_l': num, 'mu_l': num, 'd_p': num, 'e': num, 'rho_p': num}
    for inp in ['D', 'rho_l', 'mu_l', 'd_p', 'e', 'rho_p']:
        kwargs[inp] = -1
        assert np.isnan(hydro(**kwargs))
        kwargs[inp] = num


stokes_validation = [(200, 1e-5, .3, 50, .43),
                     (40, 1.5e-5, .1, 20, .27)]
@pytest.mark.parametrize('rho_m, mu_m, d_p, angle, E', stokes_validation)
def test_stokes(rho_m, mu_m, d_p, angle, E):
    assert stokes(rho_m, mu_m, d_p, angle) == E

    num = 1
    kwargs = {'rho_m': num, 'mu_m': num, 'd_p': num, 'angle': num}
    for inp in ['rho_m', 'mu_m', 'd_p']:
        kwargs[inp] = -1
        assert np.isnan(stokes(**kwargs))
        kwargs[inp] = num


def test_validate_inputs(caplog):

    # Testing input throws exceptions and warnings for all but angle
    num = 1
    kwargs = {'rho_m': num,
              'rho_l': num,
              'mu_m': num,
              'mu_l': num,
              'd_p': num,
              'angle': num,
              'e': num,
              'rho_p': num,
              'D': num
              }
    for inp in ['rho_m', 'rho_l', 'mu_m', 'mu_l', 'd_p', 'e', 'rho_p', 'D']:
        for non_number in [None, 'string', np.nan]:
            kwargs[inp] = non_number
            with pytest.raises(exc.FunctionInputFail):
                validate_inputs(**kwargs)

        kwargs[inp] = -1
        with caplog.at_level(logging.WARNING):
            validate_inputs(**kwargs)
            assert 'The model got negative value(s) of {} and returned nan.'.format(inp) in str(caplog.records)

        kwargs[inp] = num

    # Test angle exceptions and boundaries
    for non_number in [None, 'string', np.nan]:
        kwargs = {'angle': non_number}
        with pytest.raises(exc.FunctionInputFail):
            validate_inputs(**kwargs)
    for illegal_input in [-1, 90]:
        kwargs = {'angle': illegal_input}
        with pytest.raises(exc.FunctionInputFail):
            validate_inputs(**kwargs)
