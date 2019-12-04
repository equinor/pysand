import pytest
from pysand.probe import er_sand_rate, validate_inputs
import pysand.exceptions as exc
import logging

def test_validate_inputs(caplog):

    # Testing input throws exception throws exception
    E_meas = 7.23
    v_m = 3

    kwargs = {'E_meas': E_meas, 'v_m': v_m}
    for inp in ['E_meas', 'v_m']:
        for non_number in [None, 'string']:
            kwargs[inp] = non_number
            with pytest.raises(exc.FunctionInputFail) as excinfo:
                validate_inputs(**kwargs)

        kwargs['E_meas'] = E_meas
        kwargs['v_m'] = v_m

        with caplog.at_level(logging.WARNING):
            validate_inputs(**kwargs)
        assert "Mix velocity, v_m = {} m/s, is too low (< 5 m/s) to trust the quantification model.".format(kwargs['v_m']) \
               in str(caplog.records)



er_sand_rate_validation = [(7.231897272, 25, 350, 0.12, .4, pytest.approx(4.632, abs=1e-3))]
@pytest.mark.parametrize('E_meas, v_m, rho_m, D, d_p, E', er_sand_rate_validation)
def test_er_sand_rate(E_meas, v_m, rho_m, D, d_p, E):
    assert er_sand_rate(E_meas, v_m, rho_m, D, d_p) == E
