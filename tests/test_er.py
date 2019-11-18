import pytest
from pysand.probe import er_sand_rate

clampon_validation = [(7.231897272, 25, 350, 0.12, .4, pytest.approx(4.64))]
@pytest.mark.parametrize('E_meas, v_m, rho_m, D, d_p, E', clampon_validation)
def test_er_sand_rate(E_meas, v_m, rho_m, D, d_p, E):
    assert er_sand_rate(E_meas, v_m, rho_m, D, d_p) == E
