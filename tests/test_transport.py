import pytest
from pysand.transport import hydro, stokes


hydro_validation = [(0.1, 700, 1e-3, .1, [0.52, 0.86]),
                    (.01, 500, 1e-2, .1, [0.2, 0.41])]
@pytest.mark.parametrize('D, rho_l, mu_l, d_p, E', hydro_validation)
def test_hydro(D, rho_l, mu_l, d_p, E):
    assert hydro(D, rho_l, mu_l, d_p) == E


stokes_validation = [(200, 1e-5, .3, 50, .43),
                     (40, 1.5e-5, .1, 20, .27)]
@pytest.mark.parametrize('rho_m, mu_m, d_p, angle, E', stokes_validation)
def test_stokes(rho_m, mu_m, d_p, angle, E):
    assert stokes(rho_m, mu_m, d_p, angle) == E
