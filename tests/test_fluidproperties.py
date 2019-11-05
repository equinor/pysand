import pytest
from pysand.fluidproperties import mix_velocity, mix_density, mix_viscosity


velocity_validation = [(10, 40, 1200, 3000, 100000, .95, .15, 9.58)]
@pytest.mark.parametrize('P, T, Qo, Qw, Qg, Z, D, E', velocity_validation)
def test_mix_velocity(P, T, Qo, Qw, Qg, Z, D, E):
    assert mix_velocity(P, T, Qo, Qw, Qg, Z, D) == E


density_validation = [(30, 60, 300, 4000, 150000, 700, 1000, 19, .95, 439.69)]
@pytest.mark.parametrize('P, T, Qo, Qw, Qg, rho_o, rho_w, MW, Z, E', density_validation)
def test_mix_density(P, T, Qo, Qw, Qg, rho_o, rho_w, MW, Z, E):
    assert mix_density(P, T, Qo, Qw, Qg, rho_o, rho_w, MW, Z) == E


viscosity_validation = [(40, 80, 4000, 300, 400000, 2, 1, 1e-3, .9, 0.54)]
@pytest.mark.parametrize('P, T, Qo, Qw, Qg, mu_o, mu_w, mu_g, Z, E', viscosity_validation)
def test_mix_viscosity(P, T, Qo, Qw, Qg, mu_o, mu_w, mu_g, Z, E):
    assert mix_viscosity(P, T, Qo, Qw, Qg, mu_o, mu_w, mu_g, Z) == E