import pytest
import pysand.exceptions as exc
from pysand.fluidproperties import mix_velocity, mix_density, mix_viscosity, validate_fluid_props


def test_validate_fluid_props():
    # Testing non-number input throws exception
    kwargs = {'P': 1, 'T': 1, 'Qo': 1, 'Qw': 1, 'Qg': 1, 'Z': 1, 'D': 1, 'rho_o': 1, 'rho_w': 1, 'MW': 1,
              'mu_o': 1, 'mu_w': 1, 'mu_g': 1}
    for inp in ['P', 'T', 'Qo', 'Qw', 'Qg', 'Z', 'D', 'rho_o', 'rho_w', 'MW', 'mu_o', 'mu_w', 'mu_g']:
        for non_number in [None, 'string']:
            kwargs[inp] = non_number
            with pytest.raises(exc.FunctionInputFail) as excinfo:
                validate_fluid_props(**kwargs)


velocity_validation = [(10, 40, 1200, 3000, 100000, .95, .15, 9.58)]
@pytest.mark.parametrize('P, T, Qo, Qw, Qg, Z, D, E', velocity_validation)
def test_mix_velocity(P, T, Qo, Qw, Qg, Z, D, E):
    assert mix_velocity(P, T, Qo, Qw, Qg, Z, D) == E


density_validation = [(30, 60, 300, 4000, 150000, 700, 1000, 19, .95, 439.69)]
@pytest.mark.parametrize('P, T, Qo, Qw, Qg, rho_o, rho_w, MW, Z, E', density_validation)
def test_mix_density(P, T, Qo, Qw, Qg, rho_o, rho_w, MW, Z, E):
    assert mix_density(P, T, Qo, Qw, Qg, rho_o, rho_w, MW, Z) == E


viscosity_validation = [(40, 80, 4000, 300, 400000, 2, 1, 1e-3, .9, 0.538166)]
@pytest.mark.parametrize('P, T, Qo, Qw, Qg, mu_o, mu_w, mu_g, Z, E', viscosity_validation)
def test_mix_viscosity(P, T, Qo, Qw, Qg, mu_o, mu_w, mu_g, Z, E):
    assert mix_viscosity(P, T, Qo, Qw, Qg, mu_o, mu_w, mu_g, Z) == E