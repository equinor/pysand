import pytest
from pysand.choke import critical_velocity, min_choke

critical_v_validation = [(2, 1, 200, 100, pytest.approx(36.51, abs=10e-2)),
                         (40, 20, 700, 600, pytest.approx(78.45, abs=10e-2))]


@pytest.mark.parametrize('p1, p2, rho_1, rho_2, v_c', critical_v_validation)
def test_critical_velocity(p1, p2, rho_1, rho_2, v_c):
    assert critical_velocity(p1, p2, rho_1, rho_2) == v_c


min_choke_validation = [(0, 20, 0.05),
                        (0.005, 75, 0.05),
                        (0.008, 150, 0.1),
                        (0.009, 250, 0.1),
                        (0.01, 15, 0.05),
                        (0.02, 90, 0.1),
                        (0.05, 110, 0.2),
                        (0.09, 400, 0.2),
                        (0.2, 6, 0.1),
                        (0.4, 75, 0.2),
                        (1.2, 105, 0.2),
                        (4.6, 205, 0.2)]


@pytest.mark.parametrize('Qs, v_c, cvr_min', min_choke_validation)
def test_min_choke(Qs, v_c, cvr_min):
    assert min_choke(Qs, v_c) == cvr_min
