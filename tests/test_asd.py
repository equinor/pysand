import pytest
from pysand.asd import std_step_clampon, std_step_emerson, sand_rate

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


sand_rate_validation = [(5000, 5000, 2000, 0),
                        (4000, 2000, 1000, 2),
                        (2000, 4000, 500, 0),
                        (2000, 1000, 0, None),
                        (2000, 1000, -5000, None)]
@pytest.mark.parametrize('raw, zero, step, E', sand_rate_validation)
def test_sand_rate(raw, zero, step, E):
    assert sand_rate(raw, zero, step) == E