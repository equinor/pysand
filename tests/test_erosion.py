import pytest
from pysand.erosion import bend, tee, straight_pipe, welded_joint, manifold, reducer, probes


# Pipe bends #
# Bend validation 1 based on the model validations in DNVGL RP-O501, Aug 2015
# Bend validation 2 to test all versions of gamma
# Bend validation 3 based on example in DNVGL RP-O501, Aug 2015, chapter 4.7
bend_validation = [(29.3, 30, 1.5e-5, 2400*1000/86400/365, 1.75, 1, 0.0978, 0.28, pytest.approx(0.6128002)),
                   (15, 2, 4e-4, .1, 1.5, 2, .1, .4, pytest.approx(0.0115661)),
                   (15, 333.3, 3.4e-4, 1e5/365/86400, 1.5, 1, 0.1, 0.25, pytest.approx(1.433187e-3))]
@pytest.mark.parametrize('v_m, rho_m, mu_m, Q_s, R, GF, D, d_p, E', bend_validation)
def test_bend(v_m, rho_m, mu_m, Q_s, R, GF, D, d_p, E):
    assert bend(v_m, rho_m, mu_m, Q_s, R, GF, D, d_p) == E  # mm/ton


# Blinded tees #
tee_validation = [(15, 513.3, 4.02e-4, 2, 2, 0.15, 0.25, pytest.approx(107.988))]
@pytest.mark.parametrize('v_m, rho_m, mu_m, Q_s, GF, D, d_p, E', tee_validation)
def test_tee(v_m, rho_m, mu_m, Q_s, GF, D, d_p, E):
    assert tee(v_m, rho_m, mu_m, Q_s, GF, D, d_p) == E


# Smooth and straight pipes #
pipe_validation = [(15, 4, 0.1, pytest.approx(0.0114245))]
@pytest.mark.parametrize('v_m, Q_s, D, E', pipe_validation)
def test_pipe(v_m, Q_s, D, E):
    assert straight_pipe(v_m, Q_s, D) == E


# Welded joints #
weld_validation = [(15, 150, 4, 0.1, 0.3, 0.023, (pytest.approx(2.72521), pytest.approx(0.358158))),
                   (30, 300, 4, 0.1, 0.8, 0.023, (pytest.approx(20.23594), pytest.approx(2.1714649)))]
@pytest.mark.parametrize('v_m, rho_m, Q_s, D, d_p, h, E', weld_validation)
def test_weld(v_m, rho_m, Q_s, D, d_p, h, E):
    assert welded_joint(v_m, rho_m, Q_s, D, d_p, h) == E


# Manifolds #
manifold_validation = [(29.3, 30, 1.5e-5, 2400*1000/86400/365, 1, 0.0978, 0.28, 0.2, pytest.approx(0.6476766)),
                       (30, 1.2, 1.5e-5, 9700*1000/86400/365, 1, 0.128, 0.25, 0.2, pytest.approx(1.8785739))]
@pytest.mark.parametrize('v_m, rho_m, mu_m, Q_s, GF, D, d_p, Dm, E', manifold_validation)
def test_manifold(v_m, rho_m, mu_m, Q_s, GF, D, d_p, Dm, E):
    assert manifold(v_m, rho_m, mu_m, Q_s, GF, D, d_p, Dm) == E


# Reducers #
reducer_validation = [(20, 80, 1, 0.15, 0.1, 0.3, 1, 50, pytest.approx(6.3947669)),
                      (20, 120, 1, 0.15, 0.1, 0.3, 1, 50, pytest.approx(5.8375968))]
@pytest.mark.parametrize('v_m, rho_m, Q_s, D1, D2, d_p, GF, alpha, E', reducer_validation)
def test_reducer(v_m, rho_m, Q_s, D1, D2, d_p, GF, alpha, E):
    assert reducer(v_m, rho_m, Q_s, D1, D2, d_p, GF, alpha) == E


# Erosion probes #
probe_validation = [(30, 80, 1, 0.15, 0.3, 50, pytest.approx(2.22837)),
                      (20, 120, 0.1, 0.15, 0.3, 30, pytest.approx(0.0494802))]
@pytest.mark.parametrize('v_m, rho_m, Q_s, D, d_p, alpha, E', probe_validation)
def test_probes(v_m, rho_m, Q_s, D, d_p, alpha, E):
    assert probes(v_m, rho_m, Q_s, D, d_p, alpha) == E