import pytest
import numpy as np
import pandas as pd
import pysand.exceptions as exc
import logging
from pysand.erosion import validate_inputs, bend, tee, straight_pipe, \
    welded_joint, manifold, reducer, probes, flexible, choke_gallery, F, material_properties

def test_validate_inputs(caplog):

    # Testing input throws exception throws exception
    kwargs = {'v_m': 29.3, 'rho_m': 30, 'mu_m': 1.5e-5, 'Q_s': 2400*1000/86400/365}
    for inp in ['v_m', 'rho_m', 'mu_m', 'Q_s']:
        for non_number in [None, 'string', np.nan]:
            kwargs[inp] = non_number
            with pytest.raises(exc.FunctionInputFail) as excinfo:
                validate_inputs(**kwargs)

    # Test v_m boundaries
    kwargs = {'v_m': 201}
    with caplog.at_level(logging.WARNING):
        validate_inputs(**kwargs)
    assert "Mix velocity, v_m, is outside RP-O501 model boundaries (0-200 m/s)." in str(caplog.records)

    # Test rho_m boundaries
    kwargs = {'rho_m': 1}
    for illegal_input in [0.1, 1555]:
        kwargs['rho_m'] = illegal_input
        with caplog.at_level(logging.WARNING):
            validate_inputs(**kwargs)
            info = [record for record in caplog.records if record.levelname == 'WARNING']
            assert any(
                "Mix density, rho_m, is outside RP-O501 model boundaries (1-1500 kg/m3)."
                in s.message for s in info)

    # Test mu_m boundaries
    kwargs = {'mu_m': 1e-3}
    for illegal_input in [1e-7, 1]:
         kwargs['mu_m'] = illegal_input
         with caplog.at_level(logging.WARNING):
             validate_inputs(**kwargs)
             info = [record for record in caplog.records if record.levelname == 'WARNING']
             assert any(
                 "Mix viscosity, mu_m, is outside RP-O501 model boundaries (1e-6 - 1e-2 kg/ms)."
                 in s.message for s in info)

    # Test Q_s boundaries
    kwargs = {'Q_s': 1, 'v_m': 1, 'D': .1}
    for illegal_input in [100, 555]:
        kwargs['Q_s'] = illegal_input
        with caplog.at_level(logging.WARNING):
            validate_inputs(**kwargs)
            info = [record for record in caplog.records if record.levelname == 'WARNING']
            assert any(
                "The particle concentration is outside RP-O501 model boundaries ( 0-500 ppmV)."
                in s.message for s in info)

    kwargs = {'R': 29.3, 'GF': 30, 'D': 1.5e-5, 'd_p': 1, 'h': 30, 'Dm': 30, 'D1': 30, 'D2': 30}
    for inp in ['R', 'GF', 'D', 'd_p', 'h', 'Dm', 'D1', 'D2']:
        for non_number in [None, 'string', np.nan, pd.Series().any()]:
            kwargs[inp] = non_number
            with pytest.raises(exc.FunctionInputFail) as excinfo:
                validate_inputs(**kwargs)

    # Test pipe inner diameter boundaries
    kwargs = {'D': 1, 'D1': 1, 'D2': 1}
    for inp in ['D', 'D1', 'D2']:
        for illegal_input in [0.005, 5]:
            kwargs[inp] = illegal_input
            with caplog.at_level(logging.WARNING):
                validate_inputs(**kwargs)
                info = [record for record in caplog.records if record.levelname == 'WARNING']
                assert any(
                    "Pipe inner diameter, {}, is outside RP-O501 model boundaries (0.01 - 1 m).".format(inp)
                    in s.message for s in info)

    # Test particle diameter boundaries
    kwargs = {'d_p': 1}
    for illegal_input in [0.01, 6]:
        kwargs['d_p'] = illegal_input
        with caplog.at_level(logging.WARNING):
            validate_inputs(**kwargs)
            info = [record for record in caplog.records if record.levelname == 'WARNING']
            assert any(
                "Particle diameter, d_p, is outside RP-O501 model boundaries (0.02 - 5 mm)." in s.message for s in info)

    # Test geometry factor limitations
    kwargs = {'GF': 6}
    with caplog.at_level(logging.WARNING):
        validate_inputs(**kwargs)
    assert "Geometry factor, GF, can only be 1, 2, 3 or 4" in str(caplog.records)

    # Test alpha boundaries
    kwargs = {'alpha': 1}
    for illegal_input in [-1, 91]:
        kwargs['alpha'] = illegal_input
        with caplog.at_level(logging.WARNING):
            validate_inputs(**kwargs)
            info = [record for record in caplog.records if record.levelname == 'WARNING']
            assert any(
                "Particle impact angle [degrees], alpha, is outside RP-O501 model boundaries (10-80 deg)."
                in s.message for s in info)

    # Test bend radius boundaries
    kwargs = {'R': 1}
    for illegal_input in [0.1, 51]:
        kwargs['R'] = illegal_input
        with caplog.at_level(logging.WARNING):
            validate_inputs(**kwargs)
            info = [record for record in caplog.records if record.levelname == 'WARNING']
            assert any(
                "Bend radius, R, is outside RP-O501 model boundaries." in s.message for s in info)

    # Test manifold diameter against branch pipe diameter
    kwargs = {'Dm': 0.5, 'D': 0.8}
    with caplog.at_level(logging.WARNING):
        validate_inputs(**kwargs)
    assert "Manifold diameter, Dm, is expected to be bigger than branch pipe diameter, D" in str(caplog.records)

    # Test height of the weld boundaries
    kwargs = {'h': 0.01, 'D': 0.8}
    for illegal_input in [-1, 1]:
        kwargs['h'] = illegal_input
        with caplog.at_level(logging.WARNING):
            validate_inputs(**kwargs)
            info = [record for record in caplog.records if record.levelname == 'WARNING']
            assert any(
                "Height of the weld, h, must positive number not exceeding pipe inner diameter size, D"
                in s.message for s in info)

    # Test choke gallery input
    R_c = 0.15
    gap = 0.04
    H = 0.15

    kwargs = {'R_c': R_c, 'gap': gap, 'H': H}
    for i in kwargs.keys():
        kwargs[i] = -1
        with pytest.raises(exc.FunctionInputFail) as excinfo:
            validate_inputs(**kwargs)

        kwargs['R_c'] = R_c
        kwargs['gap'] = gap
        kwargs['H'] = H

    kwargs['gap'] = 0.2
    with pytest.raises(exc.FunctionInputFail):
        validate_inputs(**kwargs)

# Pipe bends #
# Bend validation 1 based on the model validations in DNVGL RP-O501, Aug 2015
# Bend validation 2 to test all versions of gamma
# Bend validation 3 based on example in DNVGL RP-O501, Aug 2015, chapter 4.7
bend_validation = [(29.3, 30, 1.5e-5, 2400*1000/86400/365, 1.75, 1, 0.0978, 0.28, 'duplex', pytest.approx(0.6128002)),
                   (15, 2, 4e-4, .1, 1.5, 2, .1, .4, 'duplex', pytest.approx(0.0115661)),
                   (15, 333.3, 3.4e-4, 1e5/365/86400, 1.5, 1, 0.1, 0.25, 'duplex', pytest.approx(1.433187e-3))]
@pytest.mark.parametrize('v_m, rho_m, mu_m, Q_s, R, GF, D, d_p, material, E', bend_validation)
def test_bend(v_m, rho_m, mu_m, Q_s, R, GF, D, d_p, material, E):
    assert bend(v_m, rho_m, mu_m, Q_s, R, GF, D, d_p, material=material) == E  # mm/ton


# Blinded tees #
tee_validation = [(30, 400, 1e-3, .1, 2, 0.1, 0.3, 'duplex', pytest.approx(0.1704876))]
@pytest.mark.parametrize('v_m, rho_m, mu_m, Q_s, GF, D, d_p, material, E', tee_validation)
def test_tee(v_m, rho_m, mu_m, Q_s, GF, D, d_p, material, E):
    assert tee(v_m, rho_m, mu_m, Q_s, GF, D, d_p, material=material) == E


# Smooth and straight pipes #
pipe_validation = [(15, 4, 0.1, pytest.approx(0.0114245))]
@pytest.mark.parametrize('v_m, Q_s, D, E', pipe_validation)
def test_pipe(v_m, Q_s, D, E):
    assert straight_pipe(v_m, Q_s, D) == E


# Welded joints #
weld_validation = [(15, 150, 4, 0.1, 0.3, 0.023, 'duplex', (pytest.approx(2.72521), pytest.approx(0.358158))),
                   (30, 300, 4, 0.1, 0.8, 0.023, 'duplex', (pytest.approx(20.23594), pytest.approx(2.1714649)))]
@pytest.mark.parametrize('v_m, rho_m, Q_s, D, d_p, h, material, E', weld_validation)
def test_weld(v_m, rho_m, Q_s, D, d_p, h, material, E):
    assert welded_joint(v_m, rho_m, Q_s, D, d_p, h, material=material) == E


# Manifolds #
manifold_validation = [(29.3, 30, 1.5e-5, 2400*1000/86400/365, 1, 0.0978, 0.28, 0.2, 'duplex',pytest.approx(0.6476766)),
                       (30, 1.2, 1.5e-5, 9700*1000/86400/365, 1, 0.128, 0.25, 0.2, 'duplex', pytest.approx(1.8785739))]
@pytest.mark.parametrize('v_m, rho_m, mu_m, Q_s, GF, D, d_p, Dm, material, E', manifold_validation)
def test_manifold(v_m, rho_m, mu_m, Q_s, GF, D, d_p, Dm, material, E):
    assert manifold(v_m, rho_m, mu_m, Q_s, GF, D, d_p, Dm, material=material) == E


# Reducers #
reducer_validation = [(20, 80, 1, 0.15, 0.1, 0.3, 1, 50, 'duplex', pytest.approx(6.3947669)),
                      (20, 120, 1, 0.15, 0.1, 0.3, 1, 50, 'duplex', pytest.approx(5.8375968))]
@pytest.mark.parametrize('v_m, rho_m, Q_s, D1, D2, d_p, GF, alpha, material, E', reducer_validation)
def test_reducer(v_m, rho_m, Q_s, D1, D2, d_p, GF, alpha, material, E):
    assert reducer(v_m, rho_m, Q_s, D1, D2, d_p, GF=GF, alpha=alpha, material=material) == E


# Erosion probes #
probe_validation = [(30, 80, 1, 0.15, 0.3, 50, 'duplex', pytest.approx(2.22837)),
                      (20, 120, 0.1, 0.15, 0.3, 30, 'duplex', pytest.approx(0.0494802))]
@pytest.mark.parametrize('v_m, rho_m, Q_s, D, d_p, alpha, material, E', probe_validation)
def test_probes(v_m, rho_m, Q_s, D, d_p, alpha, material, E):
    assert probes(v_m, rho_m, Q_s, D, d_p, alpha=alpha, material=material) == E

# Flexible pipes with interlock carcass #
flexible_validation = [(23, 350, 1e-4, .3, 15, .124, .2, 'duplex',pytest.approx(0.0871711))]
@pytest.mark.parametrize('v_m, rho_m, mu_m, Q_s, mbr, D, d_p, material, E', flexible_validation)
def test_flexible(v_m, rho_m, mu_m, Q_s, mbr, D, d_p, material, E):
    assert flexible(v_m, rho_m, mu_m, Q_s, mbr, D, d_p, material=material) == E

# Choke gallery #
gallery_validation = [(30, 450, 5e-4, .7, 1, .15, .5, .15, .04, .15, 'duplex', pytest.approx(15.48484)),
                      (30, 450, 5e-4, .7, 1, .15, .5, .15, .04, .15, 'dc_05_tungsten', pytest.approx(.0380211))]
@pytest.mark.parametrize('v_m, rho_m, mu_m, Q_s, GF, D, d_p, R_c, gap, H, material, E', gallery_validation)
def test_choke_gallery(v_m, rho_m, mu_m, Q_s, GF, D, d_p, R_c, gap, H, material, E):
    assert choke_gallery(v_m, rho_m, mu_m, Q_s, GF, D, d_p, R_c, gap, H, material=material) == E

# Angle dependency function #
F_validation = [(0, 'ductile', 0),
                (np.deg2rad(34.8), 'ductile', pytest.approx(1, abs=.1)),
                (np.pi/2, 'ductile', pytest.approx(.6)),
                (0, 'brittle', 0),
                (np.pi / 4, 'brittle', .5),
                (np.pi / 2, 'brittle', 1)]
@pytest.mark.parametrize('a_rad, angle_dependency, E', F_validation)
def test_F(a_rad, angle_dependency, E):
    assert F(a_rad, angle_dependency) == E

    angle_dependency = 'something_else'
    with pytest.raises(exc.FunctionInputFail) as excinfo:
        F(a_rad, angle_dependency)


# Material properties function #
material_validation = [('carbon_steel', (7800, 2e-9, 2.6, 'ductile')),
                       ('grp_epoxy', (1800, 3e-10, 3.6, 'ductile')),
                       ('psz_ceramic_zirconia', (5700, 4.1e-9, 2.5, 'brittle'))]
@pytest.mark.parametrize('material, E', material_validation)
def test_material_properties(material, E):
    assert material_properties(material) == E

# Test material list and error raising
def test_material():
    material_list = ['carbon_steel', 'duplex', 'ss316', 'inconel', 'grp_epoxy', 'grp_vinyl_ester', 'hdpe', 'aluminium',
                     'dc_05_tungsten', 'cs_10_tungsten', 'cr_37_tungsten', '95_alu_oxide', '99_alu_oxide',
                     'psz_ceramic_zirconia', 'ZrO2-Y3_ceramic_zirconia', 'SiC_silicon_carbide', 'Si3N4_silicon_nitride',
                     'TiB2_titanium_diboride', 'B4C_boron_carbide', 'SiSiC_ceramic_carbide']
    material = 'list'
    assert material_properties(material) == material_list

    material = 'something_else'
    with pytest.raises(exc.FunctionInputFail) as excinfo:
        material_properties(material)

def test_return_nan():
    v_m = 29.3
    rho_m = 30
    mu_m = 1.5e-5
    Q_s = 2400 * 1000 / 86400 / 365
    R = 1
    GF = 2
    D = .1
    d_p = .4
    h = 0.023
    Dm = .2
    D1 = .15
    D2 = .1
    mbr = 15
    R_c = .15
    gap = .04
    H = .15
    # test bend
    kwargs = {'v_m': v_m, 'rho_m': rho_m, 'mu_m': mu_m, 'Q_s': Q_s, 'R': R, 'GF': GF, 'D': D, 'd_p': d_p}
    for inp in ['v_m', 'rho_m', 'mu_m', 'Q_s']:
        kwargs[inp] = -1
        assert np.isnan(bend(**kwargs))
    
    # test tee
    kwargs = {'v_m': v_m, 'rho_m': rho_m, 'mu_m': mu_m, 'Q_s': Q_s, 'GF': GF, 'D': D, 'd_p': d_p}
    for inp in ['v_m', 'rho_m', 'mu_m', 'Q_s']:
        kwargs[inp] = -1
        assert np.isnan(tee(**kwargs))

    # straight pipe
    kwargs = {'v_m': v_m, 'Q_s': Q_s, 'D': D}
    for inp in ['v_m', 'Q_s']:
        kwargs[inp] = -1
        assert np.isnan(straight_pipe(**kwargs))

    # test welded joint
    kwargs = {'v_m': v_m, 'rho_m': rho_m, 'Q_s': Q_s, 'D': D, 'd_p': d_p, 'h': h}
    for inp in ['v_m', 'rho_m', 'Q_s']:
        kwargs[inp] = -1
        assert np.isnan(welded_joint(**kwargs))

    # test manifold
    kwargs = {'v_m': v_m, 'rho_m': rho_m, 'mu_m': mu_m, 'Q_s': Q_s, 'GF': GF, 'D': D, 'd_p': d_p, 'Dm': Dm}
    for inp in ['v_m', 'rho_m', 'mu_m', 'Q_s']:
        kwargs[inp] = -1
        assert np.isnan(manifold(**kwargs))

    # test reducer
    kwargs = {'v_m': v_m, 'rho_m': rho_m, 'Q_s': Q_s, 'D1': D1, 'D2': D2, 'd_p': d_p}
    for inp in ['v_m', 'rho_m', 'Q_s']:
        kwargs[inp] = -1
        assert np.isnan(reducer(**kwargs))

    # test probes
    # v_m, rho_m, Q_s, D, d_p
    kwargs = {'v_m': v_m, 'rho_m': rho_m, 'Q_s': Q_s, 'D': D, 'd_p': d_p}
    for inp in ['v_m', 'rho_m', 'Q_s']:
        kwargs[inp] = -1
        assert np.isnan(probes(**kwargs))

    # test flexible
    kwargs = {'v_m': v_m, 'rho_m': rho_m, 'mu_m': mu_m, 'Q_s': Q_s, 'mbr': mbr, 'D': D, 'd_p': d_p}
    for inp in ['v_m', 'rho_m', 'mu_m', 'Q_s']:
        kwargs[inp] = -1
        assert np.isnan(flexible(**kwargs))

    # test choke gallery
    # v_m, rho_m, mu_m, Q_s, GF, D, d_p, R_c, gap, H
    kwargs = {'v_m': v_m, 'rho_m': rho_m, 'mu_m': mu_m, 'Q_s': Q_s, 'GF': GF, 'D': D,
              'd_p': d_p, 'R_c': R_c, 'gap': gap, 'H': H}
    for inp in ['v_m', 'rho_m', 'mu_m', 'Q_s']:
        kwargs[inp] = -1
        assert np.isnan(choke_gallery(**kwargs))
