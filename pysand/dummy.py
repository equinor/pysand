def nozzlevalve_wall(v_m, d_p, GF, At, material='duplex'):
    """
    Particle valve wall erosion for non-slam nozzle type check-valve. Based on DNVGL CFD-study of Johan Sverdrup Phase 1 check-valves (13.01.2020)
    Report No: 2019-1237 Rev.1, Document No: 547341
    :param v_m: Internal valve mix velocity [m/s]. Use mixture velocity through minimum flow area of the valve.
    :param GF: Geometry factor [-]
    :param A_t: Target area [m²]. Set to minimum flow area of the valve. 
    :param d_p: Particle diameter [mm]
    :param material: Material exposed to erosion, default = 'duplex'. For others, run: materials()
    :return: Relative erosion rate [mm/ton]
    """
    # Constants:
    def c1(d_p):
        corr = 8.33 * d_p**3 - 29.2 * d_p**2 + 22.8 * d_p + 1
        return corr
    C1 = c1(d_p) # Model geometry factor

    rho_t = 7850
    K = 2e-9
    n = 2.6

    E_rel = K * v_m ** n / (2 * rho_t * At) * C1 * GF * 10 ** 6 # Relative surface thickness loss [mm/t]

    return E_rel

v_m = 10
d_p = 0.1
GF = 1
At = 0.03

E = nozzlevalve_wall(v_m, d_p, GF, At, material='duplex')

print(E)