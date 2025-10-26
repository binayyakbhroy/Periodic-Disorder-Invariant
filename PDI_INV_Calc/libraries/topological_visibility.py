import numpy as np
import kwant
from config import Config, defaults
from scipy.constants import hbar, m_e, eV, physical_constants
mu_B = physical_constants["Bohr magneton"][0] / (eV*1e3)  # Bohr magneton in meV

cfg = Config() or defaults()

s0 = np.eye(2)
sx = np.array([[0., 1.0], [1.0, 0.]])
sy = np.array([[0., -1.j], [1.j, 0.]])
sz = np.array([[1., 0.], [0., -1.]])

def make_sys(args):
    """
    Build and finalize the Kwant system with given parameters.
    """

    # Example constants
    N = cfg.Lattice_sites
    U = cfg.tunnel_barrier_amplitude
    t = cfg.hopping_amplitude
    barrier_sites = cfg.tunnel_barrier_width
    alpha_mevA = cfg.rashba_soc
    delta_sc = cfg.smsc_coupling
    a_lattice = cfg.lattice_const
    g = cfg.g_factor

    kron_s0_sz = np.kron(s0, sz)
    kron_sz_sx = np.kron(sz, sx)
    kron_sx_s0 = np.kron(sx, s0)
    kron_s0_s0 = np.kron(s0, s0)
    kron_s0_sx = np.kron(s0, sx)
    kron_sy_sz = np.kron(sy, sz)
    kron_sy_sy = np.kron(sy, sy)
    kron_sz_s0 = np.kron(sz, s0)
    kron_sz_sy = np.kron(sz, sy)

    lat = kwant.lattice.chain(a_lattice, norbs=4)

    sys = kwant.Builder(particle_hole=kron_sy_sy)
    #sys = kwant.Builder(conservation_law=-kron_s0_sz, particle_hole=kron_sy_sy)
    #sys = kwant.Builder()

    def potential(site, Vimp):
        i = site.pos[0] 
        return Vimp[int(i // a_lattice)]

    def onsite_normal(site, mu_l, B):
        return (2 * t - mu_l) * kron_s0_sz + (0.5 * B * g * mu_B) * kron_sx_s0

    def onsite_sc(site, mu, B, Vimp):
        return (
            + (2 * t - mu) * kron_s0_sz
            + (0.5 * B * g * mu_B) * kron_sx_s0
            - delta_sc * kron_s0_sx
            + potential(site, Vimp) * kron_s0_sz
        )

    def onsite_barrier(site, mu_l, B, Vimp):
        return (
            (2 * t - mu_l) * kron_s0_sz
            + (0.5 * B * g * mu_B) * kron_sx_s0
            + U * kron_s0_sz
            + potential(site, Vimp) * kron_s0_sz
        )

    def hop(site1, site2):
        return -t * kron_s0_sz + 0.5j * alpha_mevA / a_lattice * kron_sy_sz

    sys[(lat(i) for i in range(N))] = onsite_sc
    sys[lat(0)] = onsite_normal
    sys[(lat(i+1) for i in range(barrier_sites))] = onsite_barrier
    sys[(lat(N - i - 1) for i in range(barrier_sites))] = onsite_barrier
    sys[lat(N)] = onsite_normal
    sys[lat.neighbors()] = hop

    symmetry = kwant.TranslationalSymmetry(*-lat.prim_vecs)
    lead_normal = kwant.Builder(
        symmetry, conservation_law=-np.kron(s0, sz), particle_hole=np.kron(sy, sy)
    )
    lead_normal[lat(0)] = onsite_normal
    lead_normal[lat(1), lat(0)] = hop

    sys.attach_lead(lead_normal)
    sys.attach_lead(lead_normal.reversed())
    sys = sys.finalized()

    return sys

def TV_calc(params):
    """
    Calculate the topological visibility of the system at given energies.
    
    Parameters
    ----------
    params : dict containing 
        mu : float
            Chemical potential.
        B : float
            Magnetic field.
        Vimp : ndarray
            Onsite disorder values.
        mu_l : float
            Lead chemical potential.

    Returns
    -------
    float, float
        Topological visibility values corresponding to the input energies.
    """
    sys = make_sys(params)
    smatrix = kwant.smatrix(
        sys,
        energy=0.0,
        check_hermiticity=False,
        params=params,
    )

    R_LL = smatrix.submatrix(0, 0)
    R_RR = smatrix.submatrix(1, 1)
    TVL, TVR = np.linalg.det(R_LL), np.linalg.det(R_RR)

    if (np.absolute(np.imag(TVL)) + np.absolute(np.imag(TVR))) > 1e-1:
        print("PHS Error:", (np.absolute(np.imag(TVL)) + np.absolute(np.imag(TVR))))

    if (np.absolute(TVL - TVR)) > 1e-1:
        print("Topological visibility mismatch Error:", TVL, TVR)
    print(TVL, TVR)
    return np.real(TVL), np.real(TVR)