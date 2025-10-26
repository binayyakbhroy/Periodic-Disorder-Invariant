import numpy as np
from config import Config, defaults
from disorder_profiles import disorder
from scipy import sparse

cfg = Config() or defaults()
Length = cfg.Lattice_sites
# Model parameters from config
t = cfg.hopping_amplitude
alpha = cfg.rashba_soc / cfg.lattice_const
coupling = cfg.smsc_coupling
V = cfg.Disorder_amplitude * disorder()
S = np.array([[0, 0, 1, 0],[0, 0, 0, 1],[1, 0, 0, 0],[0, 1, 0, 0]]) #chiral symmetry operator

def onsite_h(i, H, mu, Vz):
    ##Onsite Chemical and Disorder Potential terms
    H[i][i] =  (2*t - mu) + V[int(i/4)]
    H[i + 1][i + 1] = (2*t - mu) + V[int(i/4)]
    H[i + 2][i + 2] = -H[i][i]
    H[i + 3][i + 3] = -H[i][i]
    ##Zeeman term
    H[i][i+1] = H[i+1][i] = Vz
    H[i+2][i+3] = H[i+3][i+2] = -Vz
    return H

def H_delta(i, H, Vz, gamma_surface, variable_coupling=False):
    Z = 1.0
    delta = cfg.smsc_coupling
    if variable_coupling:
        delta_0 = lambda Zeeman: cfg.sc_order_param - 0.17173*(Zeeman**2.5)
        delta = delta_0(Vz)*gamma_surface/(delta_0(Vz)+gamma_surface)
        Z = delta_0(Vz)/(delta_0(Vz)+gamma_surface)
    ##Superconducting parameter
    H[i][i+3] = H[i+3][i] = delta*(1/Z)
    H[i+1][i+2] = H[i+2][i+1] = -delta*(1/Z)
    if i == 4*(Length-1):
        H = Z*H
    return H

def hopping_h(i, H, phi, calc_pfaffian=False, include_rashba_pf=False):
    
    if (i < 4 * (Length - 1)):
        ##Hopping term
        H[i][i + 4] = H[i + 4][i] = -t
        H[i + 1][i + 5] = H[i + 5][i + 1] = -t
        H[i + 2][i + 6] = H[i + 6][i + 2] = t
        H[i + 3][i + 7] = H[i + 7][i + 3] = t
        ##Rashba Spin orbit coupling term
        H[i][i + 5] = H[i + 5][i] = alpha/2.0 
        H[i + 1][i + 4] = H[i + 4][i + 1] = -alpha/2.0
        H[i + 2][i + 7] = H[i + 7][i + 2] = -alpha/2.0
        H[i + 3][i + 6] = H[i + 6][i + 3] = alpha/2.0

    if calc_pfaffian:
        if (i == 4*(Length - 1)):
            ##Phased Hopping term from N-1th site to 0th site
            H[i][0] = -t*np.exp(-1.0j*phi)
            H[i+1][1] = -t*np.exp(-1.0j*phi)
            H[i+2][2] = t*np.exp(1.0j*phi)
            H[i+3][3] = t*np.exp(1.0j*phi)
            ##Phased Hopping term from 0th site to N-1th site
            H[0][i] = -t*np.exp(1.0j*phi)
            H[1][i+1] = -t*np.exp(1.0j*phi)
            H[2][i+2] = t*np.exp(-1.0j*phi)
            H[3][i+3] = t*np.exp(-1.0j*phi)

            #Options to include phased Rashba term in Pfaffian calculation
            if include_rashba_pf:
                ##Phased Rashba Spin orbit coupling term from N-1th site to 0th site
                H[i][1] = (alpha/2.0)*np.exp(-1.0j*phi)
                H[i+1][0] = -(alpha/2.0)*np.exp(-1.0j*phi)
                H[i+2][3] = -(alpha/2.0)*np.exp(1.0j*phi)
                H[i+3][2] = (alpha/2.0)*np.exp(1.0j*phi)
                ##Phased Rashba Spin orbit coupling term from 0th site to N-1th site
                H[1][i] = (alpha/2.0)*np.exp(1.0j*phi)
                H[0][i+1] = -(alpha/2.0)*np.exp(1.0j*phi)
                H[3][i+2] = -(alpha/2.0)*np.exp(-1.0j*phi)
                H[2][i+3] = (alpha/2.0)*np.exp(-1.0j*phi)
        
    return H

def construct_Hamil(mu, Vz, phi, calc_pfaffian=False):
    H_0 = np.zeros((4* (Length), 4 * (Length)), dtype=complex)
    Hamiltonian = []
    for i in range(0, 4 * (Length), 4):
        H_0 = onsite_h(i, H_0, mu, Vz)
        H_0 = hopping_h(i, H_0, phi, calc_pfaffian)
        H_0 = H_delta(i, H_0, Vz, coupling) ## Call superconducting term 
        #last since the renormalization is happening in this function
    Hamiltonian = np.array(H_0)
    if calc_pfaffian:
        Hamiltonian = np.real(np.matmul(H_0,np.kron(np.eye(Length),S)))
        Hamiltonian = np.ascontiguousarray(Hamiltonian, dtype=np.float64)
    return Hamiltonian

def generate_energy_spectrum(mu, zeeman_low=cfg.zeeman_low, zeeman_high=cfg.zeeman_high, phi_0=0.0, num_eigvals=30):
    """Calculate the energy spectrum of the Hamiltonian H for a given chemical potential."""
    Zeeman_range = np.linspace(zeeman_low, zeeman_high, cfg.zeeman_count)
    energy_spectrum = []
    for Z0 in Zeeman_range:
        H = construct_Hamil(mu, Z0, phi=phi_0, calc_pfaffian=False)
        eigenvalues, _ = sparse.linalg.eigsh(sparse.csr_matrix(H), k=num_eigvals, sigma=0, which='LM')
        eigenvalues = eigenvalues[np.argsort(eigenvalues)]
        energy_spectrum.append(eigenvalues)
    energy_spectrum = np.array(energy_spectrum)
    return energy_spectrum

def generate_wavefunctions(mu, zeeman, phi_0=0.0, num_eigvals=10, return_qmzm=False):
    """Calculate the wavefunctions of the Hamiltonian H for a given chemical potential and zeeman energy."""
    H = construct_Hamil(mu, zeeman, phi=phi_0, calc_pfaffian=False)
    eigenvalues, eigenvectors = sparse.linalg.eigsh(sparse.csr_matrix(H), k=num_eigvals, sigma=0, which='LM')
    ordering = np.argsort(np.abs(eigenvalues))
    sorted_eigenvectors = eigenvectors[:, ordering]
    wf1 = sorted_eigenvectors[:, 0] + np.conj(sorted_eigenvectors[:, 1])
    wf2 = -1.0j*(sorted_eigenvectors[:, 0] - np.conj(sorted_eigenvectors[:, 1]))
    wf3 = sorted_eigenvectors[:, 2] + np.conj(sorted_eigenvectors[:, 3])
    wf4 = -1.0j*(sorted_eigenvectors[:, 2] - np.conj(sorted_eigenvectors[:, 3]))
    wf1, wf2, wf3, wf4 = np.abs(wf1)**2, np.abs(wf2)**2, np.abs(wf3)**2, np.abs(wf4)**2
    wf1, wf2, wf3, wf4 = np.sum(wf1.reshape(Length,4), axis=1), np.sum(wf2.reshape(Length,4), axis=1), np.sum(wf3.reshape(Length,4), axis=1), np.sum(wf4.reshape(Length,4), axis=1)
    if return_qmzm:
        return wf1, wf2, wf3, wf4
    return wf1, wf2
    
