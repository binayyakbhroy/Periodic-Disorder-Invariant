import numpy as np
from config import Config, defaults

cfg = Config() or defaults()
Length = cfg.Lattice_sites

# Pauli matrices for convenience
s0 = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])

zeros_4 = np.zeros((4, 4))
zeros_8 = np.zeros((8, 8))

chirality_op = np.kron(sx, np.eye(2))
szs0 = np.kron(sz, s0)
szsx = np.kron(sz, sx)
sysy = np.kron(sy, sy)
szsy = np.kron(sz, sy)


def inv(m):
    i = np.eye(m.shape[0])
    try:
        return np.linalg.solve(m,i)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(m,rcond=1e-15)

#Ignore the temp variables/matrices

def fV(q, h_onsite, h_hopping, translation, translation_pr, sigma_right, disorder, N=Length):

    # ---------- Disorder potential ----------
    V_x = disorder
    V_x = V_x[:N]

    # ---------- Precompute constants ----------
    phase_pos = np.exp(1.0j*q*N)
    phase_neg = np.exp(-1.0j*q*N)

    g_block11 = lambda i: h_onsite + V_x[i]*szs0
    g_block12 = h_hopping.T*phase_neg
    g_block21 = h_hopping*phase_pos

    # ---------- Compute Sigma_L recursively ----------

    sigma_left = [zeros_8]
    green_g0 = np.block([[g_block11(0), g_block12],[g_block21, g_block11(-1)]])
    green_gi = lambda i: np.block([[g_block11(i), zeros_4],[zeros_4, g_block11(-i-1)]])
    green_inv = inv((-1.0*(green_g0 + zeros_8)))
    sigma_ith = translation_pr @ (green_inv @ translation)
    sigma_left.append(sigma_ith)

    for i in range(1, int(N/2) - 1):
        green_inv = inv((-1.0*(green_gi(i) + sigma_ith)))
        sigma_ith = translation_pr @ (green_inv @ translation)
        sigma_left.append(sigma_ith)

    green_temp0_block11 = np.block([[-1.0*g_block11(0), -1.0*g_block12],[-1.0*g_block21, -1.0*g_block11(-1)]])
    green_temp0_block22 = np.block([[-1.0*g_block11(1), zeros_4],[zeros_4, -1.0*g_block11(-2)]])

    green_temp0 = np.block([
        [green_temp0_block11 - sigma_left[0], -1.0*translation],
        [-1.0*translation_pr, green_temp0_block22 - sigma_right[-2]]
    ])

    # ---------- Full Green's function matrix ----------

    greens_func_matrix = inv(green_temp0)
    
    green_temp_block11 = lambda i: np.block([[-1.0*g_block11(i), zeros_4],[zeros_4, -1.0*g_block11(-i-1)]])
    green_temp_block22 = lambda i: np.block([[-1.0*g_block11(i + 1), zeros_4],[zeros_4, -1.0*g_block11(-i)]])
    green_temp = lambda i: np.block([
        [green_temp_block11(i) - sigma_left[i], -1.0*translation],
        [-1.0*translation_pr, green_temp_block22(i) - sigma_right[-i]]
    ])

    for i in range(1, int(N/2) - 2):
        greens_func_matrix += inv(green_temp(i))

    green_tempN_block11 = np.block([[-1.0*g_block11(int(N/2)-2), zeros_4],[zeros_4, -1.0*g_block11(int(N/2)+1)]])
    green_tempN_block22 = np.block([[-1.0*g_block11(int(N/2)-1), -1.0*h_hopping],[-1.0*h_hopping.T, -1.0*g_block11(int(N/2))]])

    green_tempN = np.block([
        [green_tempN_block11 - sigma_left[-2], -1.0*translation],
        [-1.0*translation_pr, green_tempN_block22 - sigma_right[0]]
    ])

    greens_func_matrix += inv(green_tempN)

    greens_func_ex_right = inv(green_temp0_block11 - sigma_right[-1])
    greens_func_ex_left = inv(green_tempN_block22 - sigma_left[-1])

    # --------- Nearest-neighbor Green's functions ---------

    G_12 = (
        greens_func_matrix[0:4, 8:12]
        + greens_func_matrix[12:16, 4:8]
        + (greens_func_ex_right[4:8, 0:4] * phase_neg)
        + greens_func_ex_left[0:4, 4:8]
    ) / N

    G_21 = (
        greens_func_matrix[8:12, 0:4]
        + greens_func_matrix[4:8, 12:16]
        + (greens_func_ex_right[0:4, 4:8] * phase_pos)
        + greens_func_ex_left[4:8, 0:4]
    ) / N

    return G_12, G_21

def winding_inv_vec(arguments, disorder):
    """
    Computes the 1D winding number for a chain with disorder.
    
    Args:
        arguments: tuple (e_z, mu, L, N)
            e_z: Zeeman energy
            mu: chemical potential
            q_N: system length (used for q discretization)
            Length: number of sites
        disorder: array of length N, disorder profile
    
    Returns:
        float: winding number ~0 or ~1
    """
    
    e_z, mu, q_N, Length = arguments
    V_x = cfg.Disorder_amplitude * disorder
    V_x = V_x[:Length]
    delta_ind = cfg.smsc_coupling
    t = cfg.hopping_amplitude
    N = Length
    alpha_mev = cfg.rashba_soc / cfg.lattice_const

    delta_k = (2*np.pi)/(q_N*N)
    q = np.arange(-np.pi/N, -0.1*np.pi/N, delta_k)

    h_onsite = (
        (2*t - mu) * szs0
        + e_z * szsx
        - delta_ind * sysy
    )

    h_hopping = (
        -t * szs0
        - 1.0j * (alpha_mev / 2.0) * szsy
    )

    translation = np.block([[h_hopping,zeros_4],[zeros_4,h_hopping.T]])
    translation_pr = np.block([[h_hopping.T,zeros_4],[zeros_4,h_hopping]])

    sigma_right = [zeros_8]

    gsr0_block11 = h_onsite + V_x[int(N/2)-1]*szs0
    gsr0_block22 = h_onsite + V_x[int(N/2)]*szs0

    green_gsr0 = np.block([[gsr0_block11, h_hopping],[h_hopping.T, gsr0_block22]])
    green_g0 = inv(-1.0*(green_gsr0 + zeros_8)) 
    sigma_ri = translation @ (green_g0 @ translation_pr)
    sigma_right.append(sigma_ri)

    gsr_i_block11 = lambda i: h_onsite + V_x[i]*szs0
    gsr_i_block22 = lambda i: h_onsite + V_x[-i-1]*szs0

    green_gsr_i = lambda i: np.block([[gsr_i_block11(i), zeros_4],[zeros_4, gsr_i_block22(i)]])

    for i in range(int(N/2) - 2, 0, -1):
        green_inv = inv(-1.0*(green_gsr_i(i) + sigma_ri))
        sigma_ri = translation @ (green_inv @ translation_pr)
        sigma_right.append(sigma_ri)
    
    ntp = 0
    
    for q0 in q:
        G_12, G_21 = fV(q0, h_onsite, h_hopping, translation, translation_pr, sigma_right, V_x, N)
        ntp += np.trace(chirality_op @ ((h_hopping.T @ G_12) - (h_hopping @ G_21)))
    q0 = (-0.1 * np.pi / N) + (0.5 *delta_k)
    while q0 < -0.00001:
        G_12, G_21 = fV(q0, h_onsite, h_hopping, translation, translation_pr, sigma_right, V_x, N)
        ntp += 0.1 * np.trace(chirality_op @ ((h_hopping.T @ G_12) - (h_hopping @ G_21)))
        q0 += 0.1*delta_k

    return np.real(-ntp/q_N)