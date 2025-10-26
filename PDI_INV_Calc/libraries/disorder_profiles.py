import numpy as np
from config import Config, defaults

cfg = Config() or defaults()

def read_disorder_profile(file_path, Length):
    """
    Reads a disorder profile from a text file.

    Parameters:
    file_path (str): Path to the text file containing the disorder profile.
    Length (int): Number of lattice sites to read.

    Returns:
    np.ndarray: Array containing the disorder profile.
    """
    disorder_profile = np.array(np.loadtxt(open(file_path, "rb"), skiprows=0), dtype=float)
    return disorder_profile[:Length]

def generate_disorder(Length, disorder_centers_N, sigma, seed=None):
    """
    Generates a Gaussian disorder profile.

    Parameters:
    Length (int): Number of lattice sites.
    disorder_centers_N (int): Number of disorder centers.
    sigma (float): Standard deviation of the Gaussian disorder.
    seed (int, optional): Seed for random number generator.

    Returns:
    np.ndarray: Array containing the generated disorder profile.
    """
    if seed is None:
        seed = np.random.randint(0, 1000)
    np.random.seed(seed)

    disorder_profile = np.zeros(Length)
    disorder_centers = np.random.choice(np.arange(Length), size=disorder_centers_N, replace=False)
    
    for center in disorder_centers:
        distances = np.arange(Length) - center
        weights = np.random.normal(0, 1)
        gaussian = weights*np.exp(-0.5 * (distances / sigma) ** 2)
        disorder_profile += gaussian

    disorder_profile -= np.mean(disorder_profile) # Zero mean
    disorder_profile /= np.sqrt(np.mean(disorder_profile**2)) # Normalize RMS to 1
    np.savetxt(f"../Disorder_profiles/disorder_profile_s_{seed}.txt", disorder_profile)
    return disorder_profile

def disorder(file_path="../Disorder_profiles/Disorder_3m_St.txt", Length=None, disorder_centers_N=None, sigma=None, seed=None):
    """
    Returns a disorder profile either by reading from a file or generating it.

    Parameters:
    file_path (str): Path to the text file containing the disorder profile.
    Length (int): Number of lattice sites.
    disorder_centers_N (int, optional): Number of disorder centers for generation.
    sigma (float, optional): Standard deviation for Gaussian disorder generation.
    seed (int, optional): Seed for random number generator.

    Returns:
    np.ndarray: Array containing the disorder profile.
    """
    Length = cfg.Lattice_sites if Length is None else Length
    if disorder_centers_N is not None and sigma is not None:
        return generate_disorder(Length, disorder_centers_N, sigma, seed)
    return read_disorder_profile(file_path, Length)