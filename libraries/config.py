from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict
from scipy.constants import hbar, m_e, eV, physical_constants
import os, json

ENV_PREFIX = "INVCALC_"

@dataclass(frozen=True)
class Config:
    #Numerical tolerance values
    pdi_conv_tol: float = 1e-1
    pfaff_tol: float = 1e-14
    tv_ph_tol: float = 1e-14
    #Algorithm parameters
    Lattice_sites: int = 300 # number of lattice sites
    Disorder_amplitude: float = 0.5 # meV
    rashba_soc: float = 140.0 # meV*Angstrom
    m_eff: float = 0.023 # effective mass in units of electron mass
    lattice_const: float = 100.0 # Angstrom
    etam: float = ((hbar**2) * (10**20.0)) / (m_e*eV) #Constant to convert to meV
    hopping_amplitude: float = (1e3 * etam) / (2.0 * (lattice_const**2.0) * m_eff) # meV
    g_factor: float = 25.0 # g-factor
    smsc_coupling: float = 0.5 # meV
    sc_order_param: float = 0.3 # meV
    dynes_broadening: float = 0.001 # meV
    mu_lead_pot: float = 8.5 # meV
    tunnel_barrier_amplitude: float = 15.0 # meV
    tunnel_barrier_width: int = 1 # lattice sites
    #Parameter sweep ranges
    mu_0: float = 0.0 # meV
    zeeman_0: float = 0.8 # meV
    c_pot_mu_low: float = -2.0 # meV
    c_pot_mu_high: float = 3.0 # meV
    c_pot_mu_count: int = 50 # number of chemical potential points
    zeeman_low: float = 0.0 # meV
    zeeman_high: float = 1.25 # meV
    zeeman_count: int = 30 # number of Zeeman field points
    v_bias_low: float = -0.1 # meV
    v_bias_high: float = 0.1 # meV
    v_bias_count: int = 30 # number of bias voltage points

def defaults() -> Config:
    return Config()

def _coerce(value: str, target_type: type) -> Any:
    if target_type == bool:
        return value.lower() in ("1", "true", "yes", "on")
    if target_type == int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is str:
        return value
    return None

def load_from_env(base: Config, *, prefix: str = ENV_PREFIX) -> Config:
    current = asdict(base)
    for k, v in list(current.items()):
        env_key = prefix + k.upper()
        if env_key in os.environ:
            current[k] = _coerce(os.environ[env_key], type(v))
    return Config(**current)

def merge_env_cli(*, env_first: Config, cli_overrides: Dict[str, Any] | None) -> Config:
    current = asdict(env_first)
    if cli_overrides is not None:
        for k, v in cli_overrides.items():
            if k in current and v is not None:
                current[k] = v
    return Config(**current)

def load_config_only_env_and_cli(cli_overrides: Dict[str, Any] | None = None, env_prefix: str = ENV_PREFIX) -> Config:
    base = defaults()
    env_applied = load_from_env(base, prefix=env_prefix)
    return merge_env_cli(env_first=env_applied, cli_overrides=cli_overrides)