#!/usr/bin/env python3
"""
Command-line interface for PDI_INV_Calc.

Features:
- Override configuration dataclass fields via repeated --set KEY=VALUE
- Subcommands that call functions in the libraries without editing them.

Subcommands:
  spectrum    : compute energy spectrum (calls generate_energy_spectrum)
  wavefuncs   : compute wavefunctions (calls generate_wavefunctions)
  pfaffian    : compute pfaffian of Hamiltonian (uses construct_Hamil with calc_pfaffian)
  winding     : compute 1D winding invariant (calls pdi.winding_inv_vec)
  topov       : compute topological visibility (calls topological_visibility.topological_visibility)

This script attempts to apply CLI overrides by constructing a new Config and then
injecting it into the imported library modules so their module-level `cfg` and
dependent variables are updated at runtime.
"""
import argparse
import importlib
import sys
import dataclasses
from typing import Any, Dict, Tuple
import os
import uuid
import time
from math import floor, log10
import numpy as np
from multiprocessing import Pool
from scipy.constants import hbar, m_e, eV, physical_constants
mu_B = physical_constants["Bohr magneton"][0] / (eV*1e3)  # Bohr magneton in meV

from libraries.config import defaults, load_config_only_env_and_cli, Config
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "libraries"))


def parse_set_args(set_args: list[str]) -> Dict[str, Any]:
    """Parse repeated KEY=VALUE pairs into dict."""
    out: Dict[str, Any] = {}
    if not set_args:
        return out
    # Infer types from the dataclass defaults (safer because of postponed annotations)
    defaults_obj = defaults()
    field_map = {f.name: type(getattr(defaults_obj, f.name)) for f in dataclasses.fields(Config)}
    for token in set_args:
        if "=" not in token:
            raise ValueError(f"Invalid --set argument: {token}. Expect KEY=VALUE")
        k, v = token.split("=", 1)
        if k not in field_map:
            raise KeyError(f"Unknown config key '{k}'. Valid keys: {', '.join(field_map.keys())}")
        target_type = field_map[k]
        out[k] = _coerce_str(v, target_type)
    return out


def _coerce_str(value: str, target_type: type):
    """Coerce string value to target_type used in Config dataclass.
    Handles basic builtins (bool,int,float,str)."""
    t = target_type
    low = value.lower()
    try:
        if t == bool:
            return low in ("1", "true", "yes", "on")
        if t == int:
            return int(value)
        if t == float:
            return float(value)
        if t == str:
            return value
    except Exception:
        raise ValueError(f"Could not coerce value '{value}' to {t}")
    # fallback
    return value


def inject_config_into_modules(cfg: Config):
    """Inject the constructed Config object into library modules and update
    common derived globals where needed. This avoids editing library files.

    We explicitly update the modules that rely on `cfg` at import time.
    """
    # modules to update
    modules = [
        "libraries.disorder_profiles",
        "libraries.system_hamiltonian_profiles",
        "libraries.pdi",
        "libraries.topological_visibility",
    ]

    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            print(f"Warning: could not import {mod_name}: {e}")
            continue
        # set the cfg object
        setattr(mod, "cfg", cfg)

        # Module-specific recalculations (best-effort):
        if mod_name.endswith("system_hamiltonian_profiles"):
            # update derived variables in that module
            Length = cfg.Lattice_sites
            t = cfg.hopping_amplitude
            alpha = cfg.rashba_soc / cfg.lattice_const
            coupling = cfg.smsc_coupling
            # regenerate disorder vector using the library's function
            try:
                V = cfg.Disorder_amplitude * importlib.import_module("libraries.disorder_profiles").disorder(Length=Length)
            except Exception:
                # fallback: keep existing V if regeneration fails
                V = getattr(mod, "V", None)
            setattr(mod, "Length", Length)
            setattr(mod, "t", t)
            setattr(mod, "alpha", alpha)
            setattr(mod, "coupling", coupling)
            setattr(mod, "V", V)
            # chiral operator S is static in code; keep as-is or recompute small matrix
            setattr(mod, "S", getattr(mod, "S", None))

        if mod_name.endswith("pdi"):
            setattr(mod, "cfg", cfg)
            setattr(mod, "Length", cfg.Lattice_sites)

        if mod_name.endswith("disorder_profiles"):
            setattr(mod, "cfg", cfg)

        if mod_name.endswith("topological_visibility"):
            setattr(mod, "cfg", cfg)


def _format_sig4(val: float) -> str:
    """Format a float into a 4-significant-figure compact token with no decimal point.

    Representation: [m]?{mantissa}e{exp}
    where mantissa is a 4-digit integer (rounded) and exp is the base-10 exponent.
    Negative values have an initial 'm' instead of '-'. Zero returns '0'.
    This keeps 4 significant figures and avoids any '.' characters.
    """
    if val == 0:
        return "0"
    sign = "m" if val < 0 else ""
    a = abs(float(val))
    order = int(floor(log10(a)))
    normalized = a / (10 ** order)
    mant = int(round(normalized * 1000))  # 4 significant digits -> 1.xxx * 10^3
    # Handle carry (e.g., 9.9996 -> 10000)
    if mant >= 10000:
        mant = mant // 10
        order += 1
    return f"{sign}{mant}e{order}"


def _make_filename(prefix: str, info: Dict[str, Any], ext: str = ".npz") -> str:
    """Create a filename with tokens based on info dict.

    For map outputs (info['is_map']=True) the filename will include only Length (L),
    disorder amplitude (disorder_amp) and delta, and the disorder_file name appended
    at the end (if provided). For single-point outputs it will include mu, vz (if
    provided), L, delta and disorder info.
    """
    parts = [prefix]

    is_map = bool(info.get("is_map", False))

    # Always include Length
    if "L" in info and info["L"] is not None:
        parts.append(f"L{int(info['L'])}")

    # For maps include disorder amplitude and delta, then append disorder file name
    if is_map:
        if "disorder_amp" in info and info.get("disorder_amp") is not None:
            parts.append(f"dis{_format_sig4(float(info['disorder_amp']))}")
        if "delta" in info and info.get("delta") is not None:
            parts.append(f"d{_format_sig4(float(info['delta']))}")
        # append mu/vz base points optionally (start of sweep)
        if "mu0" in info and info.get("mu0") is not None:
            parts.append(f"mu0{_format_sig4(float(info['mu0']))}")
        if "vz0" in info and info.get("vz0") is not None:
            parts.append(f"vz0{_format_sig4(float(info['vz0']))}")
        # If a disorder file name is provided, append it verbatim (safe characters expected)
        if "disorder_file" in info and info.get("disorder_file"):
            # strip path and keep basename
            parts.append(os.path.basename(str(info.get("disorder_file"))))

    else:
        # Single-point outputs: mu and vz tokens first (if present)
        for k in ("mu", "vz"):
            if k in info and info[k] is not None:
                v = info[k]
                if isinstance(v, int):
                    parts.append(f"{k}{v}")
                else:
                    parts.append(f"{k}{_format_sig4(float(v))}")
        # include delta
        if "delta" in info and info.get("delta") is not None:
            parts.append(f"d{_format_sig4(float(info['delta']))}")
        # include disorder amplitude or file
        if "disorder_amp" in info and info.get("disorder_amp") is not None:
            parts.append(f"dis{_format_sig4(float(info['disorder_amp']))}")
        if "disorder_file" in info and info.get("disorder_file"):
            parts.append(os.path.basename(str(info.get("disorder_file"))))

    # add timestamp and uuid for uniqueness
    ts = int(time.time())
    uid = uuid.uuid4().hex[:8]
    parts.append(f"t{ts}")
    parts.append(uid)
    filename = "_".join(parts) + ext
    # ensure output dir exists
    outdir = os.path.join(os.getcwd(), "Results")
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, filename)


def _save_npz(filename: str, **arrays):
    np.savez_compressed(filename, **arrays)


# --- Multiprocessing worker initializers and workers (module-level so they're picklable) ---
# Globals set by initializer in worker processes
_WORKER_CFG = None
_WORKER_PF_PHI = None
_WORKER_DISORDER_ARR = None
_WORKER_QN = None
_WORKER_VFULL = None
_WORKER_MU_LEAD = None

def init_pf_worker(cfg_pickle, phi):
    """Initializer for pfaffian worker processes."""
    global _WORKER_CFG, _WORKER_PF_PHI
    _WORKER_CFG = cfg_pickle
    _WORKER_PF_PHI = phi
    import importlib as _imp
    # Ensure relative disorder file paths inside library imports resolve correctly
    base_dir = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.join(base_dir, "libraries")
    orig_cwd = os.getcwd()
    try:
        if os.path.isdir(lib_dir):
            os.chdir(lib_dir)
        mods = [
            "libraries.system_hamiltonian_profiles",
            "libraries.pfaffian",
        ]
        for mn in mods:
            try:
                m = _imp.import_module(mn)
                setattr(m, "cfg", cfg_pickle)
                if mn.endswith("system_hamiltonian_profiles"):
                    setattr(m, "Length", cfg_pickle.Lattice_sites)
                    setattr(m, "t", cfg_pickle.hopping_amplitude)
                    setattr(m, "alpha", cfg_pickle.rashba_soc / cfg_pickle.lattice_const)
                    setattr(m, "coupling", cfg_pickle.smsc_coupling)
            except Exception:
                pass
    finally:
        try:
            os.chdir(orig_cwd)
        except Exception:
            pass


def pfaff_worker(task):
    mu0, vz0 = task
    import importlib as _imp
    mod_local = _imp.import_module("libraries.system_hamiltonian_profiles")
    pf_local = _imp.import_module("libraries.pfaffian")
    try:
        Hloc = mod_local.construct_Hamil(float(mu0), float(vz0), phi=_WORKER_PF_PHI, calc_pfaffian=True)
        val = pf_local.pfaffian(Hloc)
    except Exception:
        val = np.nan
    return (mu0, vz0, float(val))


def init_winding_worker(cfg_pickle, disorder_array, qN):
    """Initializer for winding worker processes."""
    global _WORKER_CFG, _WORKER_DISORDER_ARR, _WORKER_QN
    _WORKER_CFG = cfg_pickle
    _WORKER_DISORDER_ARR = disorder_array
    _WORKER_QN = qN
    import importlib as _imp
    import os
    # Ensure relative disorder file paths inside library imports resolve correctly
    base_dir = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.join(base_dir, "libraries")
    orig_cwd = os.getcwd()
    try:
        if os.path.isdir(lib_dir):
            os.chdir(lib_dir)
        mods = [
            "libraries.disorder_profiles",
            "libraries.system_hamiltonian_profiles",
            "libraries.pdi",
            "libraries.topological_visibility",
        ]
        for mn in mods:
            try:
                m = _imp.import_module(mn)
                setattr(m, "cfg", cfg_pickle)
                if mn.endswith("system_hamiltonian_profiles"):
                    setattr(m, "Length", cfg_pickle.Lattice_sites)
                    setattr(m, "t", cfg_pickle.hopping_amplitude)
                    setattr(m, "alpha", cfg_pickle.rashba_soc / cfg_pickle.lattice_const)
                    setattr(m, "coupling", cfg_pickle.smsc_coupling)
            except Exception:
                pass
    finally:
        try:
            os.chdir(orig_cwd)
        except Exception:
            pass


def winding_worker(task):
    mu0, vz0 = task
    import importlib as _imp
    pdi_m = _imp.import_module("libraries.pdi")
    dis_m = _imp.import_module("libraries.disorder_profiles")
    try:
        d_arr = dis_m.disorder(Length=pdi_m.cfg.Lattice_sites)
    except Exception:
        d_arr = _WORKER_DISORDER_ARR
    args_tuple = (vz0, mu0, _WORKER_QN, pdi_m.cfg.Lattice_sites)
    qN_new = _WORKER_QN
    val = -4.0
    while True:
        val = pdi_m.winding_inv_vec(args_tuple, d_arr)
        if ((abs(val - 1.0) < pdi_m.cfg.pdi_conv_tol) or (abs(val) < pdi_m.cfg.pdi_conv_tol)):
            break
        qN_new = 2 * qN_new
        args_tuple = (vz0, mu0, qN_new, pdi_m.cfg.Lattice_sites)
    return (mu0, vz0, float(val))


def init_topov_worker(cfg_pickle, Vimp_pickle, mu_lead):
    """Initializer for topov worker processes."""
    global _WORKER_CFG, _WORKER_VFULL, _WORKER_MU_LEAD
    _WORKER_CFG = cfg_pickle
    _WORKER_VFULL = Vimp_pickle
    _WORKER_MU_LEAD = mu_lead
    import importlib as _imp
    import os
    # Ensure relative disorder file paths inside library imports resolve correctly
    base_dir = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.join(base_dir, "libraries")
    orig_cwd = os.getcwd()
    try:
        if os.path.isdir(lib_dir):
            os.chdir(lib_dir)
        mods = [
            "libraries.disorder_profiles",
            "libraries.system_hamiltonian_profiles",
            "libraries.pdi",
            "libraries.topological_visibility",
        ]
        for mn in mods:
            try:
                m = _imp.import_module(mn)
                setattr(m, "cfg", cfg_pickle)
                if mn.endswith("system_hamiltonian_profiles"):
                    setattr(m, "Length", cfg_pickle.Lattice_sites)
                    setattr(m, "t", cfg_pickle.hopping_amplitude)
                    setattr(m, "alpha", cfg_pickle.rashba_soc / cfg_pickle.lattice_const)
                    setattr(m, "coupling", cfg_pickle.smsc_coupling)
            except Exception:
                pass
    finally:
        try:
            os.chdir(orig_cwd)
        except Exception:
            pass


def topov_worker(task):
    mu0, vz0 = task
    import importlib as _imp
    tvm = _imp.import_module("libraries.topological_visibility")
    dism = _imp.import_module("libraries.disorder_profiles")
    try:
        Vimp_local = _WORKER_VFULL
    except Exception:
        try:
            Vimp_local = dism.disorder(Length=tvm.cfg.Lattice_sites)
        except Exception:
            Vimp_local = None

    # Compute magnetic field B corresponding to this zeeman energy vz0:
    # B = vz / (0.5 * g_factor * mu_B)
    try:
        g = getattr(tvm.cfg, "g_factor", None)
        Bval = float(vz0) / (0.5 * g * mu_B) if (g is not None and g != 0) else None
    except Exception:
        Bval = None
    params = {"mu": float(mu0), "B": Bval, "Vimp": Vimp_local, "mu_l": (_WORKER_MU_LEAD if _WORKER_MU_LEAD is not None else getattr(tvm.cfg, "mu_lead_pot", None))}
    try:
        tl, tr = tvm.TV_calc(params)
    except Exception:
        tl, tr = np.nan, np.nan
    return (mu0, vz0, float(tl), float(tr))


def main(argv: list[str] | None = None):
    # Disable abbreviated option matching so short prefixes like `--mu` do not
    # ambiguously match multiple long options (e.g. --mu_lead_pot).
    parser = argparse.ArgumentParser(description="PDI_INV_Calc CLI — override config and run library routines", allow_abbrev=False)
    parser.add_argument("--set", "-s", action="append", help="Override a config field: KEY=VALUE. Can be repeated.")
    # Show effective config by default; provide a --no-show-config to disable
    parser.add_argument("--no-show-config", action="store_true", help="Disable printing the effective config (enabled by default)")
    # Dynamically add command-line flags for every Config field so users can pass
    # e.g. --zeeman_count 50 to override that config parameter. Use explicit
    # underscores (user requested names like --zeeman_count).
    def _arg_type_for(t):
        # Return a callable for argparse 'type' parameter that coerces strings to t
        # t may be a string (from postponed annotations). We'll not rely on it
        # here; instead callers should prefer passing the default value's type.
        def _b(s: str) -> bool:
            return str(s).lower() in ("1", "true", "yes", "on")
        return _b

    def _arg_type_and_name_from_default(default_value):
        """Return (argparse_type_callable, typename) based on the default value's type."""
        tv = type(default_value)
        if tv is bool:
            return (lambda s: str(s).lower() in ("1", "true", "yes", "on"), "bool")
        if tv is int:
            return (int, "int")
        if tv is float:
            return (float, "float")
        if tv is str:
            return (str, "str")
        # fallback to str
        return (str, tv.__name__ if hasattr(tv, "__name__") else str(tv))

    # Add flags for all dataclass fields with default None so we can detect
    # when the user provided them.
    # Use the dataclass defaults as argparse defaults so the parser shows the
    # actual default values from `libraries.config.Config`.
    config_defaults = defaults()
    for f in dataclasses.fields(Config):
        name = f.name
        argname = f"--{name}"
        default_value = getattr(config_defaults, name)
        arg_type_callable, type_name = _arg_type_and_name_from_default(default_value)
        # Use SUPPRESS so we can detect whether the user actually provided the flag
        # (if suppressed the attribute won't exist on args). Show the real default
        # in the help text instead of using argparse's default mechanism.
        parser.add_argument(
            argname,
            dest=name,
            type=arg_type_callable,
            default=argparse.SUPPRESS,
            help=f"Override config.{name} (type: {type_name}, default: {default_value})",
        )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sp_spec = subparsers.add_parser("spectrum", help="Generate energy spectrum")
    sp_spec.add_argument("--mu", type=float, required=False, help="Chemical potential (overrides cfg)")
    sp_spec.add_argument("--zeeman-low", type=float, required=False)
    sp_spec.add_argument("--zeeman-high", type=float, required=False)
    sp_spec.add_argument("--num-eigvals", type=int, default=30)
    sp_spec.add_argument("--set", "-s", action="append", help="Override a config field: KEY=VALUE. Can be repeated.")

    sp_wf = subparsers.add_parser("wavefuncs", help="Generate wavefunctions")
    # make mu/zeeman optional; fall back to cfg.mu_0 / cfg.zeeman_0 when omitted
    sp_wf.add_argument("--mu", type=float, required=False)
    sp_wf.add_argument("--zeeman", type=float, required=False)
    sp_wf.add_argument("--num-eigvals", type=int, default=10)
    sp_wf.add_argument("--set", "-s", action="append", help="Override a config field: KEY=VALUE. Can be repeated.")

    sp_pf = subparsers.add_parser("pfaffian", help="Compute Pfaffian of the Hamiltonian (returns +-1)")
    # mu/zeeman optional: single-point when both provided, else compute map over cfg ranges
    sp_pf.add_argument("--mu", type=float, required=False)
    sp_pf.add_argument("--zeeman", type=float, required=False)
    sp_pf.add_argument("--phi", type=float, default=0.0)
    sp_pf.add_argument("--set", "-s", action="append", help="Override a config field: KEY=VALUE. Can be repeated.")

    sp_wind = subparsers.add_parser("winding", help="Compute winding_inv_vec")
    # make mu/zeeman optional; maps use cfg ranges when omitted
    sp_wind.add_argument("--mu", type=float, required=False)
    sp_wind.add_argument("--zeeman", type=float, required=False)
    sp_wind.add_argument("--qN", type=int, default=20)
    sp_wind.add_argument("--set", "-s", action="append", help="Override a config field: KEY=VALUE. Can be repeated.")

    sp_tv = subparsers.add_parser("topov", help="Compute topological visibility via Kwant")
    # mu optional; magnetic field B will be computed from zeeman values and cfg.g_factor
    sp_tv.add_argument("--mu", type=float, required=False)
    sp_tv.add_argument("--mu-lead", type=float, required=False)
    sp_tv.add_argument("--disorder-file", type=str, required=False)
    sp_tv.add_argument("--set", "-s", action="append", help="Override a config field: KEY=VALUE. Can be repeated.")

    args = parser.parse_args(argv)

    # Parse config overrides from --set and from explicit flags we generated.
    try:
        set_overrides = parse_set_args(args.set) if args.set else {}
    except Exception as e:
        print(f"Error parsing --set arguments: {e}")
        sys.exit(2)

    # Collect flag-based overrides (these were added dynamically for each Config field)
    flag_overrides: Dict[str, Any] = {}
    for f in dataclasses.fields(Config):
        if hasattr(args, f.name):
            val = getattr(args, f.name)
            # Only treat it as a user override if the user actually provided a
            # non-None value. Subparsers define options like --zeeman-low which
            # create attributes on args even when not passed (they default to
            # None). We must not treat those as overrides that wipe out
            # --set values.
            if val is not None:
                flag_overrides[f.name] = val

    # Merge: --set values applied first, then explicit flags override them.
    cli_overrides = dict(set_overrides)
    cli_overrides.update(flag_overrides)

    # Use helper in config module to construct full Config with overrides (best-effort)
    try:
        cfg = load_config_only_env_and_cli(cli_overrides)
    except Exception:
        # fallback: start from defaults and apply overrides manually
        base = dataclasses.asdict(defaults())
        for k, v in cli_overrides.items():
            base[k] = v
        cfg = Config(**base)

    # Show effective config if requested (default: show)
    show_config = not getattr(args, "no_show_config", False)
    if show_config:
        try:
            # print dataclass as dict for readability
            from dataclasses import asdict

            print("Effective Config:")
            for k, v in asdict(cfg).items():
                print(f"  {k}: {v}")
        except Exception:
            print(cfg)

    # Inject cfg into library modules so their functions use updated values
    # Some libraries expect relative paths (e.g. disorder file) that are resolved
    # relative to the `libraries/` folder at import time. To keep imports working
    # without editing library files, temporarily change CWD to the libraries
    # directory while importing/injecting config, then restore original CWD.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.join(base_dir, "libraries")
    orig_cwd = os.getcwd()
    try:
        if os.path.isdir(lib_dir):
            os.chdir(lib_dir)
        inject_config_into_modules(cfg)
    finally:
        os.chdir(orig_cwd)

    # Resolve default disorder file path (absolute) so library calls don't depend on CWD
    default_disorder_file = os.path.join(base_dir, "Disorder_profiles", "Disorder_3m_St.txt")

    # Execute requested subcommand
    if args.command == "spectrum":
        mod = importlib.import_module("libraries.system_hamiltonian_profiles")
        mu = args.mu if args.mu is not None else cfg.c_pot_mu_low
        spectrum = mod.generate_energy_spectrum(mu, zeeman_low=getattr(cfg, "zeeman_low", None), zeeman_high=getattr(cfg, "zeeman_high", None), phi_0=0.0, num_eigvals=args.num_eigvals)
        vz_vals = np.linspace(cfg.zeeman_low, cfg.zeeman_high, cfg.zeeman_count)
        fname = _make_filename("spectrum", {"mu": mu, "vz": None, "L": cfg.Lattice_sites, "delta": cfg.smsc_coupling, "disorder_amp": cfg.Disorder_amplitude})
        _save_npz(fname, spectrum=spectrum, vz_vals=vz_vals)
        print(f"Saved spectrum to: {fname}")

    elif args.command == "wavefuncs":
        mod = importlib.import_module("libraries.system_hamiltonian_profiles")
        # use config defaults when mu/zeeman not provided
        mu_arg = args.mu if (hasattr(args, "mu") and args.mu is not None) else cfg.mu_0
        vz_arg = args.zeeman if (hasattr(args, "zeeman") and args.zeeman is not None) else cfg.zeeman_0
        wf = mod.generate_wavefunctions(mu_arg, vz_arg, phi_0=0.0, num_eigvals=args.num_eigvals)
        fname = _make_filename("wavefuncs", {"mu": mu_arg, "vz": vz_arg, "L": cfg.Lattice_sites, "delta": cfg.smsc_coupling, "disorder_amp": cfg.Disorder_amplitude})
        # wavefuncs may return tuple of arrays
        if isinstance(wf, tuple):
            arrs = {f"wf_{i}": w for i, w in enumerate(wf)}
        else:
            arrs = {"wavefuncs": wf}
        _save_npz(fname, **arrs)
        print(f"Saved wavefunctions to: {fname}")

    elif args.command == "pfaffian":
        mod = importlib.import_module("libraries.system_hamiltonian_profiles")
        pfm = importlib.import_module("libraries.pfaffian")
        # If both mu and zeeman provided -> single point; else compute map over config ranges
        if (hasattr(args, "mu") and args.mu is not None) and (hasattr(args, "zeeman") and args.zeeman is not None):
            mu_arg = args.mu
            vz_arg = args.zeeman
            H = mod.construct_Hamil(float(mu_arg), float(vz_arg), phi=args.phi, calc_pfaffian=True)
            val = pfm.pfaffian(H)
            fname = _make_filename("pfaffian", {"mu": mu_arg, "vz": vz_arg, "L": cfg.Lattice_sites, "delta": cfg.smsc_coupling, "disorder_amp": cfg.Disorder_amplitude})
            _save_npz(fname, pfaffian=val)
            print(f"Saved pfaffian to: {fname}")
        else:
            # Sweep ranges and compute pfaffian map in parallel
            mu_vals = np.linspace(cfg.c_pot_mu_low, cfg.c_pot_mu_high, cfg.c_pot_mu_count)
            vz_vals = np.linspace(cfg.zeeman_low, cfg.zeeman_high, cfg.zeeman_count)

            # pfaffian worker functions moved to module scope (init_pf_worker / pfaff_worker)

            tasks = [(mu0, vz0) for mu0 in mu_vals for vz0 in vz_vals]
            results = []
            from multiprocessing import cpu_count
            with Pool(processes=min(8, cpu_count()), initializer=init_pf_worker, initargs=(cfg, args.phi)) as pool:
                for res in pool.imap_unordered(pfaff_worker, tasks):
                    results.append(res)

            pf_map = np.zeros((len(mu_vals), len(vz_vals)), dtype=float)
            mu_index = {mu: i for i, mu in enumerate(mu_vals)}
            vz_index = {vz: j for j, vz in enumerate(vz_vals)}
            for mu0, vz0, val in results:
                i = mu_index[mu0]
                j = vz_index[vz0]
                pf_map[i, j] = val

            fname = _make_filename("pfaffian_map", {"is_map": True, "L": cfg.Lattice_sites, "disorder_amp": cfg.Disorder_amplitude, "delta": cfg.smsc_coupling, "mu0": cfg.c_pot_mu_low, "vz0": cfg.zeeman_low})
            _save_npz(fname, pfaffian_map=pf_map, mu_vals=mu_vals, vz_vals=vz_vals)
            print(f"Saved pfaffian map to: {fname}")

    elif args.command == "winding":
        # Generate a map over mu (y axis) and zeeman (x axis) and save result.
        mod_pdi = importlib.import_module("libraries.pdi")
        disorder_mod = importlib.import_module("libraries.disorder_profiles")
        # prepare disorder array
        try:
            # Prefer calling disorder with an explicit absolute file path so the
            # library does not rely on the current working directory.
            disorder_arr = disorder_mod.disorder(file_path=default_disorder_file, Length=cfg.Lattice_sites)
        except Exception:
            # Fallback: try calling with Length only but still prefer the absolute path
            disorder_arr = disorder_mod.disorder(file_path=default_disorder_file, Length=cfg.Lattice_sites)

        mu_vals = np.linspace(cfg.c_pot_mu_low, cfg.c_pot_mu_high, cfg.c_pot_mu_count)
        vz_vals = np.linspace(cfg.zeeman_low, cfg.zeeman_high, cfg.zeeman_count)

        # winding worker functions moved to module scope (init_winding_worker / winding_worker)
            

        tasks = [(mu0, vz0) for mu0 in mu_vals for vz0 in vz_vals]
        results = []
        # use multiprocessing Pool with initializer
        from multiprocessing import cpu_count
        # initializer will receive (cfg, disorder_arr, qN)
        with Pool(processes=min(8, cpu_count()), initializer=init_winding_worker, initargs=(cfg, disorder_arr, args.qN)) as pool:
            for res in pool.imap_unordered(winding_worker, tasks):
                results.append(res)

        # aggregate into 2D array: rows mu, cols vz
        map_arr = np.zeros((len(mu_vals), len(vz_vals)), dtype=float)
        mu_index = {mu: i for i, mu in enumerate(mu_vals)}
        vz_index = {vz: j for j, vz in enumerate(vz_vals)}
        for mu0, vz0, val in results:
            i = mu_index[mu0]
            j = vz_index[vz0]
            map_arr[i, j] = val

        fname = _make_filename("winding_map", {"is_map": True, "L": cfg.Lattice_sites, "disorder_amp": cfg.Disorder_amplitude, "delta": cfg.smsc_coupling, "mu0": cfg.c_pot_mu_low, "vz0": cfg.zeeman_low})
        _save_npz(fname, winding_map=map_arr, mu_vals=mu_vals, vz_vals=vz_vals)
        print(f"Saved winding map to: {fname}")

    elif args.command == "topov":
        tv_mod = importlib.import_module("libraries.topological_visibility")
        disorder_mod = importlib.import_module("libraries.disorder_profiles")

        # If user provided single mu/zeeman compute single point and save, otherwise compute map over mu/vz
        if hasattr(args, "disorder_file") and args.disorder_file:
            df = args.disorder_file
            # if a relative path was provided, interpret it relative to repo root
            if not os.path.isabs(df):
                df = os.path.join(base_dir, df)
            Vfull = disorder_mod.read_disorder_profile(df, cfg.Lattice_sites)
        else:
            Vfull = disorder_mod.disorder(file_path=default_disorder_file, Length=cfg.Lattice_sites)
        
        Vfull = cfg.Disorder_amplitude * Vfull

        # Build ranges
        mu_vals = np.linspace(cfg.c_pot_mu_low, cfg.c_pot_mu_high, cfg.c_pot_mu_count)
        vz_vals = np.linspace(cfg.zeeman_low, cfg.zeeman_high, cfg.zeeman_count)

        # topov worker functions moved to module scope (init_topov_worker / topov_worker)

        tasks = [(mu0, vz0) for mu0 in mu_vals for vz0 in vz_vals]
        results = []
        from multiprocessing import cpu_count
        mu_lead_arg = args.mu_lead if (hasattr(args, "mu_lead") and args.mu_lead is not None) else cfg.mu_lead_pot
        with Pool(processes=min(8, cpu_count()), initializer=init_topov_worker, initargs=(cfg, Vfull, mu_lead_arg)) as pool:
            for res in pool.imap_unordered(topov_worker, tasks):
                results.append(res)

        TVL_map = np.zeros((len(mu_vals), len(vz_vals)), dtype=float)
        TVR_map = np.zeros((len(mu_vals), len(vz_vals)), dtype=float)
        mu_index = {mu: i for i, mu in enumerate(mu_vals)}
        vz_index = {vz: j for j, vz in enumerate(vz_vals)}
        for mu0, vz0, tl, tr in results:
            i = mu_index[mu0]
            j = vz_index[vz0]
            TVL_map[i, j] = tl
            TVR_map[i, j] = tr

        fname = _make_filename("topov_map", {"is_map": True, "L": cfg.Lattice_sites, "disorder_amp": cfg.Disorder_amplitude, "delta": cfg.smsc_coupling, "mu0": cfg.c_pot_mu_low, "vz0": cfg.zeeman_low, "disorder_file": (args.disorder_file if args.disorder_file else None)})
        _save_npz(fname, TVL_map=TVL_map, TVR_map=TVR_map, mu_vals=mu_vals, vz_vals=vz_vals)
        print(f"Saved topological visibility map to: {fname}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
