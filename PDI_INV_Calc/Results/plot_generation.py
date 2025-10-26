#!/usr/bin/env python3
"""
Scan the project's results directory for saved .npz results and generate images
(saved to Results/Images/ or results/Images/) for 2D map results produced by
`cli.py` (winding_map, pfaffian_map, TVL_map/TVR_map, etc.).

This script is intentionally standalone and does not modify any library files.

Usage:
    python3 results/plot_generation.py

It will look for a directory named 'Results' first, then 'results' if the former
is not found. Images are written to <base_dir>/Images/.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def find_results_dir() -> str:
    for name in ("Results", "results"):
        if os.path.isdir(name):
            return name
    # fallback: create lowercase results
    os.makedirs("results", exist_ok=True)
    return "results"


def ensure_images_dir(base_dir: str) -> str:
    images_dir = os.path.join(base_dir, "Images")
    os.makedirs(images_dir, exist_ok=True)
    return images_dir


def plot_heatmap(arr: np.ndarray, xvals: np.ndarray, yvals: np.ndarray, title: str, outpath: str, cmap: str = "viridis"):
    plt.figure(figsize=(6, 5))
    extent = [float(xvals.min()), float(xvals.max()), float(yvals.min()), float(yvals.max())]
    plt.imshow(arr, origin="lower", aspect="auto", extent=extent, cmap=cmap)
    plt.colorbar(label=title)
    plt.xlabel("Zeeman (vz)")
    plt.ylabel("Chemical potential (mu)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_pfaffian_map(arr: np.ndarray, xvals: np.ndarray, yvals: np.ndarray, title: str, outpath: str):
    # Pfaffian values commonly -1 or 1 (or NaN). We'll plot discrete colormap.
    cmap = colors.ListedColormap(["#2b83ba", "#f7f7f7", "#d7191c"])  # blue, white, red
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    masked = np.ma.masked_invalid(arr)
    plt.figure(figsize=(6, 5))
    extent = [float(xvals.min()), float(xvals.max()), float(yvals.min()), float(yvals.max())]
    plt.imshow(masked, origin="lower", aspect="auto", extent=extent, cmap=cmap, norm=norm)
    plt.colorbar(ticks=[-1, 0, 1], label="Pfaffian")
    plt.xlabel("Zeeman (vz)")
    plt.ylabel("Chemical potential (mu)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_and_save(npzfile: str, images_dir: str):
    data = np.load(npzfile, allow_pickle=True)
    base = os.path.splitext(os.path.basename(npzfile))[0]

    # Helper to get mu/vz arrays if present
    mu_vals = data.get("mu_vals", None)
    vz_vals = data.get("vz_vals", None)

    # Winding map
    if "winding_map" in data:
        arr = data["winding_map"]
        if mu_vals is None or vz_vals is None:
            # try guess based on shape
            mu_vals = np.arange(arr.shape[0])
            vz_vals = np.arange(arr.shape[1])
        outpath = os.path.join(images_dir, base + "_winding.png")
        plot_heatmap(arr, vz_vals, mu_vals, "Winding invariant", outpath, cmap="viridis")
        print("Wrote", outpath)

    # Pfaffian map
    if "pfaffian_map" in data:
        arr = data["pfaffian_map"]
        if mu_vals is None or vz_vals is None:
            mu_vals = np.arange(arr.shape[0])
            vz_vals = np.arange(arr.shape[1])
        outpath = os.path.join(images_dir, base + "_pfaffian.png")
        plot_pfaffian_map(arr, vz_vals, mu_vals, "Pfaffian map", outpath)
        print("Wrote", outpath)

    # Topological visibility maps (TVL_map, TVR_map)
    if "TVL_map" in data or "TVR_map" in data:
        if "TVL_map" in data:
            TVL = data["TVL_map"]
            outpath = os.path.join(images_dir, base + "_TVL.png")
            if mu_vals is None or vz_vals is None:
                mu_vals = np.arange(TVL.shape[0])
                vz_vals = np.arange(TVL.shape[1])
            plot_heatmap(TVL, vz_vals, mu_vals, "Topological Visibility (L)", outpath, cmap="RdBu")
            print("Wrote", outpath)
        if "TVR_map" in data:
            TVR = data["TVR_map"]
            outpath = os.path.join(images_dir, base + "_TVR.png")
            plot_heatmap(TVR, vz_vals, mu_vals, "Topological Visibility (R)", outpath, cmap="RdBu")
            print("Wrote", outpath)

    # Spectrum: if present, plot eigenvalue lines vs zeeman range
    if "spectrum" in data:
        sp = data["spectrum"]
        vz_vals = data.get("vz_vals", None)
        outpath = os.path.join(images_dir, base + "_spectrum.png")
        try:
            # sp assumed shape (Nz, Neigs)
            Nz, Ne = sp.shape
            if vz_vals is None:
                vz_vals = np.arange(Nz)
            plt.figure(figsize=(7, 5))
            # Plot each eigenvalue as a line over zeeman
            for i in range(Ne):
                plt.plot(vz_vals, sp[:, i], "-k", lw=2)
            plt.xlabel("Zeeman (vz)")
            plt.ylabel("Energy")
            plt.title("Energy spectrum — eigenvalues vs Zeeman")
            plt.tight_layout()
            plt.savefig(outpath, dpi=200)
            plt.close()
            print("Wrote", outpath)
        except Exception as e:
            print(f"Failed to plot spectrum {base}: {e}")

    # Wavefunctions: save a simple plot for the first two wavefunctions if present
    if any(k.startswith("wf_") for k in data.files):
        keys = [k for k in data.files if k.startswith("wf_")]
        for k in keys[:2]:
            arr = data[k]
            outpath = os.path.join(images_dir, base + f"_{k}.png")
            plt.figure(figsize=(6, 3))
            plt.plot(arr)
            plt.xlabel("Site index")
            plt.ylabel("Probability")
            plt.title(f"{base} {k}")
            plt.tight_layout()
            plt.savefig(outpath, dpi=200)
            plt.close()
            print("Wrote", outpath)


def main():
    base_dir = find_results_dir()
    images_dir = ensure_images_dir(base_dir)
    pattern = os.path.join(base_dir, "*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        print("No .npz result files found in", base_dir)
        return
    for f in files:
        try:
            plot_and_save(f, images_dir)
        except Exception as e:
            print(f"Failed to plot {f}: {e}")


if __name__ == "__main__":
    main()
