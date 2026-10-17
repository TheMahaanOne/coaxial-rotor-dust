"""
Radial diagnostic for the downstream-rotor torque trend.
========================================================

Reads DUST sectional-loads .dat files and shows WHERE along the blade span
the thrust and torque are produced, and how that radial distribution changes
between two (or more) stacking distances.

The point: torque is Q = integral of (Fx*r + Fy*c/4) dr, so it is weighted by
radius and dominated by the outboard span. A change in the total torque tells
you nothing about its cause; the radial distribution does.

Usage
-----
    python radial_diag.py ^
        --case "0.30R=coaxial_setup\\contra\\0.3_contra\\PostPro" ^
        --case "0.40R=coaxial_setup\\contra\\0.4_contra\\PostPro" ^
        --rpm 5000 --out radial_diag.png

By default it analyses the DOWNSTREAM rotor (the one with the lower ref_off z).
Use --rotor upstream to look at the other one.

Requires: numpy, matplotlib
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HEADER_RE = re.compile(r"n_sec\s*:\s*(\d+)\s*;\s*n_time\s*:\s*(\d+)")
NAME_RE = re.compile(
    r"rotor_+0*(\d+)_+blade_+0*(\d+)_+([A-Za-z][A-Za-z0-9]*)(?:_[\d.]+)?\.dat$",
    re.IGNORECASE,
)

TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def parse_dat(path):
    text = Path(path).read_text(errors="ignore")
    m = HEADER_RE.search(text)
    if not m:
        raise ValueError(f"{Path(path).name}: no 'n_sec ; n_time' header")
    n_sec = int(m.group(1))

    rows = [np.fromstring(l.strip(), sep=" ")
            for l in text.splitlines()
            if l.strip() and not l.lstrip().startswith("#")]
    geom = [r for r in rows if r.size == n_sec]
    data = [r for r in rows if r.size in (n_sec + 1, n_sec + 13)]
    if len(geom) < 3 or not data:
        raise ValueError(f"{Path(path).name}: unexpected layout")

    arr = np.asarray(data)
    ref_off = arr[:, n_sec + 10:n_sec + 13] if arr.shape[1] == n_sec + 13 else None
    return arr[:, 0], geom[0], geom[1], geom[2], arr[:, 1:n_sec + 1], ref_off


def discover(folder):
    found = {}
    for p in sorted(Path(folder).glob("*.dat")):
        m = NAME_RE.search(p.name)
        if m:
            found.setdefault((int(m.group(1)), int(m.group(2))), {})[m.group(3)] = p
    return found


def analyse(folder, rpm, which):
    """Return (r, dTdr, dQdr) time-averaged over the last revolution,
    summed over both blades of the chosen rotor."""
    blades = discover(folder)
    if not blades:
        sys.exit(f"no sectional-loads .dat files in {folder}")

    # identify rotors by their reference-frame z offset
    zs = {}
    for (rot, bl), q in blades.items():
        if bl == 1 and "Fx" in q:
            ro = parse_dat(q["Fx"])[5]
            if ro is not None:
                zs[rot] = ro[0, 2]
    if len(zs) < 2:
        sys.exit(f"could not identify two rotors in {folder} (found z: {zs})")

    downstream = min(zs, key=zs.get)   # lower z sits in the wake
    upstream = max(zs, key=zs.get)
    target = downstream if which == "downstream" else upstream

    r = dTdr = dQdr = None
    n_blades = 0
    for (rot, bl), q in sorted(blades.items()):
        if rot != target:
            continue
        if not all(k in q for k in ("Fx", "Fy", "Fz")):
            print(f"  ! rotor {rot} blade {bl}: missing Fx/Fy/Fz, skipped")
            continue
        t, y_cen, y_span, chord, fx, _ = parse_dat(q["Fx"])
        _, _, _, _, fy, _ = parse_dat(q["Fy"])
        _, _, _, _, fz, _ = parse_dat(q["Fz"])

        dt = float(np.median(np.diff(t)))
        n_rev = max(1, int(round((60.0 / rpm) / dt)))
        sl = slice(-n_rev, None)

        # per-unit-span distributions, averaged over the final revolution
        this_T = fz[sl].mean(axis=0)
        this_Q = (fx * y_cen + fy * (chord / 4.0))[sl].mean(axis=0)

        if r is None:
            r, dTdr, dQdr = y_cen, this_T.copy(), this_Q.copy()
        else:
            dTdr += this_T
            dQdr += this_Q
        n_blades += 1

    if r is None:
        sys.exit(f"no usable blades for the {which} rotor in {folder}")
    print(f"  {folder}: {which} = rotor {target} "
          f"(z={zs[target]:+.5f} m), {n_blades} blades, {n_rev} steps/rev")
    return r, dTdr, dQdr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", action="append", required=True, metavar="LABEL=POSTPRO_DIR")
    ap.add_argument("--rpm", type=float, default=5000.0)
    ap.add_argument("--R", type=float, default=0.1143, help="rotor radius [m]")
    ap.add_argument("--rotor", default="downstream", choices=["downstream", "upstream"])
    ap.add_argument("--out", default="radial_diag.png")
    args = ap.parse_args()

    results = []
    for spec in args.case:
        if "=" not in spec:
            sys.exit(f"--case needs LABEL=PATH, got {spec}")
        label, path = spec.split("=", 1)
        if not Path(path).exists():
            sys.exit(f"not found: {path}")
        results.append((label.strip(), *analyse(path, args.rpm, args.rotor)))

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(results)))

    for (label, r, dT, dQ), c in zip(results, colors):
        x = r / args.R
        ax[0][0].plot(x, dT, "-", color=c, lw=1.8, label=label)
        ax[0][1].plot(x, dQ, "-", color=c, lw=1.8, label=label)
        cum = np.array([TRAPZ(dQ[:i + 1], r[:i + 1]) for i in range(len(r))])
        ax[1][0].plot(x, cum / cum[-1] * 100, "-", color=c, lw=1.8, label=label)

    ax[0][0].set_title(f"Thrust distribution  dT/dr  ({args.rotor} rotor)")
    ax[0][0].set_ylabel("dT/dr  (N/m)")
    ax[0][1].set_title(f"Torque distribution  dQ/dr  ({args.rotor} rotor)")
    ax[0][1].set_ylabel("dQ/dr  (Nm/m)")
    ax[1][0].set_title("Cumulative torque, % of blade total")
    ax[1][0].set_ylabel("cumulative Q (%)")

    # the money panel: where does the change between first and last case live?
    if len(results) >= 2:
        (l0, r0, T0, Q0), (l1, r1, T1, Q1) = results[0], results[-1]
        if len(r0) == len(r1):
            ax[1][1].plot(r1 / args.R, Q1 - Q0, "-", color="#B3242B", lw=2.0,
                          label=f"d(dQ/dr): {l1} - {l0}")
            ax[1][1].axhline(0, color="k", lw=0.8)
            ax[1][1].fill_between(r1 / args.R, 0, Q1 - Q0,
                                  where=(Q1 - Q0) > 0, alpha=0.25, color="#B3242B")
            ax[1][1].fill_between(r1 / args.R, 0, Q1 - Q0,
                                  where=(Q1 - Q0) < 0, alpha=0.25, color="#2B5FB3")
            net = TRAPZ(Q1 - Q0, r1)
            ax[1][1].set_title(f"CHANGE in torque distribution   (net {net:+.5f} Nm/blade-pair)")
            ax[1][1].set_ylabel("d(dQ/dr)  (Nm/m)")
        else:
            ax[1][1].text(0.5, 0.5, "cases have different spanwise meshes\n"
                                    "- cannot difference directly",
                          ha="center", va="center", transform=ax[1][1].transAxes)

    for a in ax.flat:
        a.set_xlabel("r / R")
        a.grid(True, ls=":", lw=0.6, alpha=0.6)
        a.axvspan(0.85, 1.0, color="orange", alpha=0.07)
        a.legend(fontsize=8)

    fig.suptitle("Radial load distribution - shaded band = outboard 0.85-1.0 R", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"\nWrote {args.out}")

    # numeric summary: how much of the change sits outboard of 0.85R
    if len(results) >= 2 and len(results[0][1]) == len(results[-1][1]):
        r = results[-1][1]
        d = results[-1][3] - results[0][3]
        tip = r / args.R >= 0.85
        total = TRAPZ(d, r)
        outboard = TRAPZ(d[tip], r[tip])
        print(f"\n  total torque change     : {total:+.6f} Nm")
        print(f"  from outboard of 0.85R  : {outboard:+.6f} Nm "
              f"({100 * outboard / total:.0f}% of it)" if total else "")


if __name__ == "__main__":
    main()
