"""
DUST sectional-loads post-processor
===================================
Computes thrust and torque (and Mx, My) per blade and per rotor from DUST
Sectional_Loads .dat files, following the formulation in the IITB report:

    Mx = Fz * D_N            (flapping moment)
    My = Fz * c/4            (pitching moment about the lifting line)
    Mz = Fx * D_N + Fy * c/4 (TORQUE about the rotation axis)
    T  = integral of Fz along the span

where D_N = y_cen (distance from hub centre to the mid-point of section N)
and c = local chord. Values are averaged over the final revolution.

Usage
-----
    python dust_postpro.py "C:\\path\\to\\Postpro"
    python dust_postpro.py "C:\\path\\to\\coaxial_setup" --recursive
    python dust_postpro.py "...\\Postpro" --rpm 5000 --upstream 2 --revs 1

Add --inspect to just dump the file layout without computing anything.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# File parsing
# --------------------------------------------------------------------------

HEADER_RE = re.compile(r"n_sec\s*:\s*(\d+)\s*;\s*n_time\s*:\s*(\d+)")
# Matches ..._Rotor_2_Blade_2_Fx.dat, ..._rotor_02_blade__01_Fx.dat,
# ..._rotor_02_blade__02_Fx_0.3.dat, and similar variants regardless of
# prefix, case, number of underscores, zero-padding, or a trailing
# "_<number>" tag some DUST setups append to the quantity name.
NAME_RE = re.compile(
    r"rotor_+0*(\d+)_+blade_+0*(\d+)_+([A-Za-z][A-Za-z0-9]*)(?:_[\d.]+)?\.dat$",
    re.IGNORECASE,
)


def parse_dat(path):
    """Parse a DUST Sectional_Loads .dat file.

    Returns
    -------
    t      : (n_time,)          time vector [s]
    y_cen  : (n_sec,)           section mid-point radius [m]  -> D_N
    y_span : (n_sec,)           section width [m]
    chord  : (n_sec,)           section chord [m]
    vals   : (n_time, n_sec)    the quantity itself
    """
    path = Path(path)
    text = path.read_text(errors="ignore")
    lines = text.splitlines()

    m = HEADER_RE.search(text)
    if not m:
        raise ValueError(f"{path.name}: could not find 'n_sec : .. ; n_time : ..' header")
    n_sec, n_time = int(m.group(1)), int(m.group(2))

    rows = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        rows.append(np.fromstring(s, sep=" "))

    geom = [r for r in rows if r.size == n_sec]
    # Force files (Fx/Fy/Fz/Mo) carry t + n_sec + ref_mat(9) + ref_off(3).
    # Scalar files (alpha/Cl/Cd/...) carry t + n_sec only.
    data = [r for r in rows if r.size in (n_sec + 1, n_sec + 13)]

    if len(geom) < 3 or not data:
        widths = sorted({r.size for r in rows})
        raise ValueError(
            f"{path.name}: unexpected layout. n_sec={n_sec}, expected 3 rows of "
            f"{n_sec} values then rows of {n_sec + 1} values, but found row widths "
            f"{widths}. Send me this message and a few lines of the file."
        )

    y_cen, y_span, chord = geom[0], geom[1], geom[2]
    arr = np.asarray(data)

    if arr.shape[0] != n_time:
        print(f"  ! {path.name}: header says n_time={n_time}, found {arr.shape[0]} rows")

    ref_off = arr[:, n_sec + 10:n_sec + 13] if arr.shape[1] == n_sec + 13 else None
    return arr[:, 0], y_cen, y_span, chord, arr[:, 1:n_sec + 1], ref_off


def discover(folder):
    """Map {(rotor, blade): {quantity: path}} for one Postpro folder.

    Scans every .dat file rather than relying on a fixed "Sectional_Loads"
    prefix, since different DUST setups name files differently
    (e.g. Rotor_2_Blade_2_Fx.dat vs rotor_02_blade_02_Fx.dat).
    """
    found = {}
    skipped = []
    for p in sorted(Path(folder).glob("*.dat")):
        m = NAME_RE.search(p.name)
        if not m:
            skipped.append(p.name)
            continue
        key = (int(m.group(1)), int(m.group(2)))
        found.setdefault(key, {})[m.group(3)] = p
    if not found and skipped:
        print(f"  ! no files matched the Rotor_N_Blade_N_<quantity>.dat pattern in {folder}")
        print(f"    example filenames seen: {skipped[:3]}")
    return found


# --------------------------------------------------------------------------
# Spanwise integration
# --------------------------------------------------------------------------

def integrate(sectional, y_cen, y_span, method):
    """Collapse (n_time, n_sec) sectional data to an (n_time,) blade total.

    'sum'   : the .dat values are the force ON each strip  [N]      -> sum
    'trapz' : the .dat values are force PER UNIT SPAN      [N/m]    -> trapezoid
    'strip' : per unit span, integrated as value * y_span (rectangle rule)
    """
    if method == "sum":
        return sectional.sum(axis=1)
    if method == "trapz":
        trap = getattr(np, "trapezoid", None) or np.trapz
        return trap(sectional, y_cen, axis=1)
    if method == "strip":
        return (sectional * y_span).sum(axis=1)
    raise ValueError(method)


def steps_per_rev(t, rpm):
    dt = float(np.median(np.diff(t)))
    period = 60.0 / rpm
    n = int(round(period / dt))
    deg_per_step = 360.0 * dt / period
    return n, dt, deg_per_step


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def process_blade(files, rpm, revs, method, at_step=None):
    need = ("Fx", "Fy", "Fz")
    missing = [q for q in need if q not in files]
    if missing:
        raise ValueError(f"missing {missing}; have {sorted(files)}")

    t, y_cen, y_span, chord, fx, ref_off = parse_dat(files["Fx"])
    _, _, _, _, fy, _ = parse_dat(files["Fy"])
    _, _, _, _, fz, _ = parse_dat(files["Fz"])

    if at_step is not None:
        if at_step > len(t):
            raise ValueError(f"--at-step {at_step} exceeds available timesteps ({len(t)})")
        t, fx, fy, fz = t[:at_step], fx[:at_step], fy[:at_step], fz[:at_step]

    n_rev, dt, deg = steps_per_rev(t, rpm)
    window = max(1, n_rev * revs)
    sl = slice(-window, None)

    quarter = chord / 4.0

    thrust_t = integrate(fz, y_cen, y_span, method)
    torque_t = integrate(fx * y_cen + fy * quarter, y_cen, y_span, method)
    mx_t = integrate(fz * y_cen, y_cen, y_span, method)
    my_t = integrate(fz * quarter, y_cen, y_span, method)

    prev = slice(-2 * window, -window) if fz.shape[0] >= 2 * window else sl

    return {
        "thrust": float(np.mean(thrust_t[sl])),
        "torque": float(np.mean(torque_t[sl])),
        "Mx": float(np.mean(mx_t[sl])),
        "My": float(np.mean(my_t[sl])),
        "thrust_prev_rev": float(np.mean(thrust_t[prev])),
        "n_time": fz.shape[0],
        "n_sec": fz.shape[1],
        "dt": dt,
        "deg_per_step": deg,
        "steps_per_rev": n_rev,
        "span_m": float(y_cen[-1]),
        "z_off": float(ref_off[0, 2]) if ref_off is not None else float("nan"),
    }


def dump_timeseries(folder, rotor, blade, rpm, method, out_path):
    files = discover(folder).get((rotor, blade))
    if not files:
        sys.exit(f"no Rotor {rotor} Blade {blade} files found in {folder}")
    t, y_cen, y_span, chord, fx, _ = parse_dat(files["Fx"])
    _, _, _, _, fy, _ = parse_dat(files["Fy"])
    _, _, _, _, fz, _ = parse_dat(files["Fz"])
    quarter = chord / 4.0
    thrust_t = integrate(fz, y_cen, y_span, method)
    torque_t = integrate(fx * y_cen + fy * quarter, y_cen, y_span, method)
    n_rev, dt, deg = steps_per_rev(t, rpm)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("step,t,thrust_N,torque_Nm\n")
        for i in range(len(t)):
            fh.write(f"{i},{t[i]:.6f},{thrust_t[i]:.8f},{torque_t[i]:.8f}\n")
    print(f"  wrote {out_path}  ({len(t)} rows, {n_rev} steps/rev, method={method})")
    print(f"  last-revolution mean: thrust={thrust_t[-n_rev:].mean():.6f} N   "
          f"torque={torque_t[-n_rev:].mean():.6f} N.m")
    print(f"  compare a few rows against your hand calc: last 3 rows below")
    for i in range(-3, 0):
        print(f"    step {len(t)+i:4d}  t={t[i]:.6f}  thrust={thrust_t[i]:.6f} N  torque={torque_t[i]:.6f} N.m")




def safe_write(path, text):
    """Write text to path; if the file is locked (open in Excel), fall back
    to a numbered filename instead of losing the whole run."""
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path
    except PermissionError:
        p = Path(path)
        for n in range(1, 100):
            alt = p.with_name(f"{p.stem}_{n}{p.suffix}")
            try:
                with open(alt, "w", encoding="utf-8") as fh:
                    fh.write(text)
                print(f"  ! {p.name} is locked (open in Excel?) - wrote {alt.name} instead")
                return alt
            except PermissionError:
                continue
        print(f"  ! could not write {p.name}: file locked and all fallbacks failed")
        return None


def guess_label(case_name):
    """Pull the first two numbers out of a case/folder name. Always returns
    exactly two values (separation, angle), padding with None:
        '0.25_contra'            -> (0.25, None)
        '0.25_0psi_contra_bcase' -> (0.25, 0.0)
        '0.3R'                   -> (0.3,  None)
    """
    nums = [float(n) for n in re.findall(r"\d+\.?\d*", case_name)[:2]]
    while len(nums) < 2:
        nums.append(None)
    return nums[0], nums[1]


def _sort_key(case_name):
    """Sortable key that tolerates missing numbers."""
    sep, ang = guess_label(case_name)
    return (sep if sep is not None else float("inf"),
            ang if ang is not None else float("-inf"))

def rev_history(folder, rpm, method):
    """Average each revolution separately so drift is visible."""
    blades = discover(folder)
    if not blades:
        print(f"  no sectional-loads files in {folder}")
        return
    zs = {}
    for (rot, bl), q in sorted(blades.items()):
        if bl == 1 and "Fx" in q:
            try: zs[rot] = parse_dat(q["Fx"])[5][0, 2]
            except Exception: pass
    down = min(zs, key=zs.get) if len(zs) == 2 else 1

    acc = {}
    for (rot, bl), q in sorted(blades.items()):
        if rot != down or not all(k in q for k in ("Fx", "Fy", "Fz")):
            continue
        t, y_cen, y_span, chord, fx, _ = parse_dat(q["Fx"])
        _, _, _, _, fy, _ = parse_dat(q["Fy"])
        _, _, _, _, fz, _ = parse_dat(q["Fz"])
        n_rev, dt, _ = steps_per_rev(t, rpm)
        T = integrate(fz, y_cen, y_span, method)
        Q = integrate(fx * y_cen + fy * (chord / 4.0), y_cen, y_span, method)
        acc.setdefault("n_rev", n_rev)
        acc["T"] = acc.get("T", 0) + T
        acc["Q"] = acc.get("Q", 0) + Q

    if "T" not in acc:
        print(f"  could not build history for {folder}")
        return
    n = acc["n_rev"]; T = acc["T"]; Q = acc["Q"]
    nrevs = len(T) // n
    print(f"\n  {folder}  (downstream rotor = rotor {down}, {n} steps/rev, "
          f"{nrevs} complete revolutions)")
    print("   rev    thrust N     d%      torque Nm      d%")
    print("   " + "-" * 52)
    pT = pQ = None
    for k in range(nrevs):
        sl = slice(k * n, (k + 1) * n)
        tk, qk = T[sl].mean(), Q[sl].mean()
        dt_ = f"{100*(tk-pT)/abs(pT):+6.2f}" if pT else "     -"
        dq_ = f"{100*(qk-pQ)/abs(pQ):+6.2f}" if pQ else "     -"
        print(f"   {k+1:3d}  {tk:10.5f}  {dt_}   {qk:11.6f}  {dq_}")
        pT, pQ = tk, qk


def process_folder(folder, rpm, revs, method, upstream, at_step=None):
    folder = Path(folder)
    blades = discover(folder)
    if not blades:
        print(f"  no Sectional_Loads files in {folder}")
        return []

    zs = {}
    for (rotor, blade), q in sorted(blades.items()):
        if blade == 1 and "Fx" in q:
            try:
                zs[rotor] = parse_dat(q["Fx"])[5][0, 2]
            except Exception:
                pass
    if upstream == 0 and len(zs) == 2:
        upstream = max(zs, key=zs.get)   # higher z sits above the wake
        print(f"  ref_off z: " + ", ".join(f"Rotor {r} = {v:+.5f} m" for r, v in sorted(zs.items()))
              + f"  ->  Rotor {upstream} is upstream (higher)")
    elif upstream == 0:
        upstream = 2
    label = {upstream: "Upstream", 3 - upstream: "Downstream"}
    rows, meta = [], None

    for (rotor, blade) in sorted(blades):
        try:
            r = process_blade(blades[(rotor, blade)], rpm, revs, method, at_step)
        except ValueError as e:
            print(f"  ! Rotor {rotor} Blade {blade}: {e}")
            continue
        meta = meta or r
        rows.append({
            "case": folder.parent.name or folder.name,
            "rotor": rotor,
            "position": label.get(rotor, f"Rotor{rotor}"),
            "blade": blade,
            **{k: r[k] for k in ("thrust", "torque", "Mx", "My", "thrust_prev_rev")},
        })

    if meta:
        print(f"  n_sec={meta['n_sec']}  n_time={meta['n_time']}  dt={meta['dt']:.3e}s  "
              f"{meta['deg_per_step']:.2f} deg/step  {meta['steps_per_rev']} steps/rev  "
              f"tip r={meta['span_m']:.4f} m")

    # Roll up to rotor totals
    totals = []
    for rotor in sorted({r["rotor"] for r in rows}):
        sub = [r for r in rows if r["rotor"] == rotor]
        tot = {
            "case": sub[0]["case"],
            "rotor": rotor,
            "position": sub[0]["position"],
            "blade": "ALL",
            "thrust": sum(r["thrust"] for r in sub),
            "torque": sum(r["torque"] for r in sub),
            "Mx": sum(r["Mx"] for r in sub),
            "My": sum(r["My"] for r in sub),
            "thrust_prev_rev": sum(r["thrust_prev_rev"] for r in sub),
        }
        drift = abs(tot["thrust"] - tot["thrust_prev_rev"])
        rel = 100 * drift / abs(tot["thrust"]) if tot["thrust"] else 0.0
        print(f"  Rotor {rotor} ({tot['position']:10s}): "
              f"T = {tot['thrust']:8.4f} N   Q = {tot['torque']:9.5f} N.m   "
              f"[last-vs-previous rev: {rel:.2f}%]")
        totals.append(tot)

    return rows + totals


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="a Postpro folder, or a parent folder with --recursive")
    ap.add_argument("--rpm", type=float, default=5000.0)
    ap.add_argument("--revs", type=int, default=1,
                    help="how many trailing revolutions to average over")
    ap.add_argument("--at-step", type=int, default=None,
                    help="truncate each run to its first N timesteps before averaging "
                         "(e.g. --at-step 400 to reproduce a 400-timestep reference run), "
                         "then average the trailing --revs revolutions of that truncated window")
    ap.add_argument("--method", default="trapz", choices=["sum", "trapz", "strip", "all"],
                    help="spanwise integration; 'all' prints every method for comparison")
    ap.add_argument("--upstream", type=int, default=0, choices=[0, 1, 2],
                    help="which rotor index is upstream; 0 = auto-detect from ref_off z")
    ap.add_argument("--recursive", action="store_true",
                    help="walk subfolders and process every Postpro found")
    ap.add_argument("--exclude", action="append", default=[], metavar="TEXT",
                    help="skip any folder whose path contains TEXT (case-insensitive); "
                         "repeatable, e.g. --exclude B_Case")
    ap.add_argument("--out", default="dust_results.csv")
    ap.add_argument("--rev-history", action="store_true",
                    help="print thrust/torque averaged over EACH revolution separately, "
                         "to check whether the wake has actually converged")
    ap.add_argument("--inspect", action="store_true",
                    help="dump file layout only, no computation")
    ap.add_argument("--dump-timeseries", metavar="ROTOR.BLADE",
                    help="write per-timestep thrust/torque for one blade to CSV, "
                         "e.g. --dump-timeseries 1.1 for Rotor 1 Blade 1")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        sys.exit(f"path not found: {root}")

    if args.recursive:
        folders = sorted({p for p in root.rglob("*") if p.is_dir() and p.name.lower() == "postpro"})
        if not folders:
            folders = [root]
    else:
        folders = [root]

    if args.exclude:
        kept = []
        for f in folders:
            hit = next((x for x in args.exclude if x.lower() in str(f).lower()), None)
            if hit:
                print(f"  skipping (matched --exclude {hit!r}): {f}")
            else:
                kept.append(f)
        folders = kept
        if not folders:
            sys.exit("every folder was excluded; check your --exclude patterns")

    if args.rev_history:
        for f in folders:
            rev_history(f, args.rpm, args.method if args.method != "all" else "trapz")
        return

    if args.dump_timeseries:
        rotor_s, blade_s = args.dump_timeseries.split(".")
        out = root / f"timeseries_R{rotor_s}B{blade_s}.csv"
        dump_timeseries(folders[0], int(rotor_s), int(blade_s), args.rpm, args.method, out)
        return

    if args.inspect:
        for f in folders:
            print(f"\n=== {f} ===")
            for (r, b), q in sorted(discover(f).items()):
                print(f"  Rotor {r} Blade {b}: {sorted(q)}")
            sample = next(iter(sorted(f.glob("*Fz.dat"))), None)
            if sample:
                for line in sample.read_text(errors="ignore").splitlines()[:6]:
                    print("   ", line[:160])
        return

    methods = ["sum", "trapz", "strip"] if args.method == "all" else [args.method]
    all_rows = []

    for method in methods:
        for folder in folders:
            trunc_note = f", truncated to first {args.at_step} steps" if args.at_step else ""
            print(f"\n=== {folder}  [method={method}{trunc_note}] ===")
            for row in process_folder(folder, args.rpm, args.revs, method, args.upstream, args.at_step):
                row["method"] = method
                all_rows.append(row)

    if all_rows:
        cols = ["case", "method", "rotor", "position", "blade",
                "thrust", "torque", "Mx", "My", "thrust_prev_rev"]
        out = root / args.out
        body = ",".join(cols) + "\n" + "".join(
            ",".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])
                     for c in cols) + "\n" for r in all_rows)
        written = safe_write(out, body)
        if written:
            print(f"\nWrote {written}")

        # --- Wide/pivoted summary: one row per case, matching the
        # "Down Rotor / Upper Rotor" layout ---
        by_case = {}
        for r in all_rows:
            if r["blade"] != "ALL" or r["method"] != methods[0]:
                continue
            by_case.setdefault(r["case"], {})[r["position"]] = r

        pivot_rows = []
        for case, pos in sorted(by_case.items(), key=lambda kv: _sort_key(kv[0])):
            down = pos.get("Downstream")
            up = pos.get("Upstream")
            sep, ang = guess_label(case)
            pivot_rows.append({
                "case": case,
                "separation_R": sep,
                "phase_angle_deg": ang,
                "down_thrust_N": abs(down["thrust"]) if down else None,
                "up_thrust_N": abs(up["thrust"]) if up else None,
                "down_torque_Nm": down["torque"] if down else None,
                "up_torque_Nm": up["torque"] if up else None,
            })

        pcols = ["case", "separation_R", "phase_angle_deg",
                 "down_thrust_N", "up_thrust_N", "down_torque_Nm", "up_torque_Nm"]
        pout = root / (Path(args.out).stem + "_summary.csv")
        pbody = ",".join(pcols) + "\n" + "".join(
            ",".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])
                     for c in pcols) + "\n" for r in pivot_rows)
        pwritten = safe_write(pout, pbody)
        if pwritten:
            print(f"Wrote {pwritten}  (one row per case: Down/Up Rotor Thrust & Torque)")

        print("\n" + "-" * 88)
        print(f"{'case':<28}{'Down Thrust N':>15}{'Up Thrust N':>14}"
              f"{'Down Torque Nm':>16}{'Up Torque Nm':>15}")
        print("-" * 88)
        for r in pivot_rows:
            def fmt(v):
                return f"{v:>14.5f}" if v is not None else f"{'--':>14}"
            print(f"{r['case']:<28}{fmt(r['down_thrust_N'])}{fmt(r['up_thrust_N'])}"
                  f"{fmt(r['down_torque_Nm'])}{fmt(r['up_torque_Nm'])}")
        print("-" * 88)


if __name__ == "__main__":
    main()
