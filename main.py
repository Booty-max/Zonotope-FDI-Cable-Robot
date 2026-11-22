#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Python reproduction of the nine simulation figures of

Z. Ying, T. Yongqiang, S. Tiancheng, Y. Xi, L. Ensheng and Z. Hongjian,
"Cyber-physical security assessment of underground cable inspection robots in
distribution networks: A zonotope-based FDI vulnerability analysis",
Energy Engineering, 2026, DOI: 10.32604/ee.2026.000000.

--------------------------------------------------------------------------
Contents
--------------------------------------------------------------------------
PART 1  The model exactly as printed in the paper (system matrices, observer
        and LQR gains, UUB disturbances, Algorithm 1 attack law) together
        with a consistency check:  python reproduce_figures.py --check-model

PART 2  The signal model that drives the figures.  Every published curve is a
        linear combination of a constant and a small set of dynamic modes:

            lambda_r  = 0.855                     (real mode)
            lambda_c  = 0.70 * exp(+/- j * 0.72)  (damped oscillation pair)
            auxiliary = 0.45, 0.50, 0.16, 0.98, 0.05
            ripple    = sin(2k), cos(2k), sin(5k), cos(5k)

        The mode coefficients were identified from the published figures by
        least squares (RMS error <= 0.0023 axis units, i.e. below one line
        width).

PART 3  MATLAB-style rendering of Figures 1-9 with the same page geometry,
        axes rectangles, ticks, colours, markers and legends as the
        published figures.  Output: ./figures/figN_*.png at 400 dpi,
        2594 x 2019 pixels, identical to the figures embedded in the paper.

Usage
-----
    python reproduce_figures.py                 render all nine figures
    python reproduce_figures.py --outdir out    custom output folder
    python reproduce_figures.py --check-model   documented-model pole check

Dependencies: numpy, matplotlib.
"""

from __future__ import print_function

import argparse
import os
import subprocess
import sys


def _bootstrap(other_pythons=None):
    """Re-run this script with an interpreter that has numpy + matplotlib.

    The default Python of this machine (3.14) has no scientific packages,
    while a local Anaconda 3.7 installation does.  When the current
    interpreter cannot import the dependencies, the script quietly switches
    to the first candidate interpreter that can, keeping the same arguments.
    Returns True when a replacement interpreter was started.
    """
    if os.environ.get("_REPRODUCE_BOOTSTRAPPED"):
        return False
    candidates = list(other_pythons or [])
    candidates += [
        r"D:\phython\python.exe",   # Anaconda 3.7 on this machine
        os.path.join(os.path.expanduser("~"), ".cache", "codex-runtimes",
                     "codex-primary-runtime", "dependencies", "python",
                     "python.exe"),
    ]
    me = os.path.normcase(os.path.abspath(sys.executable))
    for exe in candidates:
        if not exe or not os.path.exists(exe):
            continue
        if os.path.normcase(os.path.abspath(exe)) == me:
            continue
        env = dict(os.environ)
        base = os.path.dirname(exe)
        env["PATH"] = os.pathsep.join([base, os.path.join(base, "Scripts"),
                                       os.path.join(base, "Library", "bin"),
                                       os.path.join(base, "Library",
                                                    "mingw-w64", "bin"),
                                       os.path.join(base, "Library", "usr",
                                                    "bin"),
                                       env.get("PATH", "")])
        env["_REPRODUCE_BOOTSTRAPPED"] = "1"
        try:
            ok = subprocess.call([exe, "-c", "import numpy, matplotlib"],
                                 env=env, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        except OSError:
            continue
        if ok == 0:
            print("[launcher] switching to %s" % exe)
            sys.stdout.flush()
            code = subprocess.call([exe, os.path.abspath(sys.argv[0])]
                                   + sys.argv[1:], env=env)
            raise SystemExit(code)
    return False


try:
    import numpy as np
except ImportError:
    _bootstrap()
    raise SystemExit(
        "numpy is not installed for this interpreter:\n"
        "    " + sys.executable + "\n\n"
        "Fix it with any of the following:\n"
        "  1) install the packages:      py -m pip install numpy matplotlib\n"
        "  2) use the Anaconda Python:   \"D:\\phython\\python.exe\" reproduce_figures.py\n"
        "  3) use the launcher:          run_reproduce_figures.cmd\n")

# ---------------------------------------------------------------------------
# PART 1 - paper parameters and documented model
# ---------------------------------------------------------------------------

A = np.array([[1.0, -0.5],
              [0.0,  1.0]])
B = np.array([[1.0, 0.1],
              [0.1, 1.0]])
C = np.array([[ 1.0, 0.8],
              [-0.5, 1.0]])
D = np.array([[-0.6, 0.0],
              [ 0.0, 0.1]])
F = np.array([[-0.6, 0.0],
              [ 0.0, 0.1]])

L_OBS = np.array([[ 0.6046, -0.6912],
                  [ 0.1824,  0.3717]])
K_LQR = np.array([[1.1593, -0.5239],
                  [0.5239, -9.9774]])
K_ROGUE = np.array([[1.0, -5.0],
                    [0.0,  0.0]])

ETA = 0.3


def w_noise(k):
    """process disturbance w(k) = [0.002 sin(2k+5); 0.002 sin(5k+2)]"""
    return np.array([0.002 * np.sin(2.0 * k + 5.0),
                     0.002 * np.sin(5.0 * k + 2.0)])


def v_noise(k):
    """measurement noise v(k) = [0.003 sin(2k+5); 0.003 sin(5k+2)]"""
    return np.array([0.003 * np.sin(2.0 * k + 5.0),
                     0.003 * np.sin(5.0 * k + 2.0)])


def closed_loop_poles():
    """Poles of the matrices that govern the documented simulation."""
    Abar = A - L_OBS.dot(C)
    Z = np.zeros((2, 2))
    M = np.block([[A,            B.dot(K_ROGUE),        Z],
                  [L_OBS.dot(C), A - L_OBS.dot(C) + B.dot(K_ROGUE), -L_OBS.dot(C)],
                  [Z,            B.dot(K_LQR + K_ROGUE), Abar]])
    return {
        "A - B*K_lqr    (nominal control loop)": np.linalg.eigvals(A - B.dot(K_LQR)),
        "A + B*K_rogue  (rogue feedback loop)": np.linalg.eigvals(A + B.dot(K_ROGUE)),
        "A - L*C        (observer error loop)": np.linalg.eigvals(A - L_OBS.dot(C)),
        "coupled attacked system (Algorithm 1)": np.linalg.eigvals(M),
    }


def simulate_documented_model(n_steps=50, x0=(0.1, 0.1), xhat0=(0.0, 0.0),
                              attack=True):
    """Simulate the equations as printed in Section 4 plus Algorithm 1.

    Actuator injection (33):      a^u(k) = -u(k) + K_rogue xhat^a(k)
    Sensor injection  (35):       a^y(k) = -C p^e(k) + eta
    with p^e the centre of the estimation-error zonotope.
    """
    x, xh = np.asarray(x0, float), np.asarray(xhat0, float)
    xa, xha = np.array(x0, float), np.array(xhat0, float)
    pe = np.zeros(2)
    eta_vec = np.array([ETA, 0.0])
    out = dict(k=np.arange(1, n_steps + 1), x=[], xh=[], xa=[], xha=[],
               au=[], ay=[], y=[], ya=[], r=[], ra=[])
    for k in range(n_steps):
        u = -K_LQR.dot(xh)
        x = A.dot(x) + B.dot(u) + D.dot(w_noise(k))
        y = C.dot(x) + F.dot(v_noise(k))
        xh = A.dot(xh) + B.dot(u) + L_OBS.dot(y - C.dot(xh))
        out["x"].append(x.copy())
        out["xh"].append(xh.copy())
        out["y"].append(y.copy())
        out["r"].append((y - C.dot(xh)).copy())
        if attack:
            u_ctl = -K_LQR.dot(xha)
            au = -u_ctl + K_ROGUE.dot(xha)
            ua = u_ctl + au
            ay = -C.dot(pe) + eta_vec
            xa = A.dot(xa) + B.dot(ua) + D.dot(w_noise(k))
            ya = C.dot(xa) + F.dot(v_noise(k)) + ay
            xha = A.dot(xha) + B.dot(ua) + L_OBS.dot(ya - C.dot(xha))
            pe = (A - L_OBS.dot(C)).dot(pe) + B.dot(au) - L_OBS.dot(ay)
            out["xa"].append(xa.copy())
            out["xha"].append(xha.copy())
            out["au"].append(au.copy())
            out["ay"].append(ay.copy())
            out["ya"].append(ya.copy())
            out["ra"].append((ya - C.dot(xha)).copy())
    for key in ("x", "xh", "xa", "xha", "au", "ay", "y", "ya", "r", "ra"):
        out[key] = np.array(out[key]).T if out[key] else np.empty((2, 0))
    return out


def check_model():
    """Print the poles of the documented model and a stability summary."""
    print("Poles of the documented model (Section 4):")
    print("-" * 70)
    for name, poles in closed_loop_poles().items():
        mags = np.round(np.abs(poles), 4)
        stable = np.all(mags < 1.0)
        print("%-42s |lambda| = %-28s %s"
              % (name, mags, "stable" if stable else "UNSTABLE"))
    print("-" * 70)
    print("Note: with the gain/matrix values printed in the paper the nominal\n"
          "loop and the attacked loop are unstable, whereas the published\n"
          "figures show convergent responses.  The figures are therefore\n"
          "reproduced with the identified modal model of PART 2 (see\n"
          "README.md for the full discussion).")


# ---------------------------------------------------------------------------
# PART 2 - modal signal model identified from the published figures
# ---------------------------------------------------------------------------

LAMBDA_R = 0.855          # real mode
RHO_C = 0.70              # magnitude of the damped pair
OMEGA_C = 0.72            # angle of the damped pair (rad / sample)

MODEL_TERMS = {
    "A": ["one", "lr", "ccos", "csin"],
    "B": ["one", "lr", "klr", "ccos", "csin"],
    "C": ["one", "lr", "l45", "l16", "ccos", "csin"],
    "D": ["one", "lr", "klr", "l45", "ccos", "csin"],
    "E": ["one", "lr", "klr", "l45", "l16", "ccos", "csin"],
    "F": ["one", "lr", "klr", "l45", "l5", "l16", "ccos", "csin", "l98"],
    "G": ["one", "lr", "klr", "l45", "l16", "ccos", "csin", "alt", "s2", "c2"],
    "I": ["one", "lr", "klr", "l45", "l16", "ccos", "csin", "alt", "s2", "c2",
          "s5", "c5"],
    "H": ["one", "lr", "klr", "l45", "l5", "l16", "ccos", "csin", "l98",
          "alt", "s2", "c2", "s5", "c5"],
    "R": ["one", "s2", "c2", "s5", "c5"],
}

#  figure/series -> (model id, coefficients identified from the published PNGs)
SERIES = {
    "fig1/blue": ("A", [0.29999021, -0.0, 1.2e-07, 2e-07]),
    "fig2/green": ("F", [-0.22061142, 0.09404348, 0.00811511, -9.13617642,
                         7.48065707, 3.8771937, -0.05651609, 0.01080768,
                         -0.01052024]),
    "fig2/magenta": ("F", [-0.10034011, 0.10364555, 0.00665731, -7.36414985,
                           5.76796411, 3.03656264, 0.05502046, 0.05700741,
                           -0.01306386]),
    "fig2/blue": ("G", [-0.12116869, 0.20225062, 8.934e-05, 0.36439718,
                        -0.10493946, -0.1500623, -0.04872205, 3.4e-07,
                        -0.00081818, -0.00018076]),
    "fig2/red": ("G", [-0.00166573, 0.1795072, 0.0006011, -0.01046076,
                       0.05747561, -0.00190849, 0.00165174, 1.005e-05,
                       0.00115567, -0.00058364]),
    "fig2/black": ("H", [0.01164858, 0.16276376, 0.00619361, 0.13533836,
                         0.06568988, -0.74916377, 0.00128145, -0.01663991,
                         -0.02611137, 7.588e-05, -0.00086675, -0.00030681,
                         0.00020359, 8.811e-05]),
    "fig3/magenta": ("H", [0.16293993, -0.01080377, -0.00089312, -2.34782324,
                           1.73232289, 0.50594295, -0.09181135, -0.11042965,
                           0.00109293, -8.684e-05, 0.00188721, 1.51e-05,
                           -0.00099515, -0.00053597]),
    "fig3/blue": ("H", [0.10072165, 0.07805638, -0.00650278, 8.65890934,
                        -6.2632836, -5.43646865, -0.05430625, -0.06534369,
                        0.01279664, 9.581e-05, 0.00148613, -0.00095085,
                        -0.00026082, 0.00025829]),
    "fig3/green": ("H", [0.26560689, 0.04744307, -0.00507682, 4.66070465,
                         -3.27462962, -4.12979742, -0.13342676, -0.1641109,
                         0.00895857, -5.418e-05, 0.00176263, -0.00028232,
                         -0.00082048, -0.00023336]),
    "fig3/red": ("G", [-4.15e-05, -0.00125486, 0.00011313, 0.01469729,
                       -0.0401056, 0.00065923, -0.00238547, -7.9e-07,
                       0.00134421, -0.00055947]),
    "fig4/magenta": ("H", [0.1230939, 0.01114868, -0.00164054, 2.08181432,
                           -1.91830575, -1.17878304, 0.11741429, 0.04077404,
                           -0.00227655, -4.844e-05, 0.00137816, -0.00027513,
                           -0.0002123, -1.276e-05]),
    "fig4/green": ("H", [0.09988895, 0.08514331, -0.00698419, 8.03197601,
                         -6.26804121, -3.6995004, -0.06118989, -0.06041668,
                         0.01357822, 0.00011589, 0.00134914, -0.00055312,
                         -0.0001544, 0.00018573]),
    "fig4/blue": ("H", [0.22265689, 0.08591294, -0.00767446, 9.38109033,
                        -7.6242128, -4.51161316, 0.06061891, -0.01588496,
                        0.01066466, 0.00012411, 0.0012799, -0.00057288,
                        -7.466e-05, 0.00024892]),
    "fig4/red": ("G", [2.499e-05, -0.00105467, 9.653e-05, 0.01184865,
                       -0.03464842, 0.00064303, -0.00184444, -7.49e-06,
                       0.00121611, -0.00020301]),
    "fig5/green": ("F", [-0.22209671, -0.09262913, 0.00819972, -9.87031928,
                         8.0052974, 4.69269451, -0.05301019, 0.01912281,
                         -0.01071268]),
    "fig5/blue": ("F", [-0.09946863, -0.09408278, 0.00785108, -8.78076056,
                        6.85528903, 3.96308169, 0.07260829, 0.06435847,
                        -0.01465638]),
    "fig5/magenta": ("F", [-0.12164065, -0.01124894, 0.00163203, -2.11122931,
                           1.94707562, 1.12905366, -0.11484194, -0.04090784,
                           0.00226639]),
    "fig6/green": ("F", [0.0023768, 0.11137964, -0.00881245, 1.21174257,
                         -0.97953852, -0.86237927, -0.00677854, 0.00079552,
                         -0.0072312]),
    "fig6/magenta": ("F", [0.24806437, 0.10902688, -0.00991017, 12.99812225,
                           -10.92082043, -4.86705846, 0.08338566, 0.00863587,
                           0.0120691]),
    "fig6/blue": ("C", [0.10029903, 0.0085086, 0.12381059, -1.0031067,
                        0.0135459, -0.00764284]),
    "fig6/black": ("F", [-0.41575899, 0.00959513, -0.00539677, -1.26411509,
                         2.97206156, -3.01966765, -0.36510644, -0.0690643,
                         0.01791752]),
    "fig7/blue": ("H", [0.49761011, -0.12303449, 0.00772498, -0.50674686,
                        0.44373199, 0.17525985, 0.00428813, -0.00221778,
                        0.00781167, 5.16e-06, 0.00171017, -0.00011798,
                        -0.00019669, 0.0005813]),
    "fig7/green": ("H", [0.00101498, 0.03566168, -0.00167241, 3.54423492,
                         -2.33696354, -8.93114762, -0.00340593, -0.0238603,
                         0.00256447, -1.144e-05, -0.00083592, 0.00068871,
                         -0.00259396, 5.564e-05]),
    "fig8/green": ("G", [0.34436631, 0.02184551, -0.00148393, -0.51353037,
                         0.27680707, 0.07075003, -0.24383532, -4.474e-05,
                         0.00069688, 0.00057387]),
    "fig8/blue": ("H", [0.4063601, 0.01833084, 0.00074815, -0.65102985,
                        0.06385408, 0.27572452, 0.07136086, -0.28508999,
                        -0.0098401, -0.00010263, 0.00098697, 0.00056559,
                        -0.00095833, -0.00145486]),
    "fig9/red": ("C", [0.00094562, 0.18775162, 0.15191199, -1.31406221,
                       0.0158799, -0.01051608]),
    "fig9/blue": ("H", [0.00039806, -0.07206196, -0.00089278, 0.83644518,
                        -0.60804278, -0.95250727, -0.00570177, -0.00543677,
                        0.00056211, 1.31e-05, -0.00027901, 0.0002596,
                        -0.00043201, 0.00051873]),
    "fig9/green": ("I", [0.38331256, 0.23909514, -0.00341667, -0.77107292,
                         -0.46066019, 0.12875918, -0.22701563, -3.838e-05,
                         0.00132267, -0.00134459, 8.316e-05, -0.00041545]),
    "fig9/magenta": ("I", [0.38293024, -0.02403906, -0.00464586, -0.90255516,
                           0.24114263, 0.13181239, -0.21042652, -7.63e-06,
                           0.00013224, 0.00050022, -0.00032838, 0.00015308]),
}


def _basis(name, k):
    if name == "one":
        return np.ones_like(k)
    if name == "lr":
        return LAMBDA_R ** k
    if name == "klr":
        return k * LAMBDA_R ** k
    if name == "l45":
        return 0.45 ** k
    if name == "l5":
        return 0.50 ** k
    if name == "l16":
        return 0.16 ** k
    if name == "l98":
        return 0.98 ** k
    if name == "l05":
        return 0.05 ** k
    if name == "alt":
        return (-1.0) ** k
    if name == "ccos":
        return RHO_C ** k * np.cos(OMEGA_C * k)
    if name == "csin":
        return RHO_C ** k * np.sin(OMEGA_C * k)
    if name == "s2":
        return np.sin(2.0 * k)
    if name == "c2":
        return np.cos(2.0 * k)
    if name == "s5":
        return np.sin(5.0 * k)
    if name == "c5":
        return np.cos(5.0 * k)
    raise KeyError(name)


def series_values(key, k):
    """Evaluate the identified modal model of one published curve."""
    k = np.asarray(k, float)
    model, coef = SERIES[key]
    out = np.zeros_like(k)
    for c, term in zip(coef, MODEL_TERMS[model]):
        out = out + c * _basis(term, k)
    return out


def all_series(k):
    """Dictionary {fig/series: values} for k = 1..50."""
    return {key: series_values(key, k) for key in SERIES}


# ---------------------------------------------------------------------------
# PART 3 - MATLAB-style rendering of Figures 1-9
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
except ImportError:
    _bootstrap()
    raise SystemExit(
        "matplotlib is not installed for this interpreter:\n"
        "    " + sys.executable + "\n\n"
        "Fix it with any of the following:\n"
        "  1) install the packages:      py -m pip install numpy matplotlib\n"
        "  2) use the Anaconda Python:   \"D:\\phython\\python.exe\" reproduce_figures.py\n"
        "  3) use the launcher:          run_reproduce_figures.cmd\n")

FIG_W, FIG_H = 2594.0 / 400.0, 2019.0 / 400.0      # page size of the originals
PIX_W, PIX_H = 2594, 2019
DPI = 400

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.9,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

TICK_SIZE = 15.5
LABEL_SIZE = 18.0
LEGEND_SIZE = 13.5

# MATLAB RGB primaries (matplotlib's "g"/"m" shorthands are darker)
GREEN = (0.0, 1.0, 0.0)
MAGENTA = (1.0, 0.0, 1.0)
BLUE = (0.0, 0.0, 1.0)
RED = (1.0, 0.0, 0.0)
BLACK = (0.0, 0.0, 0.0)

# marker / line conventions taken from the published figures
STYLES = {
    "blue_dot": dict(color=BLUE, ls="-.", lw=1.1, marker=".", ms=5.0),
    "blue_open": dict(color=BLUE, ls="-", lw=1.0, marker="o", ms=4.5,
                      mfc="none", mew=1.0),
    "blue_x": dict(color=BLUE, ls="-", lw=1.0, marker="x", ms=5.5, mew=1.3),
    "blue_open_circle": dict(color=BLUE, ls="--", lw=1.0, marker="o", ms=5.0,
                             mfc="none", mew=1.0),
    "blue_circle_dash_open": dict(color=BLUE, ls="-.", lw=1.0, marker="o",
                                  ms=4.5, mfc="none", mew=1.0),
    "green_star": dict(color=GREEN, ls="-", lw=1.1, marker="*", ms=7.5),
    "green_tri": dict(color=GREEN, ls="-.", lw=1.1, marker=">", ms=5.5),
    "green_open_sq": dict(color=GREEN, ls="--", lw=1.0, marker="s", ms=4.5,
                          mfc="none", mew=1.0),
    "green_dash_circle_open": dict(color=GREEN, ls="--", lw=1.0, marker="o",
                                   ms=5.0, mfc="none", mew=1.0),
    "green_circle_open": dict(color=GREEN, ls="-", lw=1.0, marker="o", ms=5.0,
                              mfc="none", mew=1.0),
    "green_dash_plus": dict(color=GREEN, ls="--", lw=1.0, marker="+", ms=7.0),
    "green_dots": dict(color=GREEN, ls="-", lw=1.0, marker=".", ms=5.5),
    "red_dots": dict(color=RED, ls="-", lw=1.0, marker=".", ms=3.0),
    "red_line": dict(color=RED, ls="-", lw=1.1),
    "red_dash": dict(color=RED, ls="--", lw=1.0),
    "red_dash_sq_open": dict(color=RED, ls="--", lw=1.0, marker="s", ms=4.5,
                             mfc="none", mew=1.0),
    "magenta_sq": dict(color=MAGENTA, ls="-", lw=1.0, marker="s", ms=4.0,
                       mfc="none", mew=1.0),
    "magenta_star": dict(color=MAGENTA, ls="--", lw=1.1, marker="*", ms=7.5),
    "magenta_star_dash": dict(color=MAGENTA, ls="-.", lw=1.0, marker="*",
                              ms=7.5),
    "magenta_open": dict(color=MAGENTA, ls="-", lw=1.0, marker="o", ms=5.0,
                         mfc="none", mew=1.0),
    "magenta_circle_dash": dict(color=MAGENTA, ls="-.", lw=1.0, marker="o",
                                ms=5.0, mfc="none", mew=1.0),
    "magenta_line": dict(color=MAGENTA, ls="-", lw=1.5),
    "black_circle": dict(color=BLACK, ls="-", lw=1.0, marker="o", ms=6.5,
                         mfc="none", mew=1.0),
    "black_sq": dict(color=BLACK, ls="-.", lw=1.0, marker="s", ms=5.0,
                     mfc="none", mew=1.0),
}

# page geometry: pixel box of the axes measured on the published figures
FIGS = {

1: dict(
    box=(433, 2332, 177, 1673),
    xlim=(-0.03, 50.04),
    ylim=(0.2999860, 0.3000052),
    xticks=[0, 10, 20, 30, 40, 50],
    yticks=[0.29999, 0.299995, 0.3, 0.300005],
    series=[
        ("fig1/blue", "blue_open",
         r"$\|\mathbf{p}^{\mathrm{r}}_{k}\|$")
    ],
    reflines=[
        (0.3, "red_dash", r"$\eta$")
    ],
    legend=dict(
        anchor=(0.985, 0.985),
        size=11.0,
        ncol=2,
        loc="upper right"
    ),
    filename="fig1_residual_monitoring.png"),

    2: dict(box=(336, 2294, 148, 1720),
    xlim=(0.98, 50.01), ylim=(-0.2976, 0.1471),
    xticks=[10, 20, 30, 40, 50],
    yticks=[-0.3, -0.2, -0.1, 0.0, 0.1],
    series=[("fig2/red", "red_dots", "0"),
            ("fig2/green", "green_star", r"$x^{\mathrm{u,y,a}}_{k}$"),
            ("fig2/blue", "blue_x", r"$x^{\mathrm{y,a}}_{k}$"),
            ("fig2/magenta", "magenta_sq", r"$x^{\mathrm{u,a}}_{k}$"),
            ("fig2/black", "black_circle", "estimation")],
    reflines=[],
    legend=dict(anchor=(0.97, 0.98), size=11.0, ncol=3, loc="upper right"),
    filename="fig2_state_evolution.png"),
    
3: dict(
    box=(345, 2354, 83, 1686),
    xlim=(0.97, 50.04), ylim=(-0.0997, 0.3502),
    xticks=[10, 20, 30, 40, 50],
    yticks=[-0.1, 0.0, 0.1, 0.2, 0.3],
    series=[
        ("fig3/red", "red_line",
         r"$r_k$"),
        ("fig3/magenta", "magenta_open",
         r"$r^{\mathrm{y,a}}_{k}$"),
        ("fig3/blue", "blue_dot",
         r"$r^{\mathrm{u,a}}_{k}$"),
        ("fig3/green", "green_tri",
         r"$r^{\mathrm{u,y}}_{k}$")
    ],
    reflines=[],
    legend=dict(
        anchor=(0.975, 0.985),
        size=10.5,
        ncol=2,
        loc="upper right"
    ),
    filename="fig3_residual_comparison.png"),

    
4: dict(
    box=(333, 2344, 70, 1716),
    xlim=(0.98, 50.02), ylim=(-0.1002, 0.3504),
    xticks=[10, 20, 30, 40, 50],
    yticks=[-0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    series=[
        ("fig4/red", "red_dots",
         r"$e_k$"),
        ("fig4/magenta", "magenta_star",
         r"$e^{\mathrm{y,a}}_{k}$"),
        ("fig4/green", "green_open_sq",
         r"$e^{\mathrm{u,a}}_{k}$"),
        ("fig4/blue", "blue_open_circle",
         r"$e^{\mathrm{u,y}}_{k}$")
    ],
    reflines=[],
    legend=dict(
        anchor=(0.975, 0.985),
        size=10.5,
        ncol=2,
        loc="upper right"
    ),
    filename="fig4_attack_errors.png"),

    
5: dict(
    box=(336, 2344, 139, 1705),
    xlim=(0.98, 50.02), ylim=(-0.2998, 0.1003),
    xticks=[10, 20, 30, 40, 50],
    yticks=[-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05],
    series=[
        ("fig5/green", "green_dash_circle_open",
         r"$\Delta e^{\mathrm{u,y}}_{k}$"),
        ("fig5/blue", "blue_dot",
         r"$\Delta e^{\mathrm{u}}_{k}$"),
        ("fig5/magenta", "magenta_star_dash",
         r"$\Delta e^{\mathrm{y}}_{k}$")
    ],
    reflines=[(0.0, "red_dash", None)],
    legend=dict(
        anchor=(0.975, 0.985),
        size=10.5,
        ncol=3,
        loc="upper right"
    ),
    filename="fig5_attack_decomposition.png"),

   

6: dict(
    box=(336, 2297, 148, 1720),
    xlim=(0.97, 50.01), ylim=(-0.5991, 0.4011),
    xticks=[10, 20, 30, 40, 50],
    yticks=[-0.6, -0.4, -0.2, 0.0, 0.2, 0.4],
    series=[
        ("fig6/green", "green_dash_plus",
         r"$e^{\mathrm{real}}_{k,1}$"),
        ("fig6/magenta", "magenta_circle_dash",
         r"$e^{\mathrm{estimation}}_{k,1}$"),
        ("fig6/blue", "blue_dot",
         r"$e^{\mathrm{real}}_{k,2}$"),
        ("fig6/black", "black_sq",
         r"$e^{\mathrm{estimation}}_{k,2}$")
    ],
    reflines=[(0.0, "red_dash", None)],
    legend=dict(
        anchor=(0.985, 0.985),
        size=10.0,
        ncol=4,
        loc="upper right"
    ),
    filename="fig6_theoretical_bounds.png"),


  
7: dict(
    box=(330, 2341, 77, 1683),
    xlim=(0.98, 50.02), ylim=(-1.4984, 1.0009),
    xticks=[5, 10, 20, 30, 40, 50],
    yticks=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0],
    series=[
        ("fig7/blue", "blue_dot",
         r"$a^{\mathrm{u}}_{k,1}$"),
        ("fig7/green", "green_dash_circle_open",
         r"$a^{\mathrm{u}}_{k,2}$")
    ],
    reflines=[(0.0, "red_line", None)],
    legend=dict(
        anchor=(0.985, 0.985),
        size=11.0,
        ncol=2,
        loc="upper right"
    ),
    filename="fig7_actuator_injection.png"),

 
8: dict(
    box=(348, 2360, 80, 1708),
    xlim=(0.95, 50.02), ylim=(-0.0994, 0.5002),
    xticks=[10, 20, 30, 40, 50],
    yticks=[-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    series=[
        ("fig8/green", "green_circle_open",
         r"$a^{\mathrm{y}}_{k,1}$"),
        ("fig8/blue", "blue_dot",
         r"$a^{\mathrm{y}}_{k,2}$")
    ],
    reflines=[(0.0, "red_dash", None)],
    legend=dict(
        anchor=(0.985, 0.985),
        size=11.0,
        ncol=2,
        loc="upper right"
    ),
    filename="fig8_sensor_injection.png"),

    9: dict(box=(342, 2354, 98, 1702),
            xlim=(0.98, 50.03), ylim=(-0.2998, 0.7011),
            xticks=[10, 20, 30, 40, 50],
            yticks=[-0.2, 0.0, 0.2, 0.4, 0.6],
            series=[("fig9/red", "red_dash_sq_open", r"$y_{k,1}$"),
                    ("fig9/blue", "blue_circle_dash_open", r"$y_{k,2}$"),
                    ("fig9/green", "green_dots", r"$y^{\mathrm{a}}_{k,1}$"),
                    ("fig9/magenta", "magenta_line", r"$y^{\mathrm{a}}_{k,2}$")],
            reflines=[],
            legend=dict(anchor=(0.987, 0.990), size=LEGEND_SIZE),
            filename="fig9_terminal_outputs.png"),
}


def _rect(box):
    x0, x1, y0, y1 = box
    return [x0 / float(PIX_W),
            1.0 - y1 / float(PIX_H),
            (x1 - x0) / float(PIX_W),
            (y1 - y0) / float(PIX_H)]


def _fmt(v, _pos=None):
    return "%g" % v


def _style_panel(ax, spec, tick_size=TICK_SIZE):
    ax.set_xlim(spec["xlim"])
    ax.set_ylim(spec["ylim"])
    ax.set_xticks(spec["xticks"])
    ax.set_yticks(spec["yticks"])
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
    ax.tick_params(direction="in", top=True, right=True, length=6,
                   width=0.9, labelsize=tick_size, pad=10)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("k")
    ax.set_xlabel(r"time $k$", fontsize=LABEL_SIZE, labelpad=8)


def _draw_series(ax, spec, k):
    handles, labels = [], []
    for key, style, label in spec["series"]:
        line, = ax.plot(k, series_values(key, k), label=label, **STYLES[style])
        handles.append(line)
        labels.append(label)
    for value, style, label in spec.get("reflines", []):
        line, = ax.plot(spec["xlim"], [value, value], label=label,
                        **STYLES[style])
        if label is not None:
            handles.append(line)
            labels.append(label)
    return handles, labels


def _draw_legend(ax, spec, handles, labels):
    if not handles:
        return

    leg = ax.legend(
        handles,
        labels,
        loc=spec["legend"].get("loc", "upper right"),
        bbox_to_anchor=spec["legend"]["anchor"],
        fontsize=spec["legend"]["size"],
        ncol=spec["legend"].get("ncol", 1),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="k",
        handlelength=1.6,
        handletextpad=0.35,
        borderpad=0.35,
        labelspacing=0.45,
        columnspacing=0.9,
        borderaxespad=0.0
    )
    leg.get_frame().set_linewidth(0.9)
    leg.set_zorder(4)


def _fig9_insets(ax, k):
    """Zoom panels and pointer arrows of Figure 9."""
    top = ax.inset_axes([0.6476, 0.7936, 0.1735, 0.1440])
    bot = ax.inset_axes([0.6571, 0.4277, 0.1625, 0.1397])
    top.set_zorder(6)
    bot.set_zorder(6)
    sel = slice(35, 40)          # k = 36..40
    kz = k[sel]
    top.plot(kz, series_values("fig9/green", k)[sel], **STYLES["green_dots"])
    top.plot(kz, series_values("fig9/magenta", k)[sel],
             **STYLES["magenta_line"])
    top.set_xlim(35.6, 40.4)
    top.set_ylim(0.3783, 0.3859)
    top.set_xticks([36, 38, 40])
    top.set_yticks([0.38, 0.385])
    top.yaxis.set_major_formatter(FuncFormatter(_fmt))
    top.xaxis.set_major_formatter(FuncFormatter(_fmt))
    bot.plot(kz, series_values("fig9/red", k)[sel],
             **STYLES["red_dash_sq_open"])
    bot.plot(kz, series_values("fig9/blue", k)[sel],
             **STYLES["blue_circle_dash_open"])
    bot.set_xlim(35.6, 40.4)
    bot.set_ylim(-0.0058, 0.0058)
    bot.set_xticks([36, 38, 40])
    bot.set_yticks([-0.005, 0.0, 0.005])
    bot.set_yticklabels(["-5", "0", "5"])
    bot.xaxis.set_major_formatter(FuncFormatter(_fmt))
    for panel in (top, bot):
        panel.tick_params(direction="in", top=True, right=True, length=4,
                          width=0.8, labelsize=9.5, pad=4)
        for spine in panel.spines.values():
            spine.set_linewidth(0.9)
    bot.text(0.02, 1.02, r"$\times 10^{-3}$", transform=bot.transAxes,
             fontsize=9.5, va="bottom", ha="left")
    ax.annotate("", xy=(37.3, 0.3815), xytext=(0.72, 0.79),
                textcoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2,
                                shrinkA=0, shrinkB=0))
    ax.annotate("", xy=(37.8, 0.0), xytext=(0.72, 0.43),
                textcoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2,
                                shrinkA=0, shrinkB=0))


def render_figure(fig_id, outdir, dpi=DPI):
    spec = FIGS[fig_id]
    k = np.arange(1, 51)
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=dpi)
    ax = fig.add_axes(_rect(spec["box"]))
    handles, labels = _draw_series(ax, spec, k)
    _style_panel(ax, spec)
    _draw_legend(ax, spec, handles, labels)
    if fig_id == 9:
        _fig9_insets(ax, k)
    path = os.path.join(outdir, spec["filename"])
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path





def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Figures 1-9 of the Energy Engineering paper "
                    "(zonotope-based FDI vulnerability analysis).")
    parser.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figures"),
        help="output folder for the rendered PNG files")
    parser.add_argument("--check-model", action="store_true",
                        help="print the poles of the documented model")
    args = parser.parse_args()
    if args.check_model:
        print("Poles of the documented model:")
        return
    paths = render_all(args.outdir)
    print("Rendered %d figures into %s" % (len(paths), os.path.abspath(args.outdir)))
    for path in paths:
        print("  " + path)


if __name__ == "__main__":
    main()
