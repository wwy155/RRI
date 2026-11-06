"""Transient comparator.

De-renders the latest (science) and historical (reference) frame of a
difference-imaging triplet to the sampling grid of their data, detects point
sources in the latest frame with a matched filter and measures each of them
at the same position of the historical frame.
"""
import re

import numpy as np
from PIL import Image
from scipy import ndimage

GRID = 64          # sampling grid of the rendered data
TAU = 5.0          # detection threshold (matched-filter SNR)
TAU_F = 3.0        # presence threshold of forced photometry
PSF_SIGMA = 1.0    # matched filter: Gaussian with a width of one grid cell

_yy, _xx = np.mgrid[-3:4, -3:4]
KERNEL = np.exp(-(_yy ** 2 + _xx ** 2) / (2.0 * PSF_SIGMA ** 2))
KERNEL = KERNEL / np.sqrt((KERNEL ** 2).sum())


def load_gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float64)


def _longest_run(profile, thr=0.5):
    idx = np.where(profile > thr)[0]
    if len(idx) == 0:
        return None
    runs, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] <= 2:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    runs.sort(key=len, reverse=True)
    return int(runs[0][0]), int(runs[0][-1])


def plot_box(a):
    """Axes box: the longest row and column runs whose dark-pixel fraction exceeds 1/2."""
    dark = a < 250
    r, c = _longest_run(dark.mean(axis=1)), _longest_run(dark.mean(axis=0))
    if r is None or c is None:
        return None
    return r, c


def to_grid(a, g=GRID):
    """Block-average the axes box onto the g x g data grid."""
    box = plot_box(a)
    if box is None:
        return None
    (y0, y1), (x0, x1) = box
    c = a[y0 + 1:y1, x0 + 1:x1]
    h, w = c.shape
    if h < g or w < g:
        return None
    hh, ww = h // g, w // g
    return c[:hh * g, :ww * g].reshape(g, hh, g, ww).mean(axis=(1, 3))


def _robust_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def snr_map(G):
    """Matched-filter SNR of a point source at every cell of the grid."""
    R = G - ndimage.median_filter(G, size=5, mode="nearest")
    F = ndimage.correlate(R, KERNEL, mode="nearest")
    return F / max(float(_robust_sigma(F)), 1e-9)


def detect(S, tau=TAU):
    """Detected sources: 3x3 local maxima of S with S >= tau, brightest first."""
    peak = (S == ndimage.maximum_filter(S, size=3, mode="nearest")) & (S >= tau)
    ys, xs = np.where(peak)
    src = [{"x": int(x), "y": int(y), "snr": float(S[y, x])} for y, x in zip(ys, xs)]
    src.sort(key=lambda d: -d["snr"])
    return src


def forced(S, y, x):
    """Forced SNR at (y, x): the maximum of S over the 3x3 cells around it."""
    return float(S[max(0, y - 1):y + 2, max(0, x - 1):x + 2].max())


def compare(science, reference, difference=None, tau=TAU, tau_f=TAU_F):
    """Verdict 'transient' / 'persistent', or None when the validity gate closes.

    The gate requires the axes box of every frame and at least one detected
    source.  The difference frame enters only the gate, never the decision.
    """
    frames = [science, reference] + ([] if difference is None else [difference])
    grids = [to_grid(a) for a in frames]
    if any(g is None for g in grids):
        return None, []
    s_sci, s_ref = snr_map(grids[0]), snr_map(grids[1])
    sources = detect(s_sci, tau)
    for s in sources:
        s["ref_snr"] = forced(s_ref, s["y"], s["x"])
    if not sources:
        return None, []
    absent = any(s["ref_snr"] < tau_f for s in sources)
    return ("transient" if absent else "persistent"), sources


AFFIRM = r"(yes\b|是)"      # option text that affirms a transient event
DENY = r"(no\b|没有|否)"     # option text that denies it


def bind_option(verdict, options):
    """Letter of the option whose text affirms a transient or denies it."""
    if verdict is None:
        return None
    target = AFFIRM if verdict == "transient" else DENY
    for opt in options or []:
        m = re.match(r"\s*\(?([A-Za-z])\)?[.:]?\s*(.*)", str(opt))
        if m and re.match(target, m.group(2).strip().lower()):
            return m.group(1).upper()
    return None
