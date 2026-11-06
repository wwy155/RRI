"""Annotation counter.

Counts the b- and y-ion labels printed on an annotated tandem mass spectrum.
Each label is rotated text whose glyphs are stacked vertically, so the colour
mask of a class is split into column runs and then into vertical chains; the
count is the modal number of chains over a sweep of glyph-merging gaps.
"""
import collections

import numpy as np
from PIL import Image

COLOURS = {"b": [(31, 119, 180)], "y": [(214, 39, 40), (211, 47, 47)]}  # second shade: fallback
TOL = 60                      # colour tolerance, maximum over the RGB channels
GAPS = (4, 6, 8, 10, 12, 14)  # glyph-merging gaps in pixels
MIN_PLATEAU = 0.6             # gate: share of gaps that agree with the mode


def load_rgb(path):
    with Image.open(path) as im:
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        return np.asarray(im.convert("RGB"))


def colour_mask(arr, rgb, tol=TOL):
    d = np.abs(arr.astype(np.int16) - np.asarray(rgb, dtype=np.int16)).max(axis=2)
    return d <= tol


def _runs(idx, gap):
    runs = []
    for i in idx:
        if runs and i - runs[-1][1] <= gap:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return runs


def chains(mask, gap):
    """Number of labels: vertical chains (gaps <= gap) within every column run."""
    n = 0
    for x0, x1 in _runs(np.where(mask.any(axis=0))[0], 1):
        n += len(_runs(np.where(mask[:, x0:x1 + 1].any(axis=1))[0], gap))
    return n


def count_labels(path, gaps=GAPS):
    """Returns (n_b, n_y, valid)."""
    arr = load_rgb(path)
    counts, plateaus = {}, []
    for c, shades in COLOURS.items():
        mask = colour_mask(arr, shades[0])
        for alt in shades[1:]:
            if not mask.any():
                mask = colour_mask(arr, alt)
        if not mask.any():
            counts[c] = 0
            continue
        n = [chains(mask, g) for g in gaps]
        mode, k = collections.Counter(n).most_common(1)[0]
        counts[c] = mode
        plateaus.append(k / len(n))
    valid = bool(plateaus) and min(plateaus) >= MIN_PLATEAU
    return counts["b"], counts["y"], valid
