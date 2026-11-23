"""Shared helpers for the controlled-sequence dictionary of the tape (folder t1_dictionary, 2026-11-23).

Complex tape coordinates  zc(H) = (zeta_alt, z_short, z_h2, z_h3), shape (..., 4) complex (alt has zero imaginary
part);  LAMV = their eigenvalues.  to7 / from7 convert to the 7 real tape coordinates of tape_common.
A CONTROLLED SEQUENCE is given by ages: seq(L, {age: value}, filler) has length L, its last input has age 0, the
listed ages hold the listed values and every other input is the filler (number or array of length L).
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tape_common import *          # noqa: E402,F401  (numpy as np, w, W, U, run, sample, ORD, LEFT, RIGHT, LAM, ...)

TM = ["alt", "short", "h2", "h3"]
LAMV = np.array([LAM[n] for n in TM])                                  # (4,) complex
LEFTM = np.stack([LEFT[n] for n in TM])                                # (4, 13)
RIGHTM = np.stack([RIGHT[n] for n in TM])                              # (4, 13)
COORD7 = ["alt", "Re short", "Im short", "Re h2", "Im h2", "Re h3", "Im h3"]


def zc(H):
    """(..., 16) states -> (..., 4) complex tape coordinates (alt, short, h2, h3)."""
    return H[..., ORD] @ LEFTM.T


def to7(z):
    return np.stack([z[..., 0].real, z[..., 1].real, z[..., 1].imag, z[..., 2].real, z[..., 2].imag,
                     z[..., 3].real, z[..., 3].imag], -1)


def from7(c):
    return np.stack([c[..., 0] + 0j, c[..., 1] + 1j * c[..., 2], c[..., 3] + 1j * c[..., 4], c[..., 5] + 1j * c[..., 6]], -1)


def seq(L, placed, filler=0.0):
    """Length-L sequence: value v at age a (position L-1-a) for every (a, v) in placed, filler elsewhere."""
    x = np.full(L, filler, dtype=float) if np.isscalar(filler) else np.array(filler, dtype=float)
    for a, v in placed.items():
        x[L - 1 - a] = v
    return x


def final(xs):
    """Final states of a list of sequences of equal length -> (B, 16)."""
    return run(np.array(xs))[:, -1]


def fmtc(z, d=2):
    """Complex vector as 'abs@deg' strings."""
    return " ".join(f"{abs(v):7.{d}f}@{np.degrees(np.angle(v)):5.0f}" for v in np.atleast_1d(z))


def r2(y, yh):
    return 1 - ((y - yh) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()
