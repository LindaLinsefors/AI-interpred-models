"""Run one of the lead's scripts (hide/eNN_*.py) WITHOUT letting it write anything into hide/.

np.savez / np.save / pickle.dump are turned into no-ops and any open(..., 'w'/'a'/'x'/'+') is sent to
the null device, so the workspace stays read-only.  Usage:  python -B run_lead.py e10_tape_shift.py
"""
import builtins
import os
import pickle
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HIDE = HERE.parent
sys.path.insert(0, str(HIDE))
os.chdir(HIDE)

import numpy as np

_open = builtins.open


def _safe_open(file, mode="r", *a, **k):
    if any(c in mode for c in "wax+"):
        return _open(os.devnull, mode, *a, **k)
    return _open(file, mode, *a, **k)


builtins.open = _safe_open
np.savez = lambda *a, **k: None
np.savez_compressed = lambda *a, **k: None
np.save = lambda *a, **k: None
pickle.dump = lambda *a, **k: None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.savefig = lambda *a, **k: None
    from matplotlib.figure import Figure
    Figure.savefig = lambda *a, **k: None
except Exception:
    pass

script = sys.argv[1]
sys.argv = [script]
runpy.run_path(str(HIDE / script), run_name="__main__")
