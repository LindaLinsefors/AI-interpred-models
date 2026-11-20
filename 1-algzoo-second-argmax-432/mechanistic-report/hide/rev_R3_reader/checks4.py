"""Reviewer R3: fourth batch (one-step transfer of a low input to every unit; B_c spectrum). Writes checks4_out.txt."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from pathlib import Path
import numpy as np
from common import *
OUT = Path(__file__).resolve().parent / "checks4_out.txt"
lines = []
def P(*a):
    s = " ".join(str(v) for v in a); print(s); lines.append(s)
np.set_printoptions(precision=3, suppress=True, linewidth=220)
g = -w
P("d h_i(t+1)/d x_t through the comparators only,  sum_k W[i,k] w_k, k in {1,6,7,8}, i = 0..15:")
P(W[:, COMP] @ w[COMP])
P("through all 13 non-pulse units,  sum_k W[i,k] w_k, k in ORD:")
P(W[:, ORD] @ w[ORD])
P("cosine between W[i,COMP] and w[COMP], i = 0..15:")
P(np.array([W[i, COMP] @ w[COMP] / np.linalg.norm(W[i, COMP]) / np.linalg.norm(w[COMP]) for i in range(16)]))
Bc = W[np.ix_(COMP, COMP)] * g[COMP][None, :] / g[COMP][:, None]
ev = np.linalg.eigvals(Bc)
P("B_c 1 =", Bc @ np.ones(4))
P("eig(B_c):", " ".join(f"{abs(v):.4f}@{np.degrees(np.angle(v)):.2f}" for v in ev))
A = W[np.ix_(ORD, ORD)]
wo = w[ORD]
P("|A w_o| / |w_o| =", np.linalg.norm(A @ wo) / np.linalg.norm(wo), "  |A^2 w_o|/|w_o| =", np.linalg.norm(A @ A @ wo) / np.linalg.norm(wo))
OUT.write_text("\n".join(lines), encoding="utf-8")
