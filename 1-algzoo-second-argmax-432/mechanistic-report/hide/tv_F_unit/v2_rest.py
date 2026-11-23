"""V2. Claim 3: what the rest piece adds.  Quiet final states, N(0,1), own seed; fits on the first half, numbers
from the second half.
A. Entry level: D = (entry - runner-up's entry)/M for old entries below the runner-up; share of variance explained
   by an offset per (a, a_M, a_S), + S/M, + x_9, x_8, + the entry's own input.
B. Functional: true tape piece + fitted rest piece; P(runner-up is the top old entry) and agreement with the full score.
C. Controls the investigator did not run:
   - the fitted rest piece ALONE (does the cell label a_S smuggle the answer in?)
   - true tape + rest piece of a DONOR state from the same cell (a_M, a_S), rescaled by M (no fit at all)
   - true tape + rest piece of a random quiet donor (any cell)
   - slope of the comparison at age 2 split by part
Run:  python -B v2_rest.py > v2_rest.txt"""
from vcommon import *

rng = np.random.default_rng(9200)
d = population(sample(500_000, 9201)); n = len(d["h"]); ar = np.arange(n)
pc = pieces(d["h"]); full = d["h"] @ U.T; tape = sum(pc[m] for m in TAPE); rest = full - tape
topf = top_old(full, d["valid"]); train = ar < n // 2; test = ~train
M, S, T3 = d["M"], d["S"], d["T"]; x9, x8 = d["xr"][:, 9], d["xr"][:, 8]; Mn = M[:, None]
print(f"{n} quiet final states (500000 sequences, seed 9201)")

# ---------------------------------------------------------------- A
v = (d["xr"][:, :8] - S[:, None]) / Mn
below = d["valid"] & (v < 0)
cell2 = d["aM"] * 10 + d["aS"]
print("\nA. old entries below the runner-up: rms of D, and share of var(D) explained (test half) by offset per (a, a_M, a_S) | + S/M | + x_9, x_8 | + own input")
for nm, Sc in (("rest", rest), ("tape", tape), ("full", full), ("trans", pc["trans"]), ("echo", pc["echo"]), ("runner", pc["runner"]), ("clock", pc["clock"]), ("level", pc["level"])):
    D = (Sc[:, :8] - Sc[ar, d["iS"]][:, None]) / Mn
    tot_mean = D[below & test[:, None]].mean(); tot = ((D[below & test[:, None]] - tot_mean) ** 2).sum()
    sse = np.zeros(4)
    for c in np.unique(cell2):
        for i in range(8):
            rows = np.flatnonzero((cell2 == c) & below[:, i]); tr = rows[train[rows]]; te = rows[test[rows]]
            if len(tr) < 25 or len(te) == 0:
                sse += (D[te, i] ** 2).sum(); continue
            F = np.stack([np.ones(len(rows)), S[rows] / M[rows], x9[rows] / M[rows], x8[rows] / M[rows], v[rows, i]], 1)
            for j, k in enumerate((1, 2, 4, 5)):
                b = np.linalg.lstsq(F[train[rows], :k], D[tr, i], rcond=None)[0]
                sse[j] += ((D[te, i] - F[test[rows], :k] @ b) ** 2).sum()
    print(f"   {nm:7s} rms {np.sqrt((D[below & test[:, None]] ** 2).mean()):6.2f}   " + "  ".join(f"{1 - s / tot:6.3f}" for s in sse))
Dt = ((tape[:, :8] - tape[ar, d["iS"]][:, None]) / Mn); Dr = ((rest[:, :8] - rest[ar, d["iS"]][:, None]) / Mn)
print(f"   correlation of tape D and rest D, entries below the runner-up: {np.corrcoef(Dt[below], Dr[below])[0, 1]:+.3f};  the maximum's entry: "
      f"{np.corrcoef(Dt[ar, d['iM']], Dr[ar, d['iM']])[0, 1]:+.3f} (var tape {Dt[ar, d['iM']].var():.1f}, rest {Dr[ar, d['iM']].var():.1f}, full {(Dt + Dr)[ar, d['iM']].var():.1f})")

# ---------------------------------------------------------------- B
pS = lambda Sc: np.mean((top_old(Sc, d["valid"]) == d["iS"])[test])
ag = lambda Sc: np.mean((top_old(Sc, d["valid"]) == topf)[test])
print(f"\nB. true tape + fitted rest: P(S top) / agreement with the full score   [rest deleted {pS(tape):.3f} / {ag(tape):.3f}; true rest {pS(full):.3f} / 1.000]")
cells = {"no cells": np.zeros(n, int), "order of arrival": (d["aS"] > d["aM"]).astype(int), "a_M": d["aM"], "a_S": d["aS"], "(a_M, a_S)": cell2, "(a_M, a_S, a_T)": cell2 * 10 + d["aT"]}
feats = {"M": [M], "M, S": [M, S], "M, S, x9": [M, S, x9], "M, S, x9, x8": [M, S, x9, x8], "M, S, T, x9, x8": [M, S, T3, x9, x8]}
fit = {}
for cn, cell in cells.items():
    row = f"   {cn:18s}"
    for fn, F in feats.items():
        fit[cn, fn] = cell_fit(rest / Mn, np.stack(F, 1) / Mn, cell, train) * Mn
        row += f"  {fn}: {pS(tape + fit[cn, fn]):.3f} / {ag(tape + fit[cn, fn]):.3f}"
    print(row)

# ---------------------------------------------------------------- C
print("\nC. controls")
for key in (("(a_M, a_S)", "M, S"), ("(a_M, a_S)", "M, S, x9, x8"), ("(a_M, a_S, a_T)", "M, S, T, x9, x8")):
    print(f"   fitted rest piece ALONE, {key}: P(S top) {pS(fit[key]):.3f}   (true rest alone {pS(rest):.3f}, chance about {np.mean(1 / d['valid'].sum(1)):.3f})")
# donors: for every state a random TRAIN state of the same cell
def donor_index(cell):
    out = np.full(n, -1); order = np.argsort(cell, kind="stable"); bounds = np.flatnonzero(np.diff(cell[order])) + 1
    for idx in np.split(order, bounds):
        tr = idx[train[idx]]
        if len(tr):
            out[idx] = rng.choice(tr, len(idx))
    return out
for cn in ("no cells", "a_S", "(a_M, a_S)", "(a_M, a_S, a_T)"):
    dn = donor_index(cells[cn]); ok = dn >= 0
    Sc = tape + rest[dn] * (M / M[dn])[:, None]
    print(f"   true tape + rest piece of a donor with the same {cn:16s} (rescaled by M): P(S top) {np.mean((top_old(Sc, d['valid']) == d['iS'])[test & ok]):.3f}  agree {np.mean((top_old(Sc, d['valid']) == topf)[test & ok]):.3f}")
# donors matched on (a_M, a_S) and close in S/M: sort within cell by S/M, take the neighbour
sm = S / M; order = np.lexsort((sm, cell2)); nb = np.empty(n, int); nb[order] = np.roll(order, 1)
same = cell2[nb] == cell2
Sc = tape + rest[nb] * (M / M[nb])[:, None]
print(f"   true tape + rest piece of the neighbour in S/M within (a_M, a_S) (median |d S/M| {np.median(np.abs(sm - sm[nb])[same]):.4f}): P(S top) {np.mean((top_old(Sc, d['valid']) == d['iS'])[same]):.3f}"
      f"  agree {np.mean((top_old(Sc, d['valid']) == topf)[same]):.3f}")
print("   slope of the comparison near the runner-up for entries of age 2 / age 3 / ages >= 4, by part:")
i8 = np.broadcast_to(np.arange(8), (n, 8))
for nm, Sc in [("tape", tape), ("rest", rest), ("full", full)] + [(m, pc[m]) for m in OTHER]:
    print(f"      {nm:7s} " + "   ".join(f"{near_slope(Sc, d, extra=e)[0]:6.1f}" for e in (i8 == 7, i8 == 6, i8 <= 5)))
