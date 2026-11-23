"""S2. What do the completing parts add to the tape piece?  Quiet final states (t = 9).
A. Every piece, as D = (entry - runner-up's entry) / M for entries of age >= 2 below the runner-up (and, separately,
   for the maximum's entry): size, and the share of its variance explained, cumulatively, by
     (1) an offset per configuration c = (age of the entry a, a_M, a_S)            [D = k_c]
     (2) + a term in S/M per configuration      [D M = k_c M + k'_c S, the same as a term in the gap M - S]
     (3) + the two newest inputs x_9, x_8 (per configuration, linear)
     (4) + the entry's own input (per configuration, linear in (x - S)/M)
B. Functional test: replace the REST piece (level+runner+clock+trans+echo+pulse) by a description fitted per
   configuration (a_M, a_S) and add it to the true tape piece; P(runner-up on top).  And the reverse for the tape.
Fits use the first half of the states, all numbers are from the second half.
Run:  python -B s2_completion.py > s2_completion.txt"""
from t1_common import *

d = population(600_000, 201); n = len(d["h"]); ar = np.arange(n)
pc = pieces(d["h"]); full = d["h"] @ U.T
tape = sum(pc[m] for m in TAPE); rest = full - tape
top_full = top_entry(full, d["valid"])
train = ar < n // 2; test = ~train
M, S = d["M"], d["S"]
print(f"{n} quiet final states (600000 sequences, seed 201); fits on the first half, numbers from the second half")


def cell_predict(Y, F, cell):
    """Least squares of Y (n, k) on F (n, f) within each cell, fitted on `train`; returns predictions for all rows."""
    out = np.zeros_like(Y)
    order = np.argsort(cell, kind="stable"); bounds = np.flatnonzero(np.diff(cell[order])) + 1
    for idx in np.split(order, bounds):
        tr = idx[train[idx]] if train.shape == cell.shape else idx
        if len(tr) > F.shape[1] + 5:
            out[idx] = F[idx] @ np.linalg.lstsq(F[tr], Y[tr], rcond=None)[0]
    return out


# ---------------------------------------------------------------- A. entry level
i8 = np.broadcast_to(np.arange(8), (n, 8))
v = (d["xr"][:, :8] - S[:, None]) / M[:, None]
sel_below = d["valid"] & (v < 0)
sel_max = i8 == d["iM"][:, None]
for sel_name, sel in (("entries below the runner-up, age >= 2", sel_below), ("the maximum's own entry", sel_max)):
    r, i = np.nonzero(sel)
    cell = ((9 - i) * 100 + d["aM"][r] * 10 + d["aS"][r])
    tr = train[r]; te = ~tr
    feats = [np.ones(len(r)), S[r] / M[r], d["xr"][r, 9] / M[r], d["xr"][r, 8] / M[r], v[r, i]]
    print(f"\nA. {sel_name}: {te.sum()} test entries, {len(np.unique(cell))} configurations (a, a_M, a_S)")
    print("   piece        rms D   sd D  | share of var(D) explained by: (1) offset per config  (2) + S/M  (3) + x_9, x_8  (4) + own input")
    def line(nm, A):
        D = (A[r, i] - A[r, d["iS"][r]]) / M[r]
        txt = f"   {nm:11s} {np.sqrt((D[te] ** 2).mean()):6.2f} {D[te].std():6.2f}  |"
        for k in (1, 2, 4, 5):
            F = np.stack(feats[:k], 1)
            order = np.argsort(cell, kind="stable"); bounds = np.flatnonzero(np.diff(cell[order])) + 1
            pred = np.zeros(len(D))
            for idx in np.split(order, bounds):
                t_ = idx[tr[idx]]
                if len(t_) > 20:
                    pred[idx] = F[idx] @ np.linalg.lstsq(F[t_], D[t_], rcond=None)[0]
            txt += f"   {1 - ((D - pred)[te] ** 2).sum() / ((D[te] - D[te].mean()) ** 2).sum():6.3f}"
        print(txt)
    for m in OTHER:
        line(m, pc[m])
    line("REST (sum)", rest); line("TAPE", tape); line("FULL", full)
    for m in TAPE:
        line(m, pc[m])
    Dt = (tape[r, i] - tape[r, d["iS"][r]]) / M[r]; Dr = (rest[r, i] - rest[r, d["iS"][r]]) / M[r]
    print(f"   correlation of the tape's D with the rest's D: {np.corrcoef(Dt, Dr)[0, 1]:+.3f};  var(tape D) {Dt.var():.1f}, var(rest D) {Dr.var():.1f}, var(full D) {(Dt + Dr).var():.1f}")

# ---------------------------------------------------------------- B. functional substitution
print("\nB. P(runner-up is the top entry among ages >= 2), second half of the states")
cell2 = d["aM"] * 10 + d["aS"]
cell3 = cell2 * 10 + d["aT"]
x9, x8 = d["xr"][:, 9], d["xr"][:, 8]
feat_sets = {
    "nothing (rest piece deleted)": None,
    "one offset pattern per (a_M, a_S), times M": np.stack([M], 1),
    "M and S per (a_M, a_S)": np.stack([M, S], 1),
    "M, S, x_9 per (a_M, a_S)": np.stack([M, S, x9], 1),
    "M, S, x_9, x_8 per (a_M, a_S)": np.stack([M, S, x9, x8], 1),
    "M, S, x_9, x_8, [x_8 - 0.75 S]+, [x_9 - 0.75 S]+ per (a_M, a_S)": np.stack([M, S, x9, x8, np.maximum(x8 - 0.75 * S, 0), np.maximum(x9 - 0.75 * S, 0)], 1),
    "M, S, T per (a_M, a_S)": np.stack([M, S, d["T3"]], 1),
}
pS = lambda A: np.mean((top_entry(A, d["valid"]) == d["iS"])[test])
ag = lambda A: np.mean((top_entry(A, d["valid"]) == top_full)[test])
print(f"   true tape + true rest (full score): {pS(full):.3f}")
print("   true TAPE piece + REST piece replaced by a fit on ...                     P(S top)  agree with full   R2 of the fit (rest piece, indices 0-7, centred over the index)")
cen = lambda A: A[:, :8] - A[:, :8].mean(1, keepdims=True)
for nm, F in feat_sets.items():
    pred = np.zeros((n, 10)) if F is None else cell_predict(rest, F, cell2)
    r2 = 1 - ((cen(rest) - cen(pred))[test] ** 2).sum() / (cen(rest)[test] ** 2).sum()
    print(f"     {nm:66s} {pS(tape + pred):.3f}     {ag(tape + pred):.3f}        {r2:.3f}")
print("   the same with single parts replaced by 'M and S per (a_M, a_S)' and everything else true:")
for m in OTHER:
    pred = cell_predict(pc[m], np.stack([M, S], 1), cell2)
    r2 = 1 - ((cen(pc[m]) - cen(pred))[test] ** 2).sum() / (cen(pc[m])[test] ** 2).sum()
    print(f"     {m:8s} P(S top) {pS(full - pc[m] + pred):.3f}   agree {ag(full - pc[m] + pred):.3f}   R2 {r2:.3f}   (deleted instead: {pS(full - pc[m]):.3f})")
print("   the reverse: true REST piece + TAPE piece replaced by a fit on ...")
for nm, F, cell in (("nothing (tape piece deleted)", None, cell2), ("M and S per (a_M, a_S)", np.stack([M, S], 1), cell2),
                    ("M, S, T per (a_M, a_S, a_T)", np.stack([M, S, d["T3"]], 1), cell3)):
    pred = np.zeros((n, 10)) if F is None else cell_predict(tape, F, cell)
    r2 = 1 - ((cen(tape) - cen(pred))[test] ** 2).sum() / (cen(tape)[test] ** 2).sum()
    print(f"     {nm:66s} {pS(rest + pred):.3f}     {ag(rest + pred):.3f}        {r2:.3f}")

# ---------------------------------------------------------------- C. the offsets themselves, in numbers
print("\nC. The rest piece as an offset: mean of (rest entry - rest entry of the runner-up) / M by age of the entry (columns 2..9),")
print("   for some configurations (a_M, a_S); entries below the runner-up only; the tape piece's mean in the same cells below it")
r, i = np.nonzero(sel_below)
Dr = (rest[r, i] - rest[r, d["iS"][r]]) / M[r]; Dt = (tape[r, i] - tape[r, d["iS"][r]]) / M[r]; a = 9 - i
for aM, aS in ((2, 5), (2, 8), (5, 2), (8, 2), (4, 7), (7, 4), (3, 9), (9, 3)):
    q = (d["aM"][r] == aM) & (d["aS"][r] == aS)
    row = lambda D: " ".join(f"{D[q & (a == k)].mean():6.1f}" if (q & (a == k)).sum() > 50 else "    --" for k in range(2, 10))
    print(f"   a_M={aM} a_S={aS}  rest: {row(Dr)}   | sd within cell {np.mean([Dr[q & (a == k)].std() for k in range(2, 10) if (q & (a == k)).sum() > 50]):.1f}")
    print(f"               tape: {row(Dt)}   | sd within cell {np.mean([Dt[q & (a == k)].std() for k in range(2, 10) if (q & (a == k)).sum() > 50]):.1f}")
    print(f"               full: {row(Dr + Dt)}   | sd within cell {np.mean([(Dr + Dt)[q & (a == k)].std() for k in range(2, 10) if (q & (a == k)).sum() > 50]):.1f}")
