"""S5. Two follow-ups.
A. How coarse can the description of the REST piece (level+runner+clock+trans+echo+pulse) be?  It is replaced by a
   least-squares fit on (M, S) or (M, S, x_9, x_8) within cells of increasing detail and added to the true tape
   piece; P(runner-up on top among ages >= 2), quiet final states, fits on the first half, numbers from the second.
B. Deleting the tape (or one mode) from a quiet state at step t0 and continuing: the share of changed answers,
   split by what happens AFTER t0:
     same top two     the maximum and the runner-up at t0 are still the top two at the end (answer = old runner-up)
     max overtaken    a later input exceeds the old maximum, which becomes the answer
     new runner-up    a later input lands between the old top two and is the answer
Run:  python -B s5_offsets_and_deletion.py > s5_offsets_and_deletion.txt"""
from t1_common import *

d = population(600_000, 501); n = len(d["h"]); ar = np.arange(n)
pc = pieces(d["h"]); full = d["h"] @ U.T; top_full = top_entry(full, d["valid"])
tape = sum(pc[m] for m in TAPE); rest = full - tape
train = ar < n // 2; test = ~train
M, S = d["M"], d["S"]; x9, x8 = d["xr"][:, 9], d["xr"][:, 8]


def cell_fit(Y, F, cell):
    out = np.zeros_like(Y)
    order = np.argsort(cell, kind="stable"); bounds = np.flatnonzero(np.diff(cell[order])) + 1
    for idx in np.split(order, bounds):
        tr = idx[train[idx]]
        if len(tr) > 3 * F.shape[1] + 5:
            out[idx] = F[idx] @ np.linalg.lstsq(F[tr], Y[tr], rcond=None)[0]
    return out


pS = lambda A: np.mean((top_entry(A, d["valid"]) == d["iS"])[test])
ag = lambda A: np.mean((top_entry(A, d["valid"]) == top_full)[test])
cells = {"no cells (one pattern for all states)": np.zeros(n, int), "order of arrival (runner-up before / after the maximum)": (d["aS"] > d["aM"]).astype(int),
         "age of the maximum a_M": d["aM"], "age of the runner-up a_S": d["aS"], "(a_M, a_S)": d["aM"] * 10 + d["aS"],
         "(a_M, a_S, a_T)": d["aM"] * 100 + d["aS"] * 10 + d["aT"]}
Mn = M[:, None]
print(f"A. {n} quiet final states (600000 sequences, seed 501).  True tape piece + fitted rest piece.")
print(f"   rest deleted: {pS(tape):.3f} / {ag(tape):.3f};  rest true: {pS(full):.3f} / 1.000      (P(S top) / agreement with the full score's top entry)")
print("   cells                                                     fit on M, S          fit on M, S, x_9, x_8")
for nm, cell in cells.items():
    row = f"   {nm:56s}"
    for F in (np.stack([M, S], 1), np.stack([M, S, x9, x8], 1)):
        pred = cell_fit(rest / Mn, F / Mn, cell) * Mn
        row += f"  {pS(tape + pred):.3f} / {ag(tape + pred):.3f}      "
    print(row)

# ---------------------------------------------------------------- B. deletion split by later history
B = 400_000
x = sample(B, 502); truth = second_argmax(x); H = run(x)
order9 = np.argsort(-x, 1); pM9, pS9 = order9[:, 0], order9[:, 1]
PROJ = {m: projector([m]) for m in NAMES}
def delete(h, names):
    out = h.copy(); out[:, ORD] -= h[:, ORD] @ sum(PROJ[m] for m in names).T
    return out
print(f"\nB. Deleting a part of a quiet state at step t0 and continuing with the real inputs ({B} sequences, seed 502).")
print("   per class: n, accuracy before | then for each deleted part: share of answers changed (and, in brackets, with the perturbed state clipped at zero)")
dels = [("TAPE", TAPE), ("short", ["short"]), ("h2", ["h2"]), ("h3", ["h3"]), ("alt", ["alt"]), ("clock", ["clock"]), ("runner", ["runner"]), ("trans", ["trans"])]
for t0 in (5, 7, 8):
    st = prefix_stats(x, t0); q = st["quiet"]
    cls = {"same top two": q & (pM9 == st["pM"]) & (pS9 == st["pS"]), "max overtaken": q & (pM9 > t0) & (pS9 == st["pM"]),
           "new runner-up": q & (pM9 == st["pM"]) & (pS9 > t0)}
    print(f"   t0 = {t0}: {q.sum()} quiet states; classes cover {sum(c.sum() for c in cls.values()) / q.sum():.3f} of them")
    for cn, cq in cls.items():
        h = H[cq, t0]; fut = x[cq, t0 + 1:]
        a0 = answer(continue_from(h, fut)[:, -1])
        row = f"      {cn:14s} n {cq.sum():6d}  acc {np.mean(a0 == truth[cq]):.3f} |"
        for dn, names in dels:
            hp = delete(h, names)
            a1 = answer(continue_from(hp, fut)[:, -1]); a2 = answer(continue_from(np.maximum(hp, 0), fut)[:, -1])
            row += f" {dn} {np.mean(a1 != a0):.3f} ({np.mean(a2 != a0):.3f})"
            if dn == "TAPE":
                where = f"         after deleting the TAPE the answer is: old runner-up {np.mean(a1 == st['pS'][cq]):.3f}, old maximum {np.mean(a1 == st['pM'][cq]):.3f}, an input after t0 {np.mean(a1 > t0):.3f}, another old input {np.mean((a1 <= t0) & (a1 != st['pS'][cq]) & (a1 != st['pM'][cq])):.3f}"
        print(row); print(where)
