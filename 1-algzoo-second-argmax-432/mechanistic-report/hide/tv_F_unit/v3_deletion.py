"""V3. Claim 6: the double dissociation tape / clock under deletion from a quiet state at step t0, by later history.
Own code and seed.  Added attacks:
  - RESAMPLING instead of deletion: the part is replaced by the same part of a random other quiet state at t0
    (rescaled by the ratio of the maxima), which stays much closer to reachable states than a deletion
  - every single mode, so that "the tape matters" can be compared with "any part matters"
Columns: changed = share of final answers that differ from the unperturbed run; (clip) = perturbed state clipped at 0;
neg = share of perturbed states with a negative activation.
Run:  python -B v3_deletion.py > v3_deletion.txt"""
from vcommon import *

rng = np.random.default_rng(9300)
B = 300_000
x = sample(B, 9301); H = run(x); o9 = np.argsort(-x, 1); pM9, pS9 = o9[:, 0], o9[:, 1]
ans = lambda h, fut: (run(fut, h0=h)[:, -1] @ U.T).argmax(1)
PARTS = [("TAPE", TAPE), ("clock", ["clock"]), ("runner", ["runner"]), ("level", ["level"]), ("trans", ["trans"]), ("short", ["short"]), ("h2", ["h2"]), ("h3", ["h3"]), ("alt", ["alt"])]
print(f"{B} sequences, seed 9301")
for t0 in (5, 7, 8):
    st = stats(x, t0); q = st["quiet"]
    cls = {"same top two": q & (pM9 == st["pM"]) & (pS9 == st["pS"]), "max overtaken": q & (pM9 > t0) & (pS9 == st["pM"]), "new runner-up": q & (pM9 == st["pM"]) & (pS9 > t0)}
    qi = np.flatnonzero(q)
    print(f"\nt0 = {t0}: {q.sum()} quiet states; the three classes cover {sum(c.sum() for c in cls.values()) / q.sum():.3f}")
    for cn, cq in cls.items():
        h = H[cq, t0]; fut = x[cq, t0 + 1:]; a0 = ans(h, fut); nc = cq.sum()
        don = rng.choice(qi, nc); hd = H[don, t0]; sc = (st["M"][cq] / st["M"][don])[:, None]
        print(f"   {cn}: n {nc}")
        for pn, names in PARTS:
            P = sum(PROJ[m] for m in names)
            hp = h.copy(); hp[:, ORD] -= h[:, ORD] @ P.T
            hr = hp.copy(); hr[:, ORD] += (hd[:, ORD] @ P.T) * sc
            a1 = ans(hp, fut); a1c = ans(np.maximum(hp, 0), fut); a2 = ans(hr, fut); a2c = ans(np.maximum(hr, 0), fut)
            extra = ""
            if pn == "TAPE":
                extra = (f"   | after deletion the answer is: old runner-up {np.mean(a1 == st['pS'][cq]):.3f}, old maximum {np.mean(a1 == st['pM'][cq]):.3f}, later input {np.mean(a1 > t0):.3f}, "
                         f"other old input {np.mean((a1 <= t0) & (a1 != st['pS'][cq]) & (a1 != st['pM'][cq])):.3f}")
            print(f"      {pn:7s} deleted: changed {np.mean(a1 != a0):.3f} (clip {np.mean(a1c != a0):.3f}) neg {np.mean((hp < -1e-9).any(1)):.3f}   "
                  f"resampled: changed {np.mean(a2 != a0):.3f} (clip {np.mean(a2c != a0):.3f}) neg {np.mean((hr < -1e-9).any(1)):.3f}{extra}")
