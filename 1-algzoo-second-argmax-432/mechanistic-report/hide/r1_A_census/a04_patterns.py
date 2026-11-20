"""A4: share of the resting pattern and the top 15 full gate patterns, with when each occurs.
Also A2.6: delayed responses (units 5, 9 one step later, unit 0 two steps later) by the fine class of the earlier step."""
from cc import *
from a02_events import eff_class, fine_class, EFF, FINE

pid, eff, fine, sR, sN, a2, tts = [], [], [], [], [], [], []
ratio_rest = []
for seed, x, H, Z in batches():
    ev = events(x); e = eff_class(x, ev)
    B = len(x); tt = np.broadcast_to(np.arange(T), (B, T))
    lastR = np.maximum.accumulate(np.where(e == 0, tt, -1), 1)
    lastN = np.maximum.accumulate(np.where(e == 1, tt, -1), 1)
    pid.append(pattern_id(Z > 0).astype(np.int32)); eff.append(e); fine.append(fine_class(x, ev))
    sR.append(np.where(lastR >= 0, tt - lastR, -1)); sN.append(np.where(lastN >= 0, tt - lastN, -1))
pid, eff, fine, sR, sN = map(np.concatenate, (pid, eff, fine, sR, sN))
nseq = len(pid)
bit = lambda i: (pid >> i) & 1 == 1
rest = pid == RESTING
quasi = ~bit(4) & ~bit(5) & ~bit(9) & bit(1) & bit(6) & bit(7) & bit(8)          # pulses off, comparators on, wave units anything
compon = bit(1) & bit(6) & bit(7) & bit(8)

print("## A4.1 share of steps in the resting pattern (4, 5, 9 off; the other 13 on)\n")
hdr = ["pattern class"] + [f"t={t}" for t in range(T)] + ["all t", "t>=2"]
f = lambda m: [f"{m[:, t].mean():.4f}" for t in range(T)] + [f"{m.mean():.4f}", f"{m[:, 2:].mean():.4f}"]
print(md_table(hdr, [["resting (exact)"] + f(rest),
                     ["pulses 4,5,9 off and comparators on (wave units anything)"] + f(quasi),
                     ["pulses 4,5,9 off"] + f(~bit(4) & ~bit(5) & ~bit(9)),
                     ["4 comparators on"] + f(compon),
                     ["all 9 wave units on"] + f(np.all([bit(i) for i in WAVE], 0))]))
print(f"\ndistinct full patterns: {len(np.unique(pid))} (of 65536); at t >= 2: {len(np.unique(pid[:, 2:]))}")
u, c = np.unique(pid, return_counts=True); o = np.argsort(-c)
cum = np.cumsum(c[o]) / pid.size
print("patterns needed to cover 50% / 90% / 99% of all steps:", [int(np.searchsorted(cum, q) + 1) for q in (0.5, 0.9, 0.99)])
print("resting share by effective class of step t (t >= 1): " + ", ".join(f"{EFF[k]}: {rest[:, 1:][eff[:, 1:] == k].mean():.4f}" for k in range(3)))
m = np.zeros_like(rest); m[:, 2:] = (eff[:, 2:] == 2) & (eff[:, 1:-1] == 2)
print(f"resting share when steps t-1 and t are both L*: {rest[m].mean():.4f};   quasi-resting there: {quasi[m].mean():.4f}")

print("\n## A4.2 top 15 full patterns\n")
rows = []


def shift(a, k, fill=-9):
    out = np.full_like(a, fill); out[:, k:] = a[:, :-k]; return out


eff1, eff2 = shift(eff, 1), shift(eff, 2)
for rank, k in enumerate(o[:15]):
    m = pid == u[k]
    cls_t = "/".join(f"{(eff[m] == q).mean():.2f}" for q in range(3))
    cls_1 = "/".join(f"{(eff1[m] == q).mean():.2f}" for q in range(3))
    cls_2 = "/".join(f"{(eff2[m] == q).mean():.2f}" for q in range(3))
    hs = np.bincount(sR[m] + 1, minlength=11) / m.sum()
    top_sR = ", ".join(f"{('none' if a == 0 else a - 1)}: {hs[a]:.2f}" for a in np.argsort(-hs)[:3])
    hn = np.bincount(sN[m] + 1, minlength=11) / m.sum()
    top_sN = ", ".join(f"{('none' if a == 0 else a - 1)}: {hn[a]:.2f}" for a in np.argsort(-hn)[:3])
    per_t = " ".join(f"{v:.3f}" for v in m.mean(0))
    rows.append([rank + 1, int(u[k]), pstr(int(u[k])), f"{c[k] / pid.size:.4f}", f"{cum[rank]:.4f}", cls_t, cls_1, cls_2, top_sR, top_sN, per_t])
print(md_table(["rank", "id", "pattern", "share of all steps", "cumulative", "class t (R*/N*/L*)", "class t-1", "class t-2",
                "steps since last R* (top 3)", "steps since last N* (top 3)", "share of step t = 0..9"], rows))

print("\n## A2.6 delayed responses by the fine class of the earlier step\n")
print("P(unit 5 on at t), P(unit 9 on at t) by the fine class of step t-1 (t >= 2);  P(unit 0 off at t) by the fine class of step t-2 (t >= 3), "
      "the latter also restricted to 'step t-1 is not R*'\n")
rows = []
f1 = shift(fine.astype(np.int16), 1); f2 = shift(fine.astype(np.int16), 2)
for q in range(len(FINE)):
    m1 = f1 == q; m2 = f2 == q
    m2b = m2 & (eff1 != 0)
    rows.append([FINE[q], m1.sum(), f"{bit(5)[m1].mean():.4f}", f"{bit(9)[m1].mean():.4f}", m2.sum(), f"{1 - bit(0)[m2].mean():.4f}",
                 f"{1 - bit(0)[m2b].mean():.4f}"])
print(md_table(["fine class of the earlier step", "n (as step t-1)", "P(5 on at t)", "P(9 on at t)", "n (as step t-2)", "P(0 off at t)",
                "P(0 off at t), given t-1 not R*"], rows))
