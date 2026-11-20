"""Verifier V1, attack E2: does a REAL step without an event shift the tape by one position?

T1/T2 concern the resting map A applied to h[ORD].  Here: real consecutive states (h_{t-1}, h_t) of the
full network.  "No event" at step t:  x_t < S_{t-1}  (top two unchanged).  Positions 0..t-1 exist at both
times; position p sits at index 10-t+p at time t-1 and at index 9-t+p at time t.

(a) per t: R^2 of  s_t[i] ~ s_{t-1}[i+1]  over the old positions (per-state mean removed), with slope 1 and
    with a fitted common slope; controls: same index (no shift), shift by two.
(b) following ONE input over a no-event step: relative score r = (s[p] - s[p2]) / M before and after,
    by the age the input has after the step.
Seeds 5401..5404, 200k sequences each.   Run: python -B v1_attack_realshift.py -> v1_attack_realshift.txt
"""
from vlib import *

P = Tee("v1_attack_realshift.txt")
SEEDS = [5401, 5402, 5403, 5404]
NB = 200_000


def cen(a):
    return a - a.mean(1, keepdims=True)


class Acc:
    """accumulate sums for R^2 with slope 1 and with a fitted common slope"""
    def __init__(self):
        self.yy = self.pp = self.yp = 0.0; self.n = 0

    def add(self, y, p):
        y, p = cen(y), cen(p)
        self.yy += (y * y).sum(); self.pp += (p * p).sum(); self.yp += (y * p).sum(); self.n += len(y)

    def r2(self):
        if self.n == 0 or self.yy == 0:
            return np.nan, np.nan, np.nan
        r2_1 = 1 - (self.yy - 2 * self.yp + self.pp) / self.yy
        beta = self.yp / self.pp
        return r2_1, 1 - (self.yy - 2 * beta * self.yp + beta * beta * self.pp) / self.yy, beta


conds = ["no event, any S", "no event, S_{t-1}>0.1", "no event, S>0.1, 4 comparators on at t", "no event, S>0.1, positions of age>=2 at t only",
         "new runner-up, S>0.1", "new max, S>0.1"]
acc = {c: {k: [Acc() for _ in range(10)] for k in ("shift", "same", "shift2", "resting")} for c in conds}
# (b) tracking table
TR = {k: [] for k in ("age", "r0", "r1", "kind", "t", "a2", "v")}
Uo = U[:, ORD]; A = W[np.ix_(ORD, ORD)]

for seed in SEEDS:
    x = sample(NB, seed)
    H, Z = run(x, return_pre=True)
    for t in range(2, 10):
        s0, full0 = scores(H, t - 1)                 # (B, t)   positions 0..t-1
        s1, full1 = scores(H, t)                     # (B, t+1) positions 0..t
        p, val = top3(x, t - 1)
        p1, p2, M, S = p[:, 0], p[:, 1], val[:, 0], val[:, 1]
        xt = x[:, t]
        noev = xt < S
        comp_on = (Z[:, t][:, COMP] > 0).all(1)
        # resting-map prediction of the next scores from the full ordinary state (pulses ignored)
        rest = (H[:, t - 1][:, ORD] @ (Uo @ A).T)[:, 9 - t: 9]     # index i=9-t..8  <-> positions 0..t-1 at time t
        masks = {conds[0]: noev, conds[1]: noev & (S > 0.1), conds[2]: noev & (S > 0.1) & comp_on,
                 conds[4]: (xt > S) & (xt < M) & (S > 0.1), conds[5]: (xt > M) & (S > 0.1)}
        for c, m in masks.items():
            if m.sum() < 10:
                continue
            y = s1[m][:, :t]; pr = s0[m]
            acc[c]["shift"][t].add(y, pr)
            acc[c]["resting"][t].add(y, rest[m])
            if t >= 3:
                acc[c]["same"][t].add(s1[m][:, 1:t], s0[m][:, 0:t - 1])
                acc[c]["shift2"][t].add(s1[m][:, 0:t - 1], s0[m][:, 1:t])
        if t >= 4:   # positions that have age >= 2 after the step: p <= t-2
            m = masks[conds[1]]
            acc[conds[3]]["shift"][t].add(s1[m][:, :t - 1], s0[m][:, :t - 1])
            acc[conds[3]]["resting"][t].add(s1[m][:, :t - 1], rest[m][:, :t - 1])
            acc[conds[3]]["same"][t].add(s1[m][:, 1:t - 1], s0[m][:, 0:t - 2])
            acc[conds[3]]["shift2"][t].add(s1[m][:, 0:t - 2], s0[m][:, 1:t - 1])
        # (b) tracking, no event and S > 0.2
        m = noev & (S > 0.2)
        r = np.arange(m.sum())
        s0m, s1m, p1m, p2m, Mm, Sm = s0[m], s1[m][:, :t], p1[m], p2[m], M[m], S[m]
        r0 = (s0m - s0m[r, p2m][:, None]) / Mm[:, None]
        r1 = (s1m - s1m[r, p2m][:, None]) / Mm[:, None]
        kind = np.zeros(r0.shape, np.int8); kind[r, p1m] = 1; kind[r, p2m] = 2
        age = t - np.broadcast_to(np.arange(t), r0.shape)
        sel = kind != 2
        vv = (x[m][:, :t] - Sm[:, None]) / Mm[:, None]
        TR["age"].append(age[sel].astype(np.int8)); TR["r0"].append(r0[sel].astype(np.float32)); TR["r1"].append(r1[sel].astype(np.float32))
        TR["kind"].append(kind[sel]); TR["t"].append(np.full(sel.sum(), t, np.int8))
        TR["a2"].append(np.broadcast_to((t - p2m)[:, None], r0.shape)[sel].astype(np.int8)); TR["v"].append(vv[sel].astype(np.float32))
    del H, Z

P("=== (a) R^2 of s_t on shifted s_{t-1}, old positions, per-state mean removed ===")
P("    columns per t: shift by one [slope 1 | fitted slope (slope)] ; resting-map prediction U_o A h_{t-1}[ORD] [slope 1] ; controls with fitted slope: same index | shift by two")
for c in conds:
    P(f"--- {c}")
    for t in range(2, 10):
        a = acc[c]
        if a["shift"][t].n == 0:
            continue
        r1, rb, b = a["shift"][t].r2(); q1, qb, _ = a["resting"][t].r2()
        s1_, sb, _ = a["same"][t].r2(); d1, db, _ = a["shift2"][t].r2()
        P(f"   t={t}: n={a['shift'][t].n:7d}  shift1 {r1:6.3f} | {rb:6.3f} ({b:5.2f})   resting {q1:6.3f}   same-index {sb:6.3f}   shift2 {db:6.3f}")

TR = {k: np.concatenate(v) for k, v in TR.items()}
P("\n=== (b) one input followed over a no-event step (S_{t-1} > 0.2): r = (s[p]-s[p2])/M before (r0) and after (r1) ===")
P("    'other' inputs (neither max nor runner-up), by age AFTER the step:")
P("    age      n     med r0    med r1   med(r1/r0)   LS0 slope r1~r0   corr    R2(r1=r0)   P(sign kept)")
for a in range(1, 10):
    m = (TR["kind"] == 0) & (TR["age"] == a)
    r0, r1 = TR["r0"][m].astype(np.float64), TR["r1"][m].astype(np.float64)
    P(f"    {a}  {m.sum():9d} {np.median(r0):9.2f} {np.median(r1):9.2f} {np.median(r1/r0):9.3f} {(r0*r1).sum()/(r0*r0).sum():12.3f}"
      f" {np.corrcoef(r0, r1)[0,1]:10.3f} {1-((r1-r0)**2).sum()/((r1-r1.mean())**2).sum():10.3f} {np.mean((r0<0)==(r1<0)):10.4f}")
P("    same, restricted to inputs with v = (x_p - S)/M in (-0.7,-0.5]  (median r by age; a pure shift keeps it constant):")
for lo, hi in ((-0.7, -0.5), (-np.inf, -2.0), (-0.10, -0.05)):
    P(f"      v in ({lo},{hi}]: " + "  ".join(
        f"a{a}: {np.median(TR['r0'][(TR['kind']==0)&(TR['age']==a)&(TR['v']>lo)&(TR['v']<=hi)]):.1f}->{np.median(TR['r1'][(TR['kind']==0)&(TR['age']==a)&(TR['v']>lo)&(TR['v']<=hi)]):.1f}"
        for a in range(1, 10)))
P("    the max (p = p1), by age AFTER the step:")
for a in range(1, 10):
    m = (TR["kind"] == 1) & (TR["age"] == a)
    r0, r1 = TR["r0"][m].astype(np.float64), TR["r1"][m].astype(np.float64)
    P(f"    {a}  {m.sum():9d} {np.median(r0):9.2f} {np.median(r1):9.2f} {np.median(r1/r0):9.3f} {(r0*r1).sum()/(r0*r0).sum():12.3f}"
      f" {np.corrcoef(r0, r1)[0,1]:10.3f} {1-((r1-r0)**2).sum()/((r1-r1.mean())**2).sum():10.3f} {np.mean((r0<0)==(r1<0)):10.4f}")
P("    'other' inputs, ages>=2 after the step, split by age of the runner-up after the step (a2):  med(r1/r0), corr")
for a2 in range(1, 10):
    m = (TR["kind"] == 0) & (TR["age"] >= 2) & (TR["a2"] == a2)
    r0, r1 = TR["r0"][m].astype(np.float64), TR["r1"][m].astype(np.float64)
    P(f"      a2={a2}: n={m.sum():8d} med ratio {np.median(r1/r0):6.3f} corr {np.corrcoef(r0, r1)[0,1]:6.3f}")
