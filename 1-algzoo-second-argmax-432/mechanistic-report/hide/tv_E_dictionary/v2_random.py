"""V2. Random quiet states: recomputation of the investigator's claims 2 (exact writes, pulses with fixed patterns), 3 (skeletons)
and 7 (formula N1) with own code and seed 4242 (300000 iid N(0,1) sequences), at t = 9 and t = 6; then the same skeleton
numbers for every step t = 3..9 and for uniform inputs on [-1, 1].
R2 = pooled over the 7 tape coordinates / M_t, centred on the mean of the real ones, quiet states only.
"changed" = share of quiet states whose final answer changes when the tape coordinates at step t are replaced and the run continued.
Run:  python -B v2_random.py > v2_random.txt"""
from v_common import *

B = 300_000
kw, k4, k5, k9 = LEFTM @ w[ORD], LEFTM @ W[ORD, 4], LEFTM @ W[ORD, 5], LEFTM @ W[ORD, 9]
IC, IW = [ORD.index(u) for u in COMP], [ORD.index(u) for u in WAVE]
kA = from7(np.load(HERE / "v1_patterns.npz")["kA"])
rng = np.random.default_rng(11)


class Pop:
    def __init__(self, x, t):
        self.x, self.t = x, t
        self.H, self.Z = run(x, return_pre=True)
        self.st = top3(x, t); self.q = self.st["quiet"]; self.M = self.st["M"]
        self.ans0 = answer(self.H[:, 9]); self.z = zc(self.H[:, t]); self.c = to7(self.z[self.q]) / self.M[self.q, None]
        self.prevM, self.prevS = prev_top2(x)
        self.pos = np.arange(t + 1); self.pre = x[:, : t + 1]

    def R2(self, zf, s=None):
        s = self.q if s is None else s
        d = to7(zf[s] - self.z[s]) / self.M[s, None]
        return 1 - (d ** 2).sum() / ((to7(self.z[s]) / self.M[s, None] - self.c.mean(0)) ** 2).sum()

    def changed(self, zf, s=None, clip=False):
        s = self.q if s is None else s
        h = set_tape(self.H[s, self.t], zf[s]); neg = (h < -1e-9).any(1).mean()
        if clip: h = np.maximum(h, 0)
        fin = h if self.t == 9 else run(self.x[s, self.t + 1:], h0=h)[:, -1]
        return np.mean(answer(fin) != self.ans0[s]), neg

    def line(self, label, zf):
        ch, neg = self.changed(zf); chc, _ = self.changed(zf, clip=True)
        print(f"   {label:58s} R2 {self.R2(zf):7.3f}   changed {ch:.3f}  (clipped at 0: {chc:.3f}; negative activation in {neg:.2f})")

    def skel(self, keep, fill=0.0):
        return zc(run(np.where(keep, self.pre, fill))[:, -1])

    def leak(self, keep):
        return (np.where(keep, 0.0, self.pre)[:, :, None] * (kw * LAMV ** (self.t - self.pos)[:, None])[None]).sum(1)

    def sets(self):
        st, pos, pre = self.st, self.pos, self.pre
        isM, isS = pos == st["pM"][:, None], pos == st["pS"][:, None]
        rank = np.argsort(np.argsort(-pre, 1), 1)
        top2arr = pre > self.prevS[:, : self.t + 1]; near = pre > 0.75 * np.maximum(self.prevS[:, : self.t + 1], 0)
        lo, hi = np.minimum(st["pM"], st["pS"])[:, None], np.maximum(st["pM"], st["pS"])[:, None]
        return dict(isM=isM, isS=isS, rank=rank, top2arr=top2arr, cand=top2arr | near, after=pos > hi, before=pos < lo, between=(pos > lo) & (pos < hi))


def exact_terms(P):
    x, H, Z, t, st = P.x, P.H, P.Z, P.t, P.st
    T = {n: np.zeros((len(x), 4), complex) for n in ("direct", "pulse 4", "pulse 9", "pulse 5: maximum", "pulse 5: runner-up", "pulse 5: dethroned",
                                                       "pulse 5: below", "pulse 5: step 0", "clip comparators", "clip wave units")}
    for s in range(t + 1):
        for n in T: T[n] = T[n] * LAMV
        T["direct"] += x[:, s, None] * kw
        if s >= 1:
            T["pulse 4"] += H[:, s - 1, 4, None] * k4; T["pulse 9"] += H[:, s - 1, 9, None] * k9
            u = H[:, s - 1, 5, None] * k5; j = s - 2
            if j < 0: T["pulse 5: step 0"] += u
            else:
                above = x[:, j] > P.prevS[:, j]; isM, isS = st["pM"] == j, st["pS"] == j
                for n, m in (("maximum", isM), ("runner-up", isS), ("dethroned", ~isM & ~isS & above), ("below", ~isM & ~isS & ~above)):
                    T["pulse 5: " + n] += u * m[:, None]
        clip = np.maximum(-Z[:, s][:, ORD], 0)
        T["clip comparators"] += clip[:, IC] @ LEFTM[:, IC].T; T["clip wave units"] += clip[:, IW] @ LEFTM[:, IW].T
    return T


def n1(P, S_, g=19.1, th0=0.75, beta=0.44):
    st, t = P.st, P.t
    out = P.skel(S_["isM"] | S_["isS"]); start = np.maximum(st["pM"], st["pS"]) + 1; Th = th0 * st["S"]; nw = np.zeros(len(P.x))
    for p in range(t + 1):
        e = np.where(p >= start, np.maximum(P.pre[:, p] - Th, 0), 0.0)
        if t - p >= 2: out = out + (g * e)[:, None] * kA * LAMV ** (t - p - 2); nw += e > 0
        Th = Th + beta * e
    return out, nw


def recompute(x, t):
    P = Pop(x, t); q = P.q; S_ = P.sets()
    print(f"\n===== t = {t}: {q.sum()} quiet states of {len(x)};  rms |tape| / M = {np.sqrt((P.c ** 2).sum(1).mean()):.2f};  accuracy of the network on them {np.mean(P.ans0[q] == np.argsort(x, 1)[q, -2]):.3f}")
    print(" references")
    perm = rng.permutation(np.flatnonzero(q)); donor = np.zeros_like(P.z); donor[q] = P.z[perm] / P.M[perm, None] * P.M[q, None]
    P.line("zero tape", np.zeros_like(P.z)); P.line("mean tape (times M)", from7(P.c.mean(0))[None] * P.M[:, None]); P.line("tape of another quiet state (times M)", donor)
    print(" exact writes (claim 2)")
    T = exact_terms(P); tot = sum(T.values())
    print(f"   exactness: max |sum - tape| = {np.abs(tot - P.z).max():.1e};   rms |term| / M: " + ", ".join(f"{n} {np.sqrt((np.abs(v[q] / P.M[q, None]) ** 2).sum(1).mean()):.2f}" for n, v in T.items()))
    p5 = sum(v for n, v in T.items() if n.startswith("pulse 5")); pul = p5 + T["pulse 4"] + T["pulse 9"]
    P.line("three pulse units with fixed patterns", pul); P.line("... + direct input", pul + T["direct"]); P.line("all but the wave-unit clips", tot - T["clip wave units"])
    P.line("exact minus unit-5 writes of inputs below the runner-up", tot - T["pulse 5: below"]); P.line("exact minus unit-5 writes of dethroned inputs", tot - T["pulse 5: dethroned"])
    d7 = to7((pul - P.z)[q]) / P.M[q, None]
    print("   per coordinate R2 of the three pulse units:", np.round(1 - (d7 ** 2).sum(0) / ((P.c - P.c.mean(0)) ** 2).sum(0), 2))
    print(" skeletons (claim 3): dropped inputs -> 0")
    ms = S_["isM"] | S_["isS"]; r = S_["rank"]
    sk = {"M, S": ms, "top 3": r < 3, "top 4": r < 4, "top 5": r < 5, "all positive inputs": P.pre > 0, "top two at arrival": S_["top2arr"], "candidates": S_["cand"],
          "M, S + inputs after both": ms | S_["after"], "M, S + inputs before both": ms | S_["before"], "M, S + inputs between them": ms | S_["between"]}
    if t == 9: sk.update({"candidates + inputs of age 0, 1": S_["cand"] | (P.pos >= 8), "all but the newest input": np.broadcast_to(P.pos < 9, P.pre.shape)})
    Zs = {}
    for n, keep in sk.items():
        Zs[n] = P.skel(keep); P.line(f"{n} [{keep[q].sum(1).mean():.2f} kept]", Zs[n])
        if n in ("candidates", "all but the newest input", "M, S", "top 3"): P.line("      + linear leak of the dropped inputs", Zs[n] + P.leak(keep))
    P.line("top 3, the rest replaced by -M", P.skel(r < 3, -P.M[:, None]))
    print(" formula N1 (claim 7)")
    zf, nw = n1(P, S_); P.line("N1: skeleton (M, S) + explicit later writes", zf)
    ex = P.skel(ms | (S_["after"] & (P.pos <= t - 2)))
    for k in (0, 1, 2):
        s = q & ((nw == k) if k < 2 else (nw >= 2))
        dt, df = to7(ex[s] - Zs["M, S"][s]), to7(zf[s] - Zs["M, S"][s])
        print(f"   {k}{'+' if k == 2 else ' '} later writes (n = {s.sum():6d}): R2 skeleton (M, S) {P.R2(Zs['M, S'], s):.3f}, N1 {P.R2(zf, s):.3f}, skeleton with the real later inputs {P.R2(ex, s):.3f} | "
              f"changed {P.changed(Zs['M, S'], s)[0]:.3f}, {P.changed(zf, s)[0]:.3f}, {P.changed(ex, s)[0]:.3f}" + (f" | share of the later inputs' contribution reproduced {1 - ((dt - df) ** 2).sum() / (dt ** 2).sum():.3f}" if k else ""))
    return P


x = sample(B, 4242)
for t in (9, 6):
    recompute(x, t)

print("\n\n===== OTHER POPULATIONS: skeletons at every step, N(0,1) (same sequences) and uniform on [-1, 1] (seed 4243)")
for dist, xx in (("normal", x), ("uniform", sample(B, 4243, "uniform"))):
    print(f" {dist}:  t | n quiet | R2, changed: M,S | top 3 | candidates | candidates + leak | three pulses fixed patterns | mean tape changed | kept (candidates)")
    for t in range(3, 10):
        P = Pop(xx, t); S_ = P.sets(); ms = S_["isM"] | S_["isS"]; out = []
        for keep, lk in ((ms, 0), (S_["rank"] < 3, 0), (S_["cand"], 0), (S_["cand"], 1)):
            zf = P.skel(keep) + (P.leak(keep) if lk else 0); out.append(f"{P.R2(zf):6.3f} {P.changed(zf)[0]:.3f}")
        T = exact_terms(P); pul = sum(v for n, v in T.items() if n.startswith("pulse"))
        out.append(f"{P.R2(pul):6.3f} {P.changed(pul)[0]:.3f}")
        print(f"   {t} | {P.q.sum():6d} | " + " | ".join(out) + f" | {P.changed(from7(P.c.mean(0))[None] * P.M[:, None])[0]:.3f} | {S_['cand'][P.q].sum(1).mean():.2f}")
