"""C5e. Robustness of the C5 interventions: the edited state is clipped at zero (h' = max(h', 0)), so that it is a state a ReLU
network can be in.  (The plain edit leaves negative ordinary units in 19-95% of the edited states.)  200k sequences, seed 1500 (same as c5_causal.py)."""
from cl import *
NS, SEED = 200_000, 1500
x, H, Z, st = gen(NS, SEED)
M, S, a1, p1 = st["M1"], pos(st["M2"]), st["a1"], st["p1"]
true = second_argmax(x); ans0 = logits(H).argmax(1)
rank = (-x).argsort(1).argsort(1) + 1
nat = np.load(HERE / "c4_thresholds.npz")["med_u"]

def edit(h, F, clip):
    h2 = h.copy(); zc = h[:, ORD] @ LC
    h2[:, ORD] += 2 * np.real(np.multiply.outer((F - 1) * zc, RC))
    return np.maximum(h2, 0) if clip else h2

EDITS = [(f"k={k:+d}", np.exp(1j * k * OMEGA)) for k in (-2, -1, 1, 2)] + [(f"f={f}", f) for f in (0, 0.5, 2)]
print("edit   | plain edit: changed, accuracy | clipped edit: changed, accuracy, clock coordinate actually obtained relative to the target (LS, |.|@deg) | class R (max at t ends as runner-up): changed / new-old=-k, plain | clipped")
for name, F in EDITS:
    k = int(name[2:]) if name.startswith("k") else None
    out = {c: dict(n=0, ch=0, ok=0, Rn=0, Rch=0, Rk=0) for c in (False, True)}
    num = 0; den = 0
    for t in range(1, 9):
        idx = np.nonzero((M[:, t] > 0) & (a1[:, t] >= 1))[0]
        for clip in (False, True):
            h2 = edit(H[idx, t], F, clip)
            ans = (run(x[idx, t + 1:], h0=h2)[:, -1] @ U.T).argmax(1)
            o = out[clip]
            o["n"] += idx.size; o["ch"] += (ans != ans0[idx]).sum(); o["ok"] += (ans == true[idx]).sum()
            R = rank[idx, p1[idx, t]] == 2
            o["Rn"] += R.sum(); o["Rch"] += (ans != ans0[idx])[R].sum()
            if k is not None:
                o["Rk"] += ((ans - ans0[idx]) == -k)[R].sum()
            if clip:
                zt = F * (H[idx, t][:, ORD] @ LC); zg = h2[:, ORD] @ LC
                num += np.sum(zg * np.conj(zt)); den += np.sum(abs(zt) ** 2)
    a, b = out[False], out[True]
    got = num / den if den > 0 else np.nan
    print(f"{name:6s} | {a['ch'] / a['n']:.4f}  {a['ok'] / a['n']:.4f} | {b['ch'] / b['n']:.4f}  {b['ok'] / b['n']:.4f}  {fmt(got) if den > 0 else '-'} | "
          f"{a['Rch'] / a['Rn']:.3f} / {a['Rk'] / a['Rn'] if k is not None else float('nan'):.3f} | {b['Rch'] / b['Rn']:.3f} / {b['Rk'] / b['Rn'] if k is not None else float('nan'):.3f}")

print("\nthresholds for x_{t+1} after a clipped phase shift: D(target = natural age a+k), D(unshifted = natural age a), mean over (a, k) cells")
Dt, Du = [], []
for a in range(1, 9):
    for k in (-2, -1, 1, 2):
        if not (1 <= a + k <= 8): continue
        us = []
        for t in range(1, 9):
            idx = np.nonzero((M[:, t] > 0) & (a1[:, t] == a))[0]
            h2 = edit(H[idx, t], np.exp(1j * k * OMEGA), True)
            us.append((thresholds(h2) - S[idx, t, None]) / (M[idx, t] - S[idx, t])[:, None])
        med = np.median(np.concatenate(us), 0)
        Dt.append(np.mean(abs(med - nat[a + k]))); Du.append(np.mean(abs(med - nat[a])))
print(f"   mean D(target) = {np.mean(Dt):.3f}, mean D(unshifted) = {np.mean(Du):.3f}, target closer in {np.mean(np.array(Dt) < np.array(Du)):.2f} of {len(Dt)} cells")

print("\n(ii) final state h_9 only, clipped edit: share of answers changed (plain edit in brackets)")
idx = np.nonzero((M[:, 9] > 0) & (a1[:, 9] >= 1))[0]
for name, F in EDITS:
    ac = (edit(H[idx, 9], F, True) @ U.T).argmax(1); ap = (edit(H[idx, 9], F, False) @ U.T).argmax(1)
    print(f"   {name:6s}: {np.mean(ac != ans0[idx]):.4f} ({np.mean(ap != ans0[idx]):.4f})   accuracy {np.mean(ac == true[idx]):.4f} ({np.mean(ap == true[idx]):.4f}; baseline {np.mean(ans0[idx] == true[idx]):.4f})")
