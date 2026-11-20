"""C5. Causal tests of 'clock phase = age of the max'.

Edit at step t: clock coordinate z -> F z (F = exp(i k omega) for a phase shift by k steps, F = f for a radius change);
all other mode coordinates and the three pulse units unchanged; h_t[ORD] += 2 Re((F - 1) z RIGHT_clock); the run then
continues with the real inputs x_{t+1}.. .   200k N(0,1) sequences, seed 1500.  States edited: M_t > 0 and a1(t) >= 1."""
from cl import *

NS, SEED = 200_000, 1500
x, H, Z, st = gen(NS, SEED)
M, S, a1, a2, p1, p2 = st["M1"], pos(st["M2"]), st["a1"], st["a2"], st["p1"], st["p2"]
true = second_argmax(x)
ans0 = logits(H).argmax(1)
rank = (-x).argsort(1).argsort(1) + 1                          # rank[b, q] = 1 for the largest input
print(f"baseline accuracy (n={NS}, seed {SEED}): {np.mean(ans0 == true):.4f}")
nat = np.load(HERE / "c4_thresholds.npz")["med_u"]             # natural median positions by age a = 0..8 (units 1,6,7,8,4)
UN = COMP + [4]


def edit(h, F):
    h2 = h.copy()
    zc = h[:, ORD] @ LC
    h2[:, ORD] += 2 * np.real(np.multiply.outer((F - 1) * zc, RC))
    return h2


def cont(xs, t, h0):
    """Continue from the edited state h0 at step t. Returns states of steps t+1..9 (B, 9-t, 16) and pre-activations."""
    if t == 9:
        return np.empty((h0.shape[0], 0, 16)), np.empty((h0.shape[0], 0, 16))
    return run(xs[:, t + 1:], h0=h0, return_pre=True)


EDITS = [(f"k={k:+d}", np.exp(1j * k * OMEGA)) for k in (-2, -1, 0, 1, 2)] + [(f"f={f}", f) for f in (0, 0.5, 2)]
res = {name: [] for name, _ in EDITS}                # per edit: list over t of dicts
for t in range(1, 9):
    sel = (M[:, t] > 0) & (a1[:, t] >= 1)
    idx = np.nonzero(sel)[0]
    h = H[idx, t]
    xs = x[idx]
    for name, F in EDITS:
        h2 = edit(h, F)
        H2, Z2 = cont(xs, t, h2)
        ans = (H2[:, -1] @ U.T).argmax(1)
        th2 = thresholds(h2)
        u = (th2 - S[idx, t, None]) / (M[idx, t] - S[idx, t])[:, None]
        # thresholds at later steps (states t+d), positions w.r.t. the order statistics at step t
        ulater = [(thresholds(H2[:, d - 1]) - S[idx, t, None]) / (M[idx, t] - S[idx, t])[:, None] for d in (1, 2) if t + d <= 8]
        res[name].append(dict(t=t, idx=idx, ans=ans, u=u, ulater=ulater, neg=(h2[:, ORD] < -1e-9).any(1), pre1=Z2[:, 0], h2=h2))

# ---------------------------------------------------------------- sanity
same = all(np.array_equal(r["ans"], ans0[r["idx"]]) for r in res["k=+0"])
print("check: the k = 0 edit reproduces every baseline answer:", same)
print("edited states with at least one negative ordinary unit: " + ", ".join(f"{n}: {np.concatenate([r['neg'] for r in res[n]]).mean():.3f}" for n, _ in EDITS))

# ---------------------------------------------------------------- (i) thresholds after a phase shift
print("\n=== (i) thresholds for x_{t+1} after a phase shift by k steps at max-age a: median positions (theta_j - S+)/(M - S+) of units 1,6,7,8,4 ===")
print("   compared with the natural medians at age a + k ('target') and at age a ('unshifted'); D = mean over the five units of |difference of medians|")
print("   a  k | edited medians (1,6,7,8,4)            | natural at a+k                        | D(target)  D(unshifted)")
Dt, Du = [], []
for a in range(1, 9):
    for k in (-2, -1, 1, 2):
        if not (1 <= a + k <= 8):
            continue
        us = np.concatenate([r["u"][a1[r["idx"], r["t"]] == a] for r in res[f"k={k:+d}"]])
        med = np.median(us, 0)
        dt, du = np.mean(abs(med - nat[a + k])), np.mean(abs(med - nat[a]))
        Dt.append(dt); Du.append(du)
        print(f"   {a} {k:+d} | " + " ".join(f"{v:+.3f}" for v in med) + " | " + " ".join(f"{v:+.3f}" for v in nat[a + k]) + f" | {dt:.3f}  {du:.3f}")
print(f"   over all (a, k): mean D(target) = {np.mean(Dt):.3f}, mean D(unshifted) = {np.mean(Du):.3f};  D(target) < D(unshifted) in {np.mean(np.array(Dt) < np.array(Du)):.2f} of the {len(Dt)} cells")
print("   same one and two steps later (states t+1, t+2 of the edited run; inputs in between below S+_t - 0.3 so that nothing happens in either run): D(target = natural age a+k+d), D(unshifted = natural age a+d)")
for d in (1, 2):
    Dt, Du = [], []
    for a in range(1, 9):
        for k in (-2, -1, 1, 2):
            if not (1 <= a + k + d <= 8) or a + d > 8:
                continue
            chunks = []
            for r in res[f"k={k:+d}"]:
                t = r["t"]
                if t + d > 8:
                    continue
                q = (a1[r["idx"], t] == a)
                for dd in range(1, d + 1):
                    q &= x[r["idx"], t + dd] < S[r["idx"], t] - 0.3
                chunks.append(r["ulater"][d - 1][q])
            us = np.concatenate(chunks)
            if us.shape[0] < 300:
                continue
            med = np.median(us, 0)
            Dt.append(np.mean(abs(med - nat[a + k + d]))); Du.append(np.mean(abs(med - nat[a + d])))
    print(f"     d = {d}: mean D(target) = {np.mean(Dt):.3f}, mean D(unshifted) = {np.mean(Du):.3f}  ({len(Dt)} cells)")

# ---------------------------------------------------------------- answers
def answer_table(names, title):
    print(f"\n=== {title} ===")
    print("   edit   | n edited runs | answer changed | accuracy after (before) | among changed: new answer is true runner-up / the max / rank 3 / rank >= 4 | new answer arrived after t | old answer was correct")
    for n in names:
        idx = np.concatenate([r["idx"] for r in res[n]]); ans = np.concatenate([r["ans"] for r in res[n]])
        tt = np.concatenate([np.full(r["idx"].size, r["t"]) for r in res[n]])
        ch = ans != ans0[idx]
        rk = rank[idx, ans]
        print(f"   {n:6s} | {idx.size:8d} | {ch.mean():.4f} | {np.mean(ans == true[idx]):.4f} ({np.mean(ans0[idx] == true[idx]):.4f}) | "
              + (f"{np.mean(rk[ch] == 2):.3f} / {np.mean(rk[ch] == 1):.3f} / {np.mean(rk[ch] == 3):.3f} / {np.mean(rk[ch] >= 4):.3f} | {np.mean(ans[ch] > tt[ch]):.3f} | {np.mean(ans0[idx][ch] == true[idx][ch]):.3f}" if ch.any() else "-"))

answer_table([n for n, _ in EDITS], "(i)+(iii) final answers after an edit at step t (t = 1..8 pooled, states with M > 0, a1 >= 1)")

print("\n   share of answers changed, by max-age a at the edit (rows) and edit (columns)")
print("    a  |      n  | " + "  ".join(f"{n:>6s}" for n, _ in EDITS))
for a in range(1, 9):
    row = []
    for n, _ in EDITS:
        idx = np.concatenate([r["idx"][a1[r["idx"], r["t"]] == a] for r in res[n]]); ans = np.concatenate([r["ans"][a1[r["idx"], r["t"]] == a] for r in res[n]])
        row.append(f"{np.mean(ans != ans0[idx]):6.3f}")
    print(f"    {a}  | {idx.size:7d} | " + "  ".join(row))
print("   share of answers changed, by step t of the edit")
print("    t  |      n  | " + "  ".join(f"{n:>6s}" for n, _ in EDITS))
for ti, t in enumerate(range(1, 9)):
    print(f"    {t}  | {res['k=+0'][ti]['idx'].size:7d} | " + "  ".join(f"{np.mean(res[n][ti]['ans'] != ans0[res[n][ti]['idx']]):6.3f}" for n, _ in EDITS))
print("   share of answers changed, by clock quality (M - S+)/M at the edit")
for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 1.01)):
    row = []
    for n, _ in EDITS:
        idx = np.concatenate([r["idx"] for r in res[n]]); ans = np.concatenate([r["ans"] for r in res[n]]); tt = np.concatenate([np.full(r["idx"].size, r["t"]) for r in res[n]])
        ratio = (M[idx, tt] - S[idx, tt]) / M[idx, tt]
        q = (ratio >= lo) & (ratio < hi)
        row.append(f"{np.mean(ans[q] != ans0[idx][q]):6.3f}")
    print(f"    ratio [{lo:.2f},{min(hi, 1):.2f}) | " + "  ".join(row))

print("\n   where do changed answers go?  offset of the new answer from the max position p1 / from the true runner-up position p2 / from the old answer (final order statistics); share of changed answers")
for n in ("k=-2", "k=-1", "k=+1", "k=+2", "f=0", "f=0.5", "f=2"):
    idx = np.concatenate([r["idx"] for r in res[n]]); ans = np.concatenate([r["ans"] for r in res[n]])
    ch = ans != ans0[idx]
    P1, P2 = x[idx].argmax(1), true[idx]
    for lab, ref in (("new - p1", P1), ("new - p2", P2), ("new - old", ans0[idx])):
        off = (ans - ref)[ch]
        print(f"    {n:6s} {lab:9s}: " + "  ".join(f"{o:+d}:{np.mean(off == o):.3f}" for o in range(-4, 5)))

# ---------------------------------------------------------------- events at the first step after the edit
print("\n=== events at step t+1 after the edit, by the true class of x_{t+1} (record: x > M_t; runner-up: S+_t < x < M_t; low: x < S+_t) ===")
print("   edit   | record: P(unit 4 on)  P(all 4 comp. clipped) | runner-up: P(unit 4 on)  P(no comp. clipped)  mean #clipped | low: P(unit 4 on)  P(any comp. clipped)")
base_pre = {t: Z[:, t + 1] for t in range(1, 9)}
def ev_row(name, pre_of):
    acc = {c: [] for c in ("rec", "ru", "low")}
    for ti, t in enumerate(range(1, 9)):
        r = res["k=+0"][ti]; idx = r["idx"]
        pre = pre_of(ti, t, idx)
        xn = x[idx, t + 1]
        cls = {"rec": xn > M[idx, t], "ru": (xn > S[idx, t]) & (xn < M[idx, t]), "low": xn < S[idx, t]}
        for c in cls:
            acc[c].append(pre[cls[c]])
    P = {c: np.concatenate(v) for c, v in acc.items()}
    f4 = lambda A: np.mean(A[:, 4] > 0); ncl = lambda A: (A[:, COMP] <= 0).sum(1)
    print(f"   {name:8s} | {f4(P['rec']):.4f}  {np.mean(ncl(P['rec']) == 4):.4f} | {f4(P['ru']):.4f}  {np.mean(ncl(P['ru']) == 0):.4f}  {ncl(P['ru']).mean():.3f} | {f4(P['low']):.4f}  {np.mean(ncl(P['low']) > 0):.4f}")
ev_row("baseline", lambda ti, t, idx: Z[idx, t + 1])
for n, _ in EDITS:
    ev_row(n, lambda ti, t, idx, n=n: res[n][ti]["pre1"])

# ---------------------------------------------------------------- (iii) thresholds after scaling
print("\n=== (iii) thresholds for x_{t+1} after scaling the clock radius by f at max-age a: median positions of units 1,6,7,8,4 (natural medians for comparison) ===")
for a in range(1, 9):
    row = [f"natural: " + " ".join(f"{v:+.2f}" for v in nat[a])]
    for n in ("f=0", "f=0.5", "f=2"):
        us = np.concatenate([r["u"][a1[r["idx"], r["t"]] == a] for r in res[n]])
        row.append(f"{n}: " + " ".join(f"{v:+.2f}" for v in np.median(us, 0)))
    print(f"   a={a} | " + " | ".join(row))
print("   predicted position for f = 0 from the level+runner part of the C4 formula (A_j + small S term): the oscillating part vanishes, all five thresholds sit at 0.6-0.85 of the way from S+ to M")

# ---------------------------------------------------------------- (ii) edits of the final state only
print("\n=== (ii) the same edits applied to the final state h_9 only (readout follows immediately) ===")
sel = (M[:, 9] > 0) & (a1[:, 9] >= 1)
idx = np.nonzero(sel)[0]
h = H[idx, 9]
lg0 = h @ U.T
srt = np.sort(lg0, 1); margin = srt[:, -1] - srt[:, -2]
print(f"   n = {idx.size} final states (M > 0, max-age >= 1);  baseline logit margin (top - second): median {np.median(margin):.2f}")
for name, F in EDITS + [(f"k={k:+d}", np.exp(1j * k * OMEGA)) for k in (-4, 4)] + [("phase+180", -1.0)]:
    lg = edit(h, F) @ U.T
    ans = lg.argmax(1)
    dl = np.abs(lg - lg0).max(1)
    print(f"   {name:10s}: answers changed {np.mean(ans != ans0[idx]):.5f}   accuracy {np.mean(ans == true[idx]):.4f} (baseline {np.mean(ans0[idx] == true[idx]):.4f})   max |logit change| median {np.median(dl):.3f}, 95th pct {np.percentile(dl, 95):.3f}")
print("   for scale: the same phase shifts applied to h_9 through the comparators only (clock part of the comparators rotated, wave units untouched):")
for k in (-2, -1, 1, 2):
    F = np.exp(1j * k * OMEGA)
    zc = h[:, ORD] @ LC
    d = 2 * np.real(np.multiply.outer((F - 1) * zc, RC))
    d[:, wi] = 0
    h2 = h.copy(); h2[:, ORD] += d
    ans = (h2 @ U.T).argmax(1)
    print(f"     k={k:+d}: answers changed {np.mean(ans != ans0[idx]):.4f}")

# ---------------------------------------------------------------- (ii) continued: where do the answers go when only h_9 is edited
print("   destinations of the changed answers when only h_9 is edited: new answer is the true runner-up / the max / rank 3 / rank >= 4;  offsets new - old")
for name, F in [(f"k={k:+d}", np.exp(1j * k * OMEGA)) for k in (-2, -1, 1, 2)] + [("f=0", 0), ("f=2", 2)]:
    ans = (edit(h, F) @ U.T).argmax(1)
    ch = ans != ans0[idx]
    rk = rank[idx, ans][ch]
    off = (ans - ans0[idx])[ch]
    print(f"     {name:6s}: {np.mean(rk == 2):.3f} / {np.mean(rk == 1):.3f} / {np.mean(rk == 3):.3f} / {np.mean(rk >= 4):.3f}   offsets " + "  ".join(f"{o:+d}:{np.mean(off == o):.3f}" for o in range(-3, 4)))
print("   clock readout vector 2 U[q,ORD].RIGHT_clock per position q (|.|@deg), and the phase at which a max sitting at q would be read: phi(9 - q) + that phase")
phi3 = np.degrees(np.load(HERE / "c1_laws.npz")["phi3"])
URc = 2 * U[:, ORD] @ RC
print("     " + "  ".join(f"q{q}: {abs(URc[q]):.2f}@{deg(URc[q]):+.0f}" for q in range(10)))
print("     cos(phi(9-q) + arg) for q = 0..8: " + "  ".join(f"{np.cos(np.radians(phi3[9 - q]) + np.angle(URc[q])):+.2f}" for q in range(9)))

# ---------------------------------------------------------------- (i) continued: which later history matters
print("\n=== (i) continued: answer changes split by what happens after the edit step t ===")
print("   class A: nothing above S+_t arrives after t;  B: a new runner-up arrives after t, no new max;  C: a new max arrives after t")
print("   edit   class |      n   | changed | among changed: new - old = -k | new = max | new = true runner-up | new arrived after t")
for n in ("k=-2", "k=-1", "k=+1", "k=+2", "f=0", "f=0.5", "f=2"):
    k = int(n[2:]) if n.startswith("k") else 0
    idx_ = np.concatenate([r["idx"] for r in res[n]]); ans = np.concatenate([r["ans"] for r in res[n]])
    tt = np.concatenate([np.full(r["idx"].size, r["t"]) for r in res[n]])
    xm =np.where(np.arange(T)[None, :] > tt[:, None], x[idx_], -np.inf).max(1)
    cls = np.where(xm > M[idx_, tt], "C", np.where(xm > S[idx_, tt], "B", "A"))
    ch = ans != ans0[idx_]
    for c in "ABC":
        q = cls == c
        qc = q & ch
        print(f"   {n:6s}   {c}   | {q.sum():8d} | {ch[q].mean():.4f}  | {np.mean((ans - ans0[idx_])[qc] == -k) if k else float('nan'):.3f} | {np.mean(rank[idx_, ans][qc] == 1):.3f} | {np.mean(rank[idx_, ans][qc] == 2):.3f} | {np.mean(ans[qc] > tt[qc]):.3f}")
