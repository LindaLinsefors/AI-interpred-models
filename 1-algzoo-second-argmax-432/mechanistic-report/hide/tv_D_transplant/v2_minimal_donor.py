"""v2: claim 2 (a tape built from M and S alone) with own code, and the attacks on it.

Artificial donor of a quiet state A at step t: zeros in the prefix except M at the maximum's position and S (or r M)
at the runner-up's position.  fw = rest of A + tape of the donor.
ATTACK 1 (control): rv = rest of the DONOR + tape of A.  If rv works as well as fw, "a function of (a_M, a_S, S/M)
          suffices" is a property of any part of the state, not of the tape.
ATTACK 3: is the real tape close to that function?  Distance between A's real tape and the donor's tape.
ATTACK 4 (control): delete other modes instead of the tape.
ATTACK 5: other populations: every step t = 3..9, and inputs uniform on [-1, 1].
"""
from v_common import *

N = 500_000
ar_ = lambda n: np.arange(n)


def donor(x, st, t, r="true", keep="MS"):
    xB = x.copy(); xB[:, : t + 1] = 0
    n = len(x)
    if "M" in keep:
        xB[ar_(n), st["pM"]] = st["M"]
    if "S" in keep:
        xB[ar_(n), st["pS"]] = st["S"] if r == "true" else r * st["M"]
    return xB


# ---------------------------------------------------------------- t = 9
t = 9
x, st = quiet(N, t, 8101)
hA = state(x, t)
n = len(x)
acc = lambda h, clip=False: np.mean(ans_at(h, t, clip) == st["pS"])
print(f"t = 9, {n} quiet final states; intact accuracy {acc(hA):.3f}")
print("   donor                 fw = rest A + tape donor [clipped]   rv = rest donor + tape A [clipped]   neg. entry fw / rv")
for lab, r, keep in (("M, S (true S)", "true", "MS"), ("M, 0.1 M", 0.1, "MS"), ("M, 0.2 M", 0.2, "MS"), ("M, 0.4 M", 0.4, "MS"), ("M, 0.6 M", 0.6, "MS"),
                     ("M, 0.8 M", 0.8, "MS"), ("M, 0.95 M", 0.95, "MS"), ("S only", "true", "S"), ("M only", "true", "M"), ("nothing (tape = 0)", "true", "")):
    hD = state(donor(x, st, t, r, keep), t)
    fw, rv = hyb(hA, hD), hyb(hD, hA)
    print(f"   {lab:20s}  {acc(fw):.3f} [{acc(fw, True):.3f}]                      {acc(rv):.3f} [{acc(rv, True):.3f}]"
          f"                    {np.mean((fw < -1e-9).any(1)):.3f} / {np.mean((rv < -1e-9).any(1)):.3f}")
hD = state(donor(x, st, t), t)
fw, rv = hyb(hA, hD), hyb(hD, hA)
hit_f, hit_r, hit_i = ans_at(fw, t) == st["pS"], ans_at(rv, t) == st["pS"], ans_at(hA, t) == st["pS"]
print("   M, S donor by a_S:      " + "  ".join(f"{s}: fw {hit_f[st['aS'] == s].mean():.3f} rv {hit_r[st['aS'] == s].mean():.3f} intact {hit_i[st['aS'] == s].mean():.3f}" for s in range(2, 10)))
cells = [(m, s) for m in range(2, 10) for s in range(2, 10) if m != s]
cf = np.array([hit_f[(st["aM"] == m) & (st["aS"] == s)].mean() for m, s in cells])
print(f"   M, S donor, 56 cells (a_M, a_S): fw median {np.median(cf):.3f}, range {cf.min():.3f} {cells[cf.argmin()]} to {cf.max():.3f}")
near = st["T"] > st["S"] - 0.1 * st["M"]
print(f"   third within 0.1 M of S (share {near.mean():.3f}): fw {hit_f[near].mean():.3f} intact {hit_i[near].mean():.3f};  further: fw {hit_f[~near].mean():.3f} intact {hit_i[~near].mean():.3f}")

# ATTACK 3: how close is the real tape to the donor's tape (a function of a_M, a_S, S/M times M)?
tA, tD = part(hA)[:, O], part(hD)[:, O]
rel = np.linalg.norm(tA - tD, axis=1) / np.linalg.norm(tA, axis=1)
sA, sD = (tA @ U[:, O].T)[:, :8] / st["M"][:, None], (tD @ U[:, O].T)[:, :8] / st["M"][:, None]
print(f"\nATTACK 3. real tape against the M,S-donor's tape, {n} quiet final states")
print(f"   |tape_A - tape_donor| / |tape_A| (13 units): median {np.median(rel):.2f}, quartiles {np.quantile(rel, .25):.2f}, {np.quantile(rel, .75):.2f}")
print(f"   tape piece of the scores, entries of age >= 2, in M: rms real {np.sqrt((sA ** 2).mean()):.1f}, rms donor {np.sqrt((sD ** 2).mean()):.1f},"
      f" rms difference {np.sqrt(((sA - sD) ** 2).mean()):.1f};  1 - SS(real - donor) / SS(real - entry mean) = {1 - ((sA - sD) ** 2).sum() / ((sA - sA.mean(0)) ** 2).sum():.3f}")
# best function of (a_M, a_S, S/M): per cell, each entry regressed on 1, S/M, (S/M)^2; fitted on one half, scored on the other
half = ar_(n) % 2 == 0
ratio = st["S"] / st["M"]
ss_res = ss_tot = ss_cell = 0.0
for m, s in cells:
    c = (st["aM"] == m) & (st["aS"] == s)
    X = np.stack([np.ones(c.sum()), ratio[c], ratio[c] ** 2], 1)
    tr, te = half[c], ~half[c]
    beta = np.linalg.lstsq(X[tr], sA[c][tr], rcond=None)[0]
    ss_res += ((sA[c][te] - X[te] @ beta) ** 2).sum()
    ss_cell += ((sA[c][te] - sA[c][tr].mean(0)) ** 2).sum()
ss_tot = ((sA[~half] - sA[half].mean(0)) ** 2).sum()
print(f"   held-out share of the variance of the real tape piece (ages >= 2, in M) explained by: the cell (a_M, a_S) alone {1 - ss_cell / ss_tot:.3f};"
      f" cell + quadratic in S/M {1 - ss_res / ss_tot:.3f}")
for lab, s in (("real tape piece", sA), ("donor's tape piece", sD)):
    d = s[ar_(n), st["pS"]] - s[ar_(n), st["pM"]]
    print(f"   {lab:19s}: S entry - M entry (in M) 10/50/90% = {np.quantile(d, .1):.1f} {np.median(d):.1f} {np.quantile(d, .9):.1f}")

# ATTACK 4: delete other modes instead of the tape
print("\nATTACK 4. deletion controls at t = 9 (accuracy on quiet final states; answer = maximum)")
for names in (TAPE, ["clock"], ["level"], ["runner"], ["trans", "echo"], ["alt"], ["short"], ["h2"], ["h3"]):
    hz = hA - part(hA, proj(names))
    print(f"   delete {'+'.join(names):18s}: accuracy {acc(hz):.3f}   maximum named {np.mean(ans_at(hz, t) == st['pM']):.3f}"
          f"   rms of the deleted piece of the scores / M: {np.sqrt(np.mean((((hA - hz) @ U.T) / st['M'][:, None]) ** 2)):.1f}")

# ---------------------------------------------------------------- t = 6 with futures
t = 6
x, st = quiet(N, t, 8201)
rng = np.random.default_rng(8202)
hA = state(x, t)
n, pre = len(x), x[:, : t + 1]
low = lows(st["M"], 3, rng)
put = lambda v, k: np.concatenate([low[:, :k], v[:, None], low[:, k + 1:]], 1)
futs = {"3 low": low, "new max @7": put(1.3 * st["M"], 0), "new max @9": put(1.3 * st["M"], 2), "new S @7": put((st["S"] + st["M"]) / 2, 0),
        "new third @7": put((np.maximum(st["T"], 0) + st["S"]) / 2, 0), "own inputs": x[:, t + 1:]}
print(f"\nt = 6, {n} quiet states, accuracy after chosen futures [clipped]")
print("   " + " " * 26 + "".join(f"{k:>17s}" for k in futs))
rows = {"intact": hA}
for lab, r, keep in (("fw: M, S (true)", "true", "MS"), ("fw: M, 0.6 M", 0.6, "MS"), ("fw: tape = 0", "true", "")):
    rows[lab] = hyb(hA, state(donor(x, st, t, r, keep), t))
rows["rv: rest of M,S donor"] = hyb(state(donor(x, st, t), t), hA)
rows["rv: rest of M,0.6M donor"] = hyb(state(donor(x, st, t, 0.6), t), hA)
for lab, hh in rows.items():
    row = f"   {lab:26s}"
    for f in futs.values():
        truth = np.argsort(np.concatenate([pre, f], 1), 1)[:, -2]
        row += f"  {np.mean(final(hh, f) == truth):.3f} [{np.mean(final(hh, f, True) == truth):.3f}]"
    print(row)

# ---------------------------------------------------------------- ATTACK 5: other populations
print("\nATTACK 5. other populations: S_pos follow rate and M,S-donor accuracy (running answer at step t)")
for dist in ("normal", "uniform"):
    for t in (3, 4, 5, 6, 7, 8, 9):
        x, st = quiet(300_000, t, 8300 + t, dist)
        rng = np.random.default_rng(8400 + t)
        hA = state(x, t)
        j, ok = pick_low_slot(x, st, t, rng)
        xB = swapped(x, st["pS"], j)
        sB = stats(xB, t)
        ok &= sB["quiet"]
        fw = hyb(hA[ok], state(xB[ok], t))
        an = ans_at(fw, t)
        fd = hyb(hA, state(donor(x, st, t), t))
        f6 = hyb(hA, state(donor(x, st, t, 0.6), t))
        print(f"   {dist:8s} t = {t}: quiet {len(x):6d};  S_pos n = {ok.sum():6d}: tape's S {np.mean(an == sB['pS'][ok]):.3f}, rest's S {np.mean(an == st['pS'][ok]):.3f}"
              f" [clipped {np.mean(ans_at(fw, t, True) == sB['pS'][ok]):.3f}];   intact {np.mean(ans_at(hA, t) == st['pS']):.3f},"
              f" M,S donor {np.mean(ans_at(fd, t) == st['pS']):.3f}, M,0.6M donor {np.mean(ans_at(f6, t) == st['pS']):.3f}")
