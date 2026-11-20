"""Verifier V5, part D: what the readout sees ON DATA (attack on M3 / M4 and on "the readout does not see the clock").

 D1  contribution of every mode to the logits of the final state h_9, weighted by the coordinates that occur on data
 D2  deleting / rescaling / rotating one mode in h_9 only (no dynamics afterwards): how often does the answer change?
 D3  cancellation of the clock's comparator and wave parts on data; all 715 ways to split the 13 units 4 + 9
All quantities here are invariant under per-unit rescaling h_i -> c_i h_i and under the gauge of the eigenvectors.
Seeds 5020, 5021 (2 x 200k sequences).   Run:  python -B v5_d_readout_on_data.py > v5_d_readout_on_data.txt
"""
from itertools import combinations
from v5_lib import *

SEEDS_D = [5020, 5021]
RO = {n: Uo @ RIGHT[n] for n in NAMES}                  # readout of each right vector (10,), complex for pairs
RO_c = {n: U[:, COMP] @ RIGHT[n][CI] for n in NAMES}
RO_w = {n: U[:, WAVE] @ RIGHT[n][WI] for n in NAMES}

def contrib(mk, R, cplx):
    """logit contribution (B, 10) of a mode with coordinates mk (B,) and readout vector R (10,)."""
    return 2 * np.real(mk[:, None] * R[None, :]) if cplx else np.real(mk[:, None] * R[None, :])

rows = []
for seed in SEEDS_D:
    x, H, Z, st, m = batch(seed)
    h9 = H[:, 9]
    lg = h9 @ U.T
    ans = lg.argmax(1)
    true = second_argmax(x)
    C = {n: contrib(m[n][:, 9], RO[n], IS_COMPLEX[n]) for n in NAMES}
    C["pulses"] = h9[:, PULSE] @ U[:, PULSE].T
    assert np.abs(sum(C.values()) - lg).max() < 1e-8
    rows.append(dict(x=x, lg=lg, ans=ans, true=true, C=C, m={n: m[n][:, 9] for n in NAMES}, M=st["M"][:, 9], S=st["S"][:, 9],
                     a1=st["a1"][:, 9], a2=st["a2"][:, 9], h9=h9,
                     Cc=contrib(m["clock"][:, 9], RO_c["clock"], True), Cw=contrib(m["clock"][:, 9], RO_w["clock"], True)))
cat = lambda k: np.concatenate([r[k] for r in rows])
lg, ans, true, M, S, a1, a2 = (cat(k) for k in ("lg", "ans", "true", "M", "S", "a1", "a2"))
C = {n: np.concatenate([r["C"][n] for r in rows]) for n in rows[0]["C"]}
mm = {n: np.concatenate([r["m"][n] for r in rows]) for n in NAMES}
Cc, Cw = cat("Cc"), cat("Cw")
B = len(ans)
print(f"seeds {SEEDS_D}, {NB} sequences each; accuracy at t=9: {np.mean(ans == true):.4f}")

rm = lambda v, k=10: v[:, :k] - v[:, :k].mean(1, keepdims=True)
nrm = lambda v: np.sqrt((v ** 2).sum(1))
rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
print("=" * 120)
print("D1  contribution of each mode to the logits of h_9 (units: logits).  'size 0-8' = norm over positions 0..8 after removing their mean (the M4 measure x coordinate)")
print("   mode     rms |coord|/w4   rms size 0-8   rms size 0-9   median size 0-8  |  share of sum of squares (0-8, mean removed; modes are not orthogonal, shares need not add to 1)")
tot8, tot10 = nrm(rm(lg, 9)), nrm(rm(lg))
for n in list(NAMES) + ["pulses"]:
    s8, s10 = nrm(rm(C[n], 9)), nrm(rm(C[n]))
    coord = f"{rms(np.abs(mm[n]) / W4):10.3f}" if n in mm else f"{'':10s}"
    print(f"   {n:7s} {coord}      {rms(s8):9.2f}      {rms(s10):9.2f}      {np.median(s8):9.2f}       |  {np.mean(s8 ** 2) / np.mean(tot8 ** 2):7.4f}")
print(f"   TOTAL   {'':10s}      {rms(tot8):9.2f}      {rms(tot10):9.2f}      {np.median(tot8):9.2f}")
print(f"   clock, comparator part only: rms size 0-8 = {rms(nrm(rm(Cc, 9))):.2f};  wave part only: {rms(nrm(rm(Cw, 9))):.2f};  both together: {rms(nrm(rm(C['clock'], 9))):.2f}")
ratio = nrm(C["clock"]) / (nrm(Cc) + nrm(Cw))
print(f"   per-state cancellation ratio |clock total| / (|comp part| + |wave part|) over 10 positions: median {np.median(ratio):.4f}, 5%-95%: {np.percentile(ratio, 5):.4f} - {np.percentile(ratio, 95):.4f}")

print("\n   margin = logit[true answer] - max logit elsewhere; contribution of each part to  logit[true] - logit[strongest competitor]:")
idx = np.arange(B)
comp_lg = lg.copy(); comp_lg[idx, true] = -np.inf
rival = comp_lg.argmax(1)
marg = lg[idx, true] - lg[idx, rival]
print(f"   total margin: mean {marg.mean():.2f}, median {np.median(marg):.2f}, 10th pct {np.percentile(marg, 10):.2f}")
print("   part      mean      rms     median   |  the same among sequences with (M - max(S,0))/M > 0.3 and a1 >= 2 at t = 9")
big = ((M - np.maximum(S, 0)) / np.maximum(M, 1e-9) > 0.3) & (a1 >= 2) & (M > 0.3)
for n in list(NAMES) + ["pulses"]:
    d = C[n][idx, true] - C[n][idx, rival]
    print(f"   {n:7s} {d.mean():8.2f} {rms(d):8.2f} {np.median(d):8.2f}   |  {d[big].mean():8.2f} {rms(d[big]):8.2f} {np.median(d[big]):8.2f}")
d = Cc[idx, true] - Cc[idx, rival]; print(f"   clock/comp part {d.mean():8.2f} {rms(d):8.2f} {np.median(d):8.2f}   |  {d[big].mean():8.2f} {rms(d[big]):8.2f} {np.median(d[big]):8.2f}")
d = Cw[idx, true] - Cw[idx, rival]; print(f"   clock/wave part {d.mean():8.2f} {rms(d):8.2f} {np.median(d):8.2f}   |  {d[big].mean():8.2f} {rms(d[big]):8.2f} {np.median(d[big]):8.2f}")
print(f"   margin (those sequences): mean {marg[big].mean():.2f} median {np.median(marg[big]):.2f};  fraction of sequences in this subset: {big.mean():.3f}")

print("=" * 120)
print("D2  intervention on h_9 only: remove / change ONE mode's component, then read out.  P(answer changes), accuracy after")
print(f"   baseline accuracy {np.mean(ans == true):.4f}")
print("   intervention                      P(answer changes)   accuracy   | subset 'big clock' ((M-S+)/M > 0.3, a1 >= 2, M > 0.3): P(changes), accuracy (baseline %.4f)" % np.mean(ans[big] == true[big]))
def report(name, new_lg):
    a = new_lg.argmax(1)
    print(f"   {name:32s}  {np.mean(a != ans):8.4f}          {np.mean(a == true):.4f}    |   {np.mean(a[big] != ans[big]):8.4f}   {np.mean(a[big] == true[big]):.4f}")
for n in NAMES:
    report(f"delete {n}", lg - C[n])
report("delete pulses (units 4,5,9)", lg - C["pulses"])
report("delete clock, comparator part", lg - Cc)
report("delete clock, wave part", lg - Cw)
zc = mm["clock"]
for f in (0.5, 1.5, 2.0, 3.0):
    report(f"clock x {f}", lg + contrib((f - 1) * zc, RO["clock"], True))
for k in (-2, -1, 1, 2, 4):
    report(f"clock rotated by {k:+d} steps", lg + contrib(zc * (np.exp(1j * k * OMEGA) - 1), RO["clock"], True))
report("clock conjugated (z -> conj z)", lg + contrib(np.conj(zc) - zc, RO["clock"], True))
for n in ("level", "runner", "h2", "short"):
    report(f"{n} x 2", lg + C[n])
# what does the answer change TO when the clock is deleted?
a = (lg - C["clock"]).argmax(1)
ch = a != ans
pM = np.concatenate([r["x"] for r in rows]).argmax(1)
print(f"\n   when deleting the clock changes the answer ({ch.mean():.4f} of sequences): was correct before {np.mean(ans[ch] == true[ch]):.3f}, correct after {np.mean(a[ch] == true[ch]):.3f},"
      f" new answer = position of the max {np.mean(a[ch] == pM[ch]):.3f}; median baseline margin of those {np.median(marg[ch]):.2f} (all: {np.median(marg):.2f})")

print("   by age of the max at t = 9 (a1 = 9 - position of the max): P(answer changes | clock deleted), accuracy before -> after, and for comparison P(changes | clock x 2)")
a2x = (lg + C["clock"]).argmax(1)
for A1 in range(0, 10):
    s = a1 == A1
    print(f"      a1={A1}: {np.mean(a[s] != ans[s]):.4f}   {np.mean(ans[s] == true[s]):.4f} -> {np.mean(a[s] == true[s]):.4f}   x2: {np.mean(a2x[s] != ans[s]):.4f}   (share of sequences {s.mean():.3f})")

print("=" * 120)
print("D3  is the comparator/wave split special?  ratio |U_o v| / (|U[:,G] v_G| + |U[:,G'] v_G'|) for the clock vector, all 715 subsets G of 4 ordinary units")
r = RIGHT["clock"]
nn = np.linalg.norm
tot = nn(Uo @ r)
vals = []
for G in combinations(range(13), 4):
    Gc = [i for i in range(13) if i not in G]
    vals.append((tot / (nn(Uo[:, list(G)] @ r[list(G)]) + nn(Uo[:, Gc] @ r[Gc])), [ORD[i] for i in G]))
vals.sort()
ratios = np.array([v[0] for v in vals])
mine = [v for v in vals if sorted(v[1]) == sorted(COMP)][0]
print(f"   comparators {COMP}: ratio {mine[0]:.4f};  rank {int((ratios < mine[0]).sum()) + 1} of {len(vals)} (smallest ratio = strongest cancellation)")
print(f"   quantiles of the ratio over all 4-subsets: min {ratios.min():.4f}, 5% {np.quantile(ratios, .05):.4f}, median {np.median(ratios):.4f}, max {ratios.max():.4f}")
print("   five strongest:", [(round(v[0], 4), v[1]) for v in vals[:5]])
print("   per-unit pieces |U[:, i] v_i| (10-vector norms) of the clock vector, by unit:", {ORD[i]: round(float(nn(Uo[:, i] * r[i])), 3) for i in range(13)})
print(f"   |U_o v| = {tot:.3f};  sum over units of |U[:, i] v_i| = {sum(nn(Uo[:, i] * r[i]) for i in range(13)):.3f};"
      f"  within comparators: |sum| / sum|.| = {nn(U[:, COMP] @ r[CI]) / sum(nn(U[:, j] * r[ORD.index(j)]) for j in COMP):.3f};"
      f"  within wave units: {nn(U[:, WAVE] @ r[WI]) / sum(nn(U[:, j] * r[ORD.index(j)]) for j in WAVE):.3f}")
