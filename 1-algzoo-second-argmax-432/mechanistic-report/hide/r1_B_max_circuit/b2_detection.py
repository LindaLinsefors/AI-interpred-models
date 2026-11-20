"""B2. Detection quality of unit 4 as a 'new maximum' detector.
Data: N(0,1), seeds 1020, 1021 (400k each). Mp = max(0, x_0..x_{t-1}) (positive part; 0 at t = 0).
"""
from blib import *

SEEDS = (1020, 1021)
X, ON, H4, MP, MT, SP, AG, TT, TH, ERR = [], [], [], [], [], [], [], [], [], []
for seed in SEEDS:
    x = sample(400_000, seed)
    H, Z = run(x, return_pre=True)
    st = prefix_stats(x)
    on = Z[:, :, 4] > 0
    Mtrue = st["M1prev"]                                  # -inf at t = 0
    Mp = pos(np.where(np.isfinite(Mtrue), Mtrue, 0))
    Sprev = st["M2prev"]
    Sp = pos(np.where(np.isfinite(Sprev), Sprev, 0))
    th = np.concatenate([np.zeros((len(x), 1)), theta4(H[:, :-1])], 1)     # theta_4(0) = 0 exactly (h_{-1} = 0)
    X.append(x); ON.append(on); H4.append(H[:, :, 4]); MP.append(Mp); MT.append(Mtrue); SP.append(Sp)
    AG.append(st["a1prev"]); TT.append(np.broadcast_to(np.arange(10), x.shape)); TH.append(th)
    ERR.append(logits(H).argmax(1) != second_argmax(x))
x, on, h4, Mp, Mt, Sp, age, t, th = map(np.concatenate, (X, ON, H4, MP, MT, SP, AG, TT, TH))
err = np.concatenate(ERR)
n = len(x)
print(f"[MEASURED n = {n} sequences x 10 steps, N(0,1), seeds {SEEDS}]")
assert np.allclose(h4, pos(W4 * (x - th)), atol=1e-9), "h4 = 10.157*ReLU(x - theta4) must hold exactly"
print("check passed: h4 == 10.157 * ReLU(x_t - theta_4(t)) to 1e-9 on all states (F2 for unit 4)")

rec_p = x > Mp                    # new max with positive part (the first positive input counts)
rec_t = x > Mt                    # true new max (includes negative records and t = 0)
print("\n--- overall confusion (all t = 0..9) ---")
for name, rec in (("x_t > max(0, M_{t-1})  [positive-part record]", rec_p), ("x_t > M_{t-1}  [true record, t=0 counts]", rec_t)):
    print(f"  {name}: P(on | record) = {on[rec].mean():.5f} (n={rec.sum()}),  P(on | not record) = {on[~rec].mean():.5f} (n={(~rec).sum()})")
    print(f"      misses = {(rec & ~on).sum()}, false firings = {(~rec & on).sum()}, P(record | on) = {rec[on].mean():.5f}")
negrec = rec_t & (x <= 0)
print(f"  true records with x_t <= 0: n = {negrec.sum()}, P(on) = {on[negrec].mean():.6f}")
print(f"  positive-part records are missed only if theta_4 > x_t > Mp: misses n = {(rec_p & ~on).sum()}; "
      f"largest missed x_t - Mp = {((x - Mp)[rec_p & ~on]).max() if (rec_p & ~on).any() else 0:.4f}; of them with Mp = 0: {((rec_p & ~on) & (Mp == 0)).sum()}")

print("\n--- by step t ---")
print("t | P(on) | P(record+) | P(on | record+) | P(on | not record+) | P(record+ | on)")
for tt in range(10):
    s = t == tt
    print(f"{tt} | {on[s].mean():.4f} | {rec_p[s].mean():.4f} | {on[s & rec_p].mean():.5f} | {on[s & ~rec_p].mean():.5f} | {rec_p[s & on].mean():.4f}")

print("\n--- false firings by age of the standing max (age at t-1), states with Mp > 0 and x_t <= Mp ---")
print("age | n not-record | P(on) | P(on | x_t > S+) | P(on | x_t <= S+) | share of non-records with x_t > S+")
for a in range(9):
    s = (age == a) & (Mp > 0) & ~rec_p
    above = x > Sp
    print(f"{a} | {s.sum()} | {on[s].mean():.5f} | {on[s & above].mean():.5f} | {on[s & ~above].mean():.6f} | {above[s].mean():.4f}")

print("\n--- P(on) by relative margin u = (x_t - Mp)/Mp (rows: age of the max at t-1; Mp > 0) ---")
edges = [-np.inf, -0.5, -0.3, -0.2, -0.1, -0.05, -0.02, -0.01, 0, 0.01, 0.02, 0.05, np.inf]
lab = ["<-.5", "[-.5,-.3)", "[-.3,-.2)", "[-.2,-.1)", "[-.1,-.05)", "[-.05,-.02)", "[-.02,-.01)", "[-.01,0)", "[0,.01)", "[.01,.02)", "[.02,.05)", ">=.05"]
u = (x - Mp) / np.where(Mp > 0, Mp, 1)
print("age | " + " | ".join(lab))
for a in range(9):
    row = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (age == a) & (Mp > 0) & (u >= lo) & (u < hi)
        row.append(f"{on[s].mean():.3f}" if s.sum() >= 100 else "-")
    print(f"{a} | " + " | ".join(row))
print("n per cell:")
for a in range(9):
    print(f"{a} | " + " | ".join(str(((age == a) & (Mp > 0) & (u >= lo) & (u < hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])))

print("\n--- P(on) by position inside the gap, v = (x_t - S+)/(Mp - S+), non-records with x_t > S+ (new runner-up), Mp > 0 ---")
ve = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0000001]
vl = ["[0,.5)", "[.5,.6)", "[.6,.7)", "[.7,.8)", "[.8,.9)", "[.9,.95)", "[.95,1]"]
v = (x - Sp) / np.where(Mp - Sp > 0, Mp - Sp, 1)
print("age | " + " | ".join(vl) + " | v at which P(on)=0.5 (median of theta-implied v*)")
vstar = (th - Sp) / np.where(Mp - Sp > 0, Mp - Sp, 1)
for a in range(9):
    row = []
    base = (age == a) & (Mp > 0) & ~rec_p & (x > Sp)
    for lo, hi in zip(ve[:-1], ve[1:]):
        s = base & (v >= lo) & (v < hi)
        row.append(f"{on[s].mean():.3f}" if s.sum() >= 100 else "-")
    sa = (age == a) & (Mp > 0)
    print(f"{a} | " + " | ".join(row) + f" | {np.median(vstar[sa]):.3f} [IQR {np.percentile(vstar[sa],25):.3f}, {np.percentile(vstar[sa],75):.3f}]")

print("\n--- P(on) by relative gap of the standing max g = (Mp - S+)/Mp, non-records (x_t <= Mp), Mp > 0 ---")
ge = [0, 0.1, 0.25, 0.5, 0.75, 0.999999, 1.0000001]
gl = ["[0,.1)", "[.1,.25)", "[.25,.5)", "[.5,.75)", "[.75,1)", "S<=0"]
g = (Mp - Sp) / np.where(Mp > 0, Mp, 1)
print("age | " + " | ".join(gl))
for a in range(9):
    row = []
    for lo, hi in zip(ge[:-1], ge[1:]):
        s = (age == a) & (Mp > 0) & ~rec_p & (g >= lo) & (g < hi)
        row.append(f"{on[s].mean():.4f}" if s.sum() >= 100 else "-")
    print(f"{a} | " + " | ".join(row))

print("\n--- h4 against 10.157 * (x_t - Mp) ---")
tgt = W4 * (x - Mp)
def fitline(y, xx):
    b0, r0u, r0c = slope0(y, [xx])
    A_ = np.column_stack([xx, np.ones_like(xx)])
    b, *_ = np.linalg.lstsq(A_, y, rcond=None)
    r = 1 - (y - A_ @ b).var() / y.var()
    return b0[0], r0c, b[0], b[1], r
print("subset | n | slope (no intercept) | R2 (no intercept, centred) | slope, intercept (affine) | R2 (affine)")
for name, s in (("positive-part records (x_t > Mp), all t", rec_p),
                ("records with Mp = 0 (first positive input)", rec_p & (Mp == 0)),
                ("records with Mp > 0", rec_p & (Mp > 0)),
                ("all states, target 10.157*ReLU(x_t - Mp)", np.ones_like(on))):
    y = h4[s]; xx = tgt[s] if "all states" not in name else pos(tgt[s])
    b0, r0, b1, c1, r1 = fitline(y, xx)
    print(f"{name} | {s.sum()} | {b0:.4f} | {r0:.5f} | {b1:.4f}, {c1:.4f} | {r1:.5f}")
print("by age of the previous max (records with Mp > 0): slope/R2 and the excess h4/10.157 - (x_t - Mp) = Mp - theta_4")
print("age | n | slope0 | R2 | affine slope, intercept | R2 | mean excess | sd excess | mean excess / Mp")
for a in range(9):
    s = rec_p & (Mp > 0) & (age == a)
    b0, r0, b1, c1, r1 = fitline(h4[s], tgt[s])
    ex = h4[s] / W4 - (x[s] - Mp[s])
    print(f"{a} | {s.sum()} | {b0:.4f} | {r0:.5f} | {b1:.4f}, {c1:.4f} | {r1:.5f} | {ex.mean():.4f} | {ex.std():.4f} | {(ex*Mp[s]).sum()/(Mp[s]**2).sum():.4f}")

print("\n--- size of false firings: h4/(10.157*Mp) for not-record & on ---")
ff = ~rec_p & on
r = h4[ff] / (W4 * Mp[ff])
print(f"  n = {ff.sum()}; quantiles of h4/(10.157 Mp): 50% {np.median(r):.4f}, 90% {np.percentile(r,90):.4f}, 99% {np.percentile(r,99):.4f}, max {r.max():.4f}")
r2 = h4[rec_p & on & (Mp > 0)] / (W4 * Mp[rec_p & on & (Mp > 0)])
print(f"  true records (Mp > 0): quantiles of h4/(10.157 Mp): 10% {np.percentile(r2,10):.4f}, 50% {np.median(r2):.4f}, 90% {np.percentile(r2,90):.4f}")

print("\n--- do false firings cost accuracy? per sequence ---")
anyff = ff.any(1)
anymiss = (rec_p & ~on).any(1)
print(f"  sequences with >= 1 false firing: {anyff.mean():.4f};  error rate with / without: {err[anyff].mean():.4f} / {err[~anyff].mean():.4f};  overall error {err.mean():.4f}")
print(f"  sequences with >= 1 missed positive-part record: {anymiss.mean():.6f}; error rate among them: {err[anymiss].mean() if anymiss.any() else float('nan'):.4f}")
# false firings where x_t is the final runner-up or not
sa = second_argmax(x)
ffpos = np.where(ff, np.arange(10)[None, :], -1)
is_final_second = (ff & (np.arange(10)[None, :] == sa[:, None])).any(1)
print(f"  false firing at the position of the final 2nd largest: {is_final_second.mean():.4f} of sequences; error rate there: {err[is_final_second].mean():.4f}")
other = anyff & ~is_final_second
print(f"  false firing only at other positions: {other.mean():.4f}; error rate there: {err[other].mean():.4f}")
