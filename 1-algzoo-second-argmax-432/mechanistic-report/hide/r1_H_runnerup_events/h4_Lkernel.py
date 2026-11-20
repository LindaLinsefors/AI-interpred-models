"""H4 part 4: what a class-L record (x_t just below the runner-up, unit 5 fires at t+1) does to the final logits.

Single-event intervention: for one L event per sequence, force h_{t+1}[5] := 0 and compare the final logits.
"""
from hcommon import *
from h4_intervention import run_force
from h3_kernels_exact import Kp

cols = {}
for seed in (1016, 1017):
    x = sample(400_000, seed)
    e = Ev(x)
    L0 = logits(e.H)
    y = second_argmax(x)
    for t in range(2, 9):
        S, M = e.So[:, t], e.Mo[:, t]
        sel = (S > 0) & (x[:, t] < S) & (x[:, t] >= 0.5 * S) & e.G[:, t + 1, 5]
        idx = np.nonzero(sel)[0]
        mask = np.zeros((len(idx), T), bool); mask[:, t + 1] = True
        Lf = logits(run_force(x[idx], mask))
        dL = L0[idx] - Lf                                   # effect of the pulse
        p1, p2 = e.p1o[idx, t], e.p2o[idx, t]
        ar = np.arange(len(idx))
        rec = dict(t=np.full(len(idx), t), a=e.a1[idx, t], ratio=x[idx, t] / S[idx], S=S[idx], M=M[idx], h5=e.H[idx, t + 1, 5],
                   d_own=dL[:, t], d_p2=dL[ar, p2], d_p1=dL[ar, p1], d_next=dL[:, t + 1], d_prev=dL[:, t - 1],
                   k_own=Kp[5][9 - t - 1][t] * e.H[idx, t + 1, 5],            # exact direct kernel of the pulse at its own position
                   k_p2=Kp[5][9 - t - 1][p2] * e.H[idx, t + 1, 5],
                   marg=L0[idx, t] - L0[ar + 0, :][ar, p2] if False else L0[idx][ar, t] - L0[idx][ar, p2],
                   marg_f=Lf[ar, t] - Lf[ar, p2], ans_t=L0[idx].argmax(1) == t, ansf_t=Lf.argmax(1) == t,
                   ok=L0[idx].argmax(1) == y[idx], okf=Lf.argmax(1) == y[idx], S_final=(y[idx] == p2),
                   qmax=x[idx, t + 1:].max(1), th5=np.zeros(len(idx)))
        for k, v in rec.items():
            cols.setdefault(k, []).append(v)
    del e
d = {k: np.concatenate(v) for k, v in cols.items()}
n = len(d["t"])
print(f"L events with unit 5 on at t+1 (S_old>0, 0.5<=x_t/S<1, 2<=t<=8): n = {n} from 800k sequences (seeds 1016, 1017)")
RB = [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
print("\n ratio x_t/S | n | mean h5 | mean h5/(K S) | effect of the pulse on the final logits, mean of (dlogit / h5): own pos, t+1, t-1, runner-up pos, max pos | exact K5 kernel own (mean per h5)")
for i in range(len(RB) - 1):
    s = (d["ratio"] >= RB[i]) & (d["ratio"] < RB[i + 1])
    f = lambda v: np.sum(v[s]) / np.sum(d["h5"][s])        # pulse-weighted mean per unit h5
    print(f"  {RB[i]:.2f}-{RB[i+1]:.2f} | {s.sum():6d} | {d['h5'][s].mean():6.2f} | {np.mean(d['h5'][s] / (K * d['S'][s])):6.3f} | "
          f"{f(d['d_own']):6.2f} {f(d['d_next']):6.2f} {f(d['d_prev']):6.2f} {f(d['d_p2']):6.2f} {f(d['d_p1']):6.2f} | {f(d['k_own']):6.2f}  (K5 at runner-up pos: {f(d['k_p2']):6.2f})")

print("\n regression of the unit-5 pulse size on the inputs:  h5 ~ c1 x_t + c2 S + c3 M (L events, unit 5 on): ", end="")
b, R2 = lsq(d["h5"], [d["ratio"] * d["S"], d["S"], d["M"]]); print(np.round(b, 2), f"R2 = {R2:.3f}")

print("\n margin of position t over the runner-up's position at the end (only sequences where that runner-up is still the final runner-up):")
print(" ratio x_t/S | n | P(logit[t] > logit[p2]) actual | with the pulse removed | mean margin/S actual | removed | mean dmargin / S")
for i in range(len(RB) - 1):
    s = (d["ratio"] >= RB[i]) & (d["ratio"] < RB[i + 1]) & d["S_final"]
    print(f"  {RB[i]:.2f}-{RB[i+1]:.2f} | {s.sum():6d} | {np.mean(d['marg'][s] > 0):.4f} | {np.mean(d['marg_f'][s] > 0):.4f} | {np.mean(d['marg'][s] / d['S'][s]):7.2f} | {np.mean(d['marg_f'][s] / d['S'][s]):7.2f} | {np.mean((d['marg'][s] - d['marg_f'][s]) / d['S'][s]):6.2f}")
print("\n final answers in those sequences:  P(answer = t) actual / removed;   P(correct) actual / removed")
for i in range(len(RB) - 1):
    s = (d["ratio"] >= RB[i]) & (d["ratio"] < RB[i + 1]) & d["S_final"]
    print(f"  {RB[i]:.2f}-{RB[i+1]:.2f} | P(ans=t) {np.mean(d['ans_t'][s]):.4f} / {np.mean(d['ansf_t'][s]):.4f} | P(correct) {np.mean(d['ok'][s]):.4f} / {np.mean(d['okf'][s]):.4f}")
print("\n all L events with unit 5 on (any later history): P(correct) actual / removed:", round(d["ok"].mean(), 4), "/", round(d["okf"].mean(), 4),
      "  broke:", round(np.mean(d["ok"] & ~d["okf"]), 4), " fixed:", round(np.mean(~d["ok"] & d["okf"]), 4))
s = ~d["S_final"]
print(" L events after which the runner-up changes later (n = %d): P(correct) actual / removed: %.4f / %.4f   broke %.4f fixed %.4f" % (s.sum(), d["ok"][s].mean(), d["okf"][s].mean(), np.mean(d["ok"][s] & ~d["okf"][s]), np.mean(~d["ok"][s] & d["okf"][s])))
