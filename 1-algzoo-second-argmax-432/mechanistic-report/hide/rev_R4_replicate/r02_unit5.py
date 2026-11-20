"""R4 / r02: section 5.2 (unit 5's input formula, Theta_5, firing fractions) and
section 5.3 (share of new runners-up that cut no comparator, by age of the maximum).
Seeds 7031-7036.  Output: r02_unit5_out.txt
"""
from rlib import *
import modes

out = open(__file__.replace(".py", "_out.txt"), "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(v) for v in a)
    print(s); out.write(s + "\n"); out.flush()


# ------------------------------------------------------------------ from the weights
c5 = W[5, ORD] @ w[ORD]
P("=== SECTION 5.2, from the weights")
P("W[5,ORD].w[ORD] = %.4f ; w_5 = %.4f" % (c5, w[5]))
P("kappa_j = W[5,j]*g_j for comparators 1,6,7,8:", np.round(W[5, COMP] * G, 4), " (term kappa_j (x_t - theta_j)_+ when comparator j is cut)")
P("slope of z5(t+1) in x_t: none cut %.3f ; 8 cut %.3f ; 7 cut %.3f ; 7 and 8 cut %.3f ; all four %.3f" % (
    c5, c5 + W[5, 8] * G[3], c5 + W[5, 7] * G[2], c5 + W[5, 7] * G[2] + W[5, 8] * G[3], c5 + (W[5, COMP] * G).sum()))
P("W[5,PULSE] (units 4,5,9) =", np.round(W[5, PULSE], 4), "; W[5,4]*w4 = %.3f ; W[9,4]*w4 = %.3f" % (W[5, 4] * W4, W[9, 4] * W4))
A = W[np.ix_(ORD, ORD)]
P("Theta_5 per unit mode coordinate of h_(t-1)  [= -W[5,ORD].A.R_n / c5]  and of the pre-input state A h_(t-1) [= -W[5,ORD].R_n / c5]:")
for nme in modes.NAMES:
    R = modes.RIGHT[nme]
    a = -(W[5, ORD] @ (A @ R)) / c5
    b = -(W[5, ORD] @ R) / c5
    if modes.IS_COMPLEX[nme]:
        P("  %-7s  %.5f (2|.|)   %.5f (2|.|)" % (nme, 2 * abs(a), 2 * abs(b)))
    else:
        P("  %-7s  %+.5f          %+.5f" % (nme, a.real, b.real))

# ------------------------------------------------------------------ data
N = 400_000
acc = {}


def add(k, v):
    acc[k] = acc.get(k, 0) + v


err_full = 0.0
errs_rep = []                         # error of the report's formula on '7 and 8 not cut'
reg = {a: [[], []] for a in range(1, 9)}      # per age: X rows, y (state-only Theta5)
reg_ex = {a: [[], []] for a in range(1, 9)}
bins = np.array([-np.inf, 0, .3, .5, .6, .7, .8, .9, 1.0])
fire = np.zeros((len(bins) - 1, 2))
nocut = np.zeros((10, 6))             # by age: n, none cut & u4 off, none cut, u4 on, >=1 cut & u4 off, all four
nocut2 = np.zeros((10, 6))            # population S+ < x_t < M (allows S <= 0, t >= 1)
lagacc = np.zeros((10, 6))            # final runner-up arrived lag steps after final max: n, ok | none&u4off n, ok | cut&u4off n, ok
thr5_rule = [0, 0]
for sd in (7031, 7032, 7033):
    x = gauss(N, sd)
    H, Z = fwd(x)
    st = prefix_stats(x)
    Mp, Sp, pMp = prev(st["M"], -np.inf), prev(st["S"], -np.inf), prev(st["pM"], -1)
    ans = (H[:, -1] @ U.T).argmax(1)
    ok = ans == st["pS"][:, -1]
    for t in range(0, 9):
        hm = H[:, t - 1] if t > 0 else np.zeros((N, 16))
        pre = hm @ W.T                                   # W h_{t-1}
        z5 = Z[:, t + 1, 5]
        clip = np.maximum(-Z[:, t], 0.0)                 # clipped amounts at step t
        lin = w[5] * x[:, t + 1] + c5 * x[:, t] + pre[:, ORD] @ W[5, ORD] + H[:, t][:, PULSE] @ W[5, PULSE]
        full = lin + clip[:, ORD] @ W[5, ORD]
        err_full = max(err_full, np.abs(full - z5).max())
        # the report's formula: 18.009 (x_t - Theta5) - 0.281 x_{t+1} + clipped WAVE units
        rep = lin + clip[:, WAVE] @ W[5, WAVE]
        m78 = (Z[:, t, 7] > 0) & (Z[:, t, 8] > 0)
        errs_rep.append((rep - z5)[m78])
        if t < 2:
            continue
        theta5 = -(pre[:, ORD] @ W[5, ORD] + H[:, t][:, PULSE] @ W[5, PULSE]) / c5          # exact (pulses at t included)
        p5 = np.maximum(pre[:, 5], 0.0); p9 = np.maximum(pre[:, 9], 0.0)                     # units 5, 9 at step t for x_t = 0
        theta5s = -(pre[:, ORD] @ W[5, ORD] + W[5, 5] * p5 + W[5, 9] * p9) / c5              # state-only version, unit 4 dropped
        S0, M0 = Sp[:, t], Mp[:, t]
        age = t - pMp[:, t]
        m = S0 > 0
        for a in range(1, 9):
            ma = m & (age == a)
            reg[a][0].append(np.c_[S0[ma], M0[ma]]); reg[a][1].append(theta5s[ma])
            mb = ma & (Z[:, t, 4] <= 0)
            reg_ex[a][0].append(np.c_[S0[mb], M0[mb]]); reg_ex[a][1].append(theta5[mb])
        # firing of unit 5 after inputs below the runner-up (S>0, unit 4 silent, 2<=t<=8)
        mf = m & (x[:, t] < S0) & (Z[:, t, 4] <= 0)
        r = x[:, t] / np.where(m, S0, 1.0)
        on5 = Z[:, t + 1, 5] > 0
        k = np.digitize(r, bins) - 1
        for i in range(len(bins) - 1):
            sel = mf & (k == i)
            fire[i] += [sel.sum(), (sel & on5).sum()]
        # rule 'unit 5 on iff x_t > Theta5_state' on all below-runner-up inputs
        thr5_rule[0] += ((x[:, t] > theta5s) == on5)[mf].sum(); thr5_rule[1] += mf.sum()
    # ---------------- section 5.3: which comparators does a new runner-up cut
    for t in range(1, 10):
        S0, M0 = Sp[:, t], Mp[:, t]
        age = t - pMp[:, t]
        offc = Z[:, t][:, COMP] <= 0
        u4 = Z[:, t, 4] > 0
        none = ~offc.any(1)
        for pop, tab in (((S0 > 0) & (x[:, t] > S0) & (x[:, t] < M0), nocut),
                         ((M0 > 0) & (x[:, t] > np.maximum(S0, 0)) & (x[:, t] < M0), nocut2)):
            for a in range(1, 10):
                s = pop & (age == a)
                tab[a] += [s.sum(), (s & none & ~u4).sum(), (s & none).sum(), (s & u4).sum(), (s & ~none & ~u4).sum(), (s & offc.all(1)).sum()]
    # accuracy when the final runner-up arrived 'lag' steps after the final max
    pM, pS = st["pM"][:, -1], st["pS"][:, -1]
    lag = pS - pM
    rows = np.arange(N)
    offc_at = Z[rows, pS][:, COMP] <= 0
    u4_at = Z[rows, pS, 4] > 0
    for L in range(1, 10):
        s = lag == L
        s1 = s & ~offc_at.any(1) & ~u4_at
        s2 = s & offc_at.any(1) & ~u4_at
        lagacc[L] += [s.sum(), (s & ok).sum(), s1.sum(), (s1 & ok).sum(), s2.sum(), (s2 & ok).sum()]

P("\n=== SECTION 5.2, data (3 x 400k sequences, seeds 7031-7033)")
P("full identity z5(t+1) = w5 x_(t+1) + c5 (x_t - Theta5) + sum_ORD W[5,i] clip_i : max abs error %.2e" % err_full)
e = np.concatenate(errs_rep)
P("report's formula (wave clips only) on steps where comparators 7 and 8 are both on at t (n = %d):" % len(e))
P("   max |error| %.4f ; mean |error| %.5f ; share with |error| > 1e-9: %.4f ; > 0.01: %.4f ; > 0.05: %.4f ; > 0.1: %.4f" % (
    np.abs(e).max(), np.abs(e).mean(), (np.abs(e) > 1e-9).mean(), (np.abs(e) > 0.01).mean(), (np.abs(e) > 0.05).mean(), (np.abs(e) > 0.1).mean()))
P("Theta5 ~ bS*S + bM*M (no intercept), S>0, steps t=2..8, by age of the max at the step a=1..8")
P("  state-only Theta5 (unit 4 dropped, units 5 and 9 evaluated for x_t=0):")
for a in range(1, 9):
    X = np.concatenate(reg[a][0]); y = np.concatenate(reg[a][1])
    c, r2c, r2u = lsq(X, y)
    P("   a=%d n=%7d  bS %.3f  bM %+.3f  R2 centred %.3f  uncentred %.3f" % (a, len(y), c[0], c[1], r2c, r2u))
P("  exact Theta5 (pulses at t included), steps with unit 4 silent:")
for a in range(1, 9):
    X = np.concatenate(reg_ex[a][0]); y = np.concatenate(reg_ex[a][1])
    c, r2c, r2u = lsq(X, y)
    P("   a=%d n=%7d  bS %.3f  bM %+.3f  R2 centred %.3f" % (a, len(y), c[0], c[1], r2c))
P("P(unit 5 on at t+1 | x_t < S, S>0, unit 4 silent, t=2..8) by x_t/S bin:")
lab = ["<0", "0-.3", ".3-.5", ".5-.6", ".6-.7", ".7-.8", ".8-.9", ".9-1"]
for i in range(len(lab)):
    P("   %-6s n=%8d  share %.3f  P(on) %.4f" % (lab[i], fire[i, 0], fire[i, 0] / fire[:, 0].sum(), fire[i, 1] / fire[i, 0]))
P("   all below-runner-up inputs: P(on) = %.4f (n = %d)" % (fire[:, 1].sum() / fire[:, 0].sum(), fire[:, 0].sum()))
P("   rule 'unit 5 on iff x_t > Theta5(state-only)' correct in %.4f" % (thr5_rule[0] / thr5_rule[1]))

P("\n=== SECTION 5.3: new runner-up (S < x_t < M) by age of the max at the event step")
for name, tab in (("population S>0 (t>=2)", nocut), ("population S+ < x_t < M, M>0 (t>=1)", nocut2)):
    P(name)
    P("   age      n   none cut & u4 silent   none cut (any u4)   unit 4 fires   >=1 cut & u4 silent   all four cut")
    for a in range(1, 10):
        r = tab[a]
        if r[0] > 0:
            P("   %d %8d        %.3f               %.3f              %.3f           %.3f               %.4f" % (a, r[0], r[1] / r[0], r[2] / r[0], r[3] / r[0], r[4] / r[0], r[5] / r[0]))
    r = tab.sum(0)
    P("   all %7d        %.3f               %.3f              %.3f" % (r[0], r[1] / r[0], r[2] / r[0], r[3] / r[0]))
P("accuracy by lag = (position of final runner-up) - (position of final max):")
P("   lag     n     acc | share 'cut nothing, unit 4 silent'  acc | acc when >=1 cut & unit 4 silent")
for L in range(1, 10):
    r = lagacc[L]
    P("   %d %7d  %.3f |  %.3f  %.3f | %.3f" % (L, r[0], r[1] / r[0], r[2] / r[0], r[3] / max(r[2], 1), r[5] / max(r[4], 1)))
out.close()
