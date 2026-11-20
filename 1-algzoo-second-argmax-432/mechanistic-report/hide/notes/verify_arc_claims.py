"""Check ARC's published statements about the 2nd-argmax RNNs against the actual weights.

Run:  python -B verify_arc_claims.py   (about 1 min on CPU; prints to stdout)
Reads the checkpoints in arc_raw/models/ (downloaded from gs://arc-ml-public/alg/...).
Indexing: H[:, s] is the state after inputs x_0..x_s, so ARC's h_t (t inputs seen) is H[:, t-1].
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
MODELS = HERE / "arc_raw" / "models"
np.set_printoptions(precision=3, suppress=True, linewidth=250)
rng = np.random.default_rng(0)


def load(name):
    sd = torch.load(MODELS / name, weights_only=True, map_location="cpu")
    f = lambda k: sd[k].numpy().astype(np.float64)
    return f("rnn.weight_ih_l0")[:, 0], f("rnn.weight_hh_l0"), f("linear.weight")


def states(wih, whh, x):
    """All hidden states, shape (N, n, d)."""
    h = np.zeros((len(x), len(wih)))
    out = np.empty((x.shape[0], x.shape[1], len(wih)))
    for s in range(x.shape[1]):
        h = np.maximum(0, h @ whh.T + x[:, s : s + 1] * wih)
        out[:, s] = h
    return out


def final(wih, whh, x, zero=()):
    """Final hidden state; units in `zero` are clamped to 0 at every step."""
    h = np.zeros((len(x), len(wih)))
    for s in range(x.shape[1]):
        h = np.maximum(0, h @ whh.T + x[:, s : s + 1] * wih)
        h[:, list(zero)] = 0
    return h


def acc(wih, whh, woh, x, **kw):
    return ((final(wih, whh, x, **kw) @ woh.T).argmax(1) == np.argsort(x, 1)[:, -2]).mean()


def r2_fit(y, f):
    """Best no-intercept fit y ~ c f. Returns c, centred R^2 and uncentred R^2."""
    c = (y * f).sum() / (f * f).sum()
    sse = ((y - c * f) ** 2).sum()
    return c, 1 - sse / ((y - y.mean()) ** 2).sum(), 1 - sse / (y**2).sum()


def gauss(N, n=10):
    return rng.standard_normal((N, n))


def unif(N, n=10):
    return rng.uniform(-1, 1, (N, n))


# ---------------------------------------------------------------- M_{16,10}
def part_m16():
    print("=" * 30, "M_16,10 = example_2nd_argmax()", "=" * 30)
    wih, whh, woh = load("one_layer_16_hidden_94_acc_2nd_argmax.pth")
    N = 2**20
    xg, xu = gauss(N), unif(N)
    print(f"A1 accuracy, {N} samples: N(0,1) {acc(wih, whh, woh, xg):.4f}   Unif(-1,1) {acc(wih, whh, woh, xu):.4f}")
    z5 = acc(*load("2nd_argmax_16_10_1073741824_2.pth"), xg)
    print(f"   zoo_2nd_argmax(16, 10) (seed 2) on N(0,1): {z5:.4f}")
    print("   W_ih:", wih)

    # A2 standardisation: |W_ih| = 1 for the input units, unit 2 scaled so that W_hh'[2, 4] = +1
    d = np.ones(16)
    for i in (1, 4, 6, 7, 8):
        d[i] = 1 / abs(wih[i])
    d[2] = d[4] / whh[2, 4]
    wih_s, whh_s = d * wih, d[:, None] * whh / d[None, :]
    sub = [2, 4, 6, 7, 1]
    print("\nA2 standardised weights (h' = D h; D_i = 1/|W_ih,i| for units 1,4,6,7,8; D_2 chosen so W_hh'[2,4] = 1)")
    print("   W_ih' on units (2,4,6,7,1):", wih_s[sub], "   ARC: (0, +1, -1, -1, -1)")
    print("   W_hh'[(2,4),(2,4)]:\n", whh_s[np.ix_([2, 4], [2, 4])], "   ARC: [[+1,+1],[-1,-1]]")
    print("   W_hh'[(6,7),(2,4)]:\n", whh_s[np.ix_([6, 7], [2, 4])], "   ARC: [[+1,0],[+1,+1]]")
    print("   W_hh'[(1),(2,4,6,7)]:", whh_s[1, [2, 4, 6, 7]], "   ARC: (+1,+1,+1,-1)")
    print("   full standardised block, rows and columns in order (2,4,6,7,1,8):\n", whh_s[np.ix_(sub + [8], sub + [8])])

    # scale-invariant size of each term of the pre-activation z_i = W_ih,i x_s + sum_j W_hh,ij h_{s-1,j}
    n = 2**17
    x = gauss(n)
    H = states(wih, whh, x)
    prev = H[:, :-1].reshape(-1, 16)  # h_{s-1}, s = 1..9
    xs = x[:, 1:].reshape(-1)
    print("\n   rms of each term of the pre-activation (N(0,1) inputs, steps s = 1..9), per target unit:")
    print("   target | input term | recurrent terms from units 0..15 | rms of total z")
    for i in sub + [8]:
        terms = prev * whh[i]
        z = terms.sum(1) + wih[i] * xs
        rms = lambda a: np.sqrt((a**2).mean(0))
        print(f"   {i:>2} | {rms(wih[i] * xs):6.2f} |", np.round(rms(terms), 2), f"| {rms(z):6.2f}")

    core = [1, 2, 4, 6, 7, 8]
    rest = [j for j in range(16) if j not in core]
    zc, zr = prev[:, core] @ whh[:, core].T, prev[:, rest] @ whh[:, rest].T  # summed recurrent input from each group
    rmsv = lambda a: np.sqrt((a**2).mean(0))
    print("   summed recurrent input from units", core, "(rms, per target 0..15):", np.round(rmsv(zc), 2))
    print("   summed recurrent input from the other ten units (rms, per target):  ", np.round(rmsv(zr), 2))
    print("   ratio (other ten) / (total pre-activation), per target:              ", np.round(rmsv(zr) / rmsv(zc + zr + xs[:, None] * wih), 3))

    # A3 the blog's feature formulas, tested on the activations
    M = np.maximum.accumulate(np.maximum(x, 0), 1)  # M[:, s] = max(0, x_0..x_s)
    Mp = lambda k: np.concatenate([np.zeros((n, k)), M[:, : 10 - k]], 1)  # M_{s-k}, zero when s < k
    xp1 = np.concatenate([np.zeros((n, 1)), x[:, :-1]], 1)  # x_{s-1}
    srt = np.sort(np.where(np.tri(10, dtype=bool)[None], x[:, None, :], -np.inf), 2)  # prefix sorts
    S = np.maximum(srt[:, :, -2], 0)  # max(0, 2nd largest of x_0..x_s); 0 at s = 0
    Sp1 = np.concatenate([np.zeros((n, 1)), S[:, :-1]], 1)
    feats = {
        "n2 ~ max(0,x_0..x_{s-1})  [blog]": (2, Mp(1)),
        "n4 ~ M_s - M_{s-1}  [blog]": (4, M - Mp(1)),
        "n7 ~ M_s - x_s = ReLU(M_{s-1} - x_s)  [blog]": (7, M - x),
        "n6 ~ max(0,x_0..x_{s-2},x_s) - x_s  [blog]": (6, np.maximum(Mp(2), x) - x),
        "n1 ~ max(0,x_0..x_{s-3},x_{s-1},x_s) - x_s  [blog]": (1, np.maximum(np.maximum(Mp(3), np.where(np.arange(10) >= 1, xp1, 0)), x) - x),
        "n8 ~ ReLU(2ndmax_{s-1} - x_s)  [Chen]": (8, np.maximum(Sp1 - x, 0)),
        "n2 ~ max(0,x_0..x_s)  [Chen: running max]": (2, M),
    }
    print("\nA3 no-intercept fit h_s[unit] ~ c * feature (N(0,1) inputs). c' = c * D_unit is the coefficient in standardised units.")
    print("   centred R^2 pooled over s = 0..9, then per s (ARC's t = s + 1):")
    for name, (u, f) in feats.items():
        c, r2, _ = r2_fit(H[:, :, u].ravel(), f.ravel())
        per = []
        for s in range(10):
            y, g = H[:, s, u], f[:, s]
            per.append(r2_fit(y, g)[1] if g.std() > 0 and y.std() > 0 else np.nan)
        print(f"   {name:<52} c = {c:7.3f}  c' = {c * d[u]:6.3f}  R2 = {r2:.3f} | per s:", np.round(per, 2))

    # A3b the same claims as thresholds: unit i is active at step s iff sign(W_ih,i) (x_s - theta_i) > 0,
    # with theta_i(h_{s-1}) = -(W_hh h_{s-1})_i / W_ih,i, a linear readout of the previous state
    theta = -(H[:, :-1] @ whh.T) / wih  # (n, 9, 16): thresholds for x_s, s = 1..9
    Sraw = srt[:, :, -2]  # 2nd largest of x_0..x_s (-inf at s = 0)
    claims = {
        "theta_4 ~ M_{s-1} (unit 4 fires iff x_s is a new max)": (4, Mp(1)[:, 1:]),
        "theta_7 ~ M_{s-1}": (7, Mp(1)[:, 1:]),
        "theta_6 ~ M_{s-2}": (6, Mp(2)[:, 1:]),
        "theta_1 ~ max(M_{s-3}, x_{s-1}, 0)": (1, np.maximum(np.maximum(Mp(3), xp1), 0)[:, 1:]),
        "theta_8 ~ max(0, 2nd max of x_0..x_{s-1})  [Chen]": (8, Sp1[:, 1:]),
    }
    print("\nA3b thresholds theta_i = -(W_hh h_{s-1})_i / W_ih,i against the claimed feature f (input units, N(0,1) inputs):")
    print("   pooled s = 1..9: slope of the no-intercept fit, centred R^2, mean |theta - f|; then mean (theta - f) per s = 1..9")
    for name, (u, f) in claims.items():
        th = theta[:, :, u]
        c, r2, _ = r2_fit(th.ravel(), f.ravel())
        print(f"   {name:<54} slope {c:.3f}  R2 {r2:.3f}  mean|err| {np.abs(th - f).mean():.3f} | mean err per s:", np.round((th - f).mean(0), 3))
    two = Sraw[:, :-1] > 0  # states whose 2nd max so far is positive (s - 1 >= 1)
    th8, f8 = theta[:, :, 8][two], Sp1[:, 1:][two]
    print(f"   theta_8 vs 2nd max, only states with 2nd max > 0: R2 {r2_fit(th8, f8)[1]:.3f}, mean|err| {np.abs(th8 - f8).mean():.3f}")
    ex = np.array([[0.77, 0.90, 0.79] + [0.0] * 7])
    hx = states(wih, whh, ex)[0, 2]
    print("   notebook example, prefix (0.77, 0.90, 0.79): thresholds for x_3 of units 4, 7, 6, 1, 8 =", np.round(-(whh @ hx)[[4, 7, 6, 1, 8]] / wih[[4, 7, 6, 1, 8]], 3),
          " (claims: 0.90, 0.90, 0.90, 0.79, 0.79)")

    # A4 Chen's notebook numbers (Unif(-1,1) inputs)
    xu = unif(2**17)
    print("\nA4 notebook checks on Unif(-1,1):")
    abl = []
    for i in range(16):
        w = wih.copy()
        w[i] = 0
        abl.append(acc(w, whh, woh, xu))
    print("   accuracy with W_ih[i] = 0, i = 0..15:", np.round(abl, 3))
    print("   notebook:                            [0.938 0.154 0.937 0.938 0.182 0.928 0.151 0.12  0.202 0.932 0.938 0.935 0.93  0.938 0.859 0.938]")
    hf = final(wih, whh, xu)
    for excl in ([1, 6, 7, 8], [1, 4, 6, 7, 8, 9], [1, 4, 6, 7, 8, 9, 14]):
        keep = np.ones(16)
        keep[excl] = 0
        a = (((hf * keep) @ woh.T)[:, :9].argmax(1) == xu[:, :9].argmax(1)).mean()
        print(f"   'max sub-circuit': readout without units {excl}: argmax of first 9 logits = argmax of x_0..x_8 with accuracy {a:.4f}")
    print("   notebook: 0.8684 for [1, 6, 7, 8]")
    base = acc(wih, whh, woh, xu)
    w2 = whh.copy()
    w2[:, 9] = 0
    wo2 = woh.copy()
    wo2[:, 9] = 0
    print(f"   unit 9: base {base:.4f} | clamp h[9] = 0 everywhere {acc(wih, whh, woh, xu, zero=[9]):.4f} (readout column kept)"
          f" | W_hh[:, 9] = 0 only {acc(wih, w2, woh, xu):.4f} | W_out[:, 9] = 0 only {acc(wih, whh, wo2, xu):.4f}"
          f" | both {acc(wih, w2, wo2, xu):.4f}")
    cs = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
    sh = woh @ whh
    print("   cos(W_out[i] W_hh, W_out[i+1]), i = 0..8:", np.round([cs(sh[i], woh[i + 1]) for i in range(9)], 2))
    print("   cos(W_out[i] W_hh, W_out[i]),   i = 0..9:", np.round([cs(sh[i], woh[i]) for i in range(10)], 2))
    print("   column sums of W_out:", np.round(woh.sum(0), 2))
    print("   singular values of W_out:", np.round(np.linalg.svd(woh, compute_uv=False), 2))
    xp = rng.uniform(0, 1, (2**15, 10))
    print("   fraction of nonzero activations over all steps, Unif(0,1) inputs (as in the notebook):", np.round((states(wih, whh, xp) > 0).mean((0, 1)), 2))

    # A5 Chen's general remarks: intermediate logits, sign of the inputs, per-class accuracy
    print("\nA5 Chen's remarks on the setup:")
    lg = H @ woh.T  # logits read at every step, N(0,1) inputs
    inter = []
    for s in range(1, 10):
        tgt = np.argsort(x[:, : s + 1], 1)[:, -2] + 9 - s  # 2nd argmax of the prefix, in output coordinates
        two = (x[:, : s + 1] > 0).sum(1) >= 2
        inter.append(((lg[:, s].argmax(1) == tgt).mean(), (lg[:, s].argmax(1) == tgt)[two].mean()))
    print("   'intermediate logits': P(argmax of W_out h_s = 9 - s + 2nd argmax of x_0..x_s), s = 1..9:", np.round([a for a, _ in inter], 3))
    print("   same, given at least two positive inputs so far:                                       ", np.round([b for _, b in inter], 3))
    ok = lg[:, 9].argmax(1) == np.argsort(x, 1)[:, -2]
    npos = (x > 0).sum(1)
    print(f"   N(0,1) accuracy by number of positive inputs: 0: {ok[npos == 0].mean():.3f} (n = {(npos == 0).sum()}), 1: {ok[npos == 1].mean():.3f} (n = {(npos == 1).sum()}), >=2: {ok[npos >= 2].mean():.4f}")
    print(f"   accuracy on Unif(0,1): {acc(wih, whh, woh, rng.uniform(0, 1, (2**18, 10))):.4f}   on Unif(-1,0): {acc(wih, whh, woh, rng.uniform(-1, 0, (2**18, 10))):.4f}")
    tg = np.argsort(xu, 1)[:, -2]
    pu = (final(wih, whh, xu) @ woh.T).argmax(1)
    print("   accuracy by target position 0..9, Unif(-1,1):", np.round([(pu[tg == i] == i).mean() for i in range(10)], 3))
    print("   notebook:                                     [0.942 0.955 0.955 0.944 0.944 0.936 0.931 0.923 0.917 0.923]")
    tgg = np.argsort(x, 1)[:, -2]
    print("   accuracy by target position 0..9, N(0,1):    ", np.round([(lg[tgg == i, 9].argmax(1) == i).mean() for i in range(10)], 3))


# ---------------------------------------------------------------- handcrafted models
def handcrafted_10():
    """alg_zoo/handcrafted.py: handcrafted_2nd_argmax_10 (22 units), transcribed to numpy."""
    wih, whh, woh = np.zeros(22), np.zeros((22, 22)), np.zeros((10, 22))
    wih[9] = 1
    for i in range(9):
        whh[i, i + 1] = 1
    wih[19] = 1
    whh[19, 19] = whh[19, 20] = -1
    whh[20, 19] = whh[20, 20] = 1
    wih[21] = -1
    whh[21, 19] = whh[21, 20] = 1
    whh[18, 20] = 1
    wih[18] = -1
    for i in range(10, 18):
        whh[i, i + 1] = 1
        whh[i, 21] = -1
        whh[i, 20] = whh[i, 19] = 1
        wih[i] = -1
    for i in range(10):
        woh[i, i] = 1
    for i in range(9):
        woh[i, 10 + i] = 2
        woh[i, 21] = -2
        woh[i, 20] = woh[i, 19] = 2
    woh[9, 20] = 2
    return wih, whh, woh


def chen_21():
    """The 21-unit hand-coded RNN of Chen's notebook (cell 47)."""
    woh = np.zeros((10, 21))
    woh[9, 0:3] = (1, -1, -1)
    for k in range(9):
        woh[k, 2] = -2
        woh[k, (9 - k) * 2 + 1] = 1
        woh[k, (9 - k) * 2 + 2] = 2
    wih = np.zeros(21)
    wih[1] = 1
    for k in range(1, 11):
        wih[2 * k] = -1
    whh = np.zeros((21, 21))
    whh[0, 0] = whh[0, 1] = 1
    whh[1] = -whh[0]
    whh[2] = whh[0]
    whh[3, 0:3] = (1, 1, -1)
    whh[4] = whh[0] - 0.5 * whh[3] + 0.5 * woh[9]
    for k in range(5, 20, 2):
        whh[k, k - 2] = 1
        whh[k + 1] = whh[0] - 0.5 * whh[k] + 0.5 * woh[int((21 - k) / 2)]
    return wih, whh, woh


def part_handcrafted():
    print("\n" + "=" * 30, "handcrafted 10-input models", "=" * 30)
    N = 2**20
    xg, xu = gauss(N), unif(N)
    wih, whh, woh = handcrafted_10()
    print(f"B1 handcrafted_2nd_argmax(10), 22 units: accuracy N(0,1) {acc(wih, whh, woh, xg):.4f}, Unif(-1,1) {acc(wih, whh, woh, xu):.4f}")
    x = xg[:4096]
    logits = final(wih, whh, x) @ woh.T
    xp = np.maximum(x, 0)
    loo = np.stack([np.delete(xp, i, 1).max(1) for i in range(10)], 1)  # max(0, x_j : j != i)
    print("   max |logit_i - (max(0,x_i) + 2 max(0, x_j: j != i))| =", np.abs(logits - (xp + 2 * loo)).max(),
          "  (the code comment says '- 2 *'; the weights give '+ 2 *')")
    npos = (xg > 0).sum(1)
    ok = (final(wih, whh, xg) @ woh.T).argmax(1) == np.argsort(xg, 1)[:, -2]
    print(f"   accuracy by number of positive inputs: >=2: {ok[npos >= 2].mean():.5f}, 1: {ok[npos == 1].mean():.4f}, 0: {ok[npos == 0].mean():.4f};"
          f" P(<2 positive) = {11 / 1024:.4f}")
    wih, whh, woh = chen_21()
    print(f"B2 Chen's 21-unit hand-coded RNN: accuracy N(0,1) {acc(wih, whh, woh, xg):.4f}, Unif(-1,1) {acc(wih, whh, woh, xu):.4f} (notebook: 0.9886 on 10^4 uniform samples)")


# ---------------------------------------------------------------- M_{2,2}
def part_m22():
    print("\n" + "=" * 30, "M_2,2 = zoo_2nd_argmax(2, 2), seed 2", "=" * 30)
    wih, whh, woh = load("2nd_argmax_2_2_1073741824_2.pth")
    print("C1 raw W_ih", wih, "\n   raw W_hh\n", whh, "\n   raw W_oh\n", woh)
    d = 1 / np.abs(wih)
    wih, whh, woh = d * wih, d[:, None] * whh / d[None, :], woh / d[None, :]
    if wih[0] < 0:  # order the units as (+1, -1)
        p = [1, 0]
        wih, whh, woh = wih[p], whh[np.ix_(p, p)], woh[:, p]
    print("   standardised W_ih", wih, "\n   standardised W_hh\n", whh, "\n   standardised W_oh\n", woh)
    print("   ARC: W_hi = (+1,-1); W_hh in [[[-1,0), [1,inf)], [[1,inf), [-1,0)]]; W_oh signs [[+,-],[-,+]]")
    # linear pieces of Delta = logit_1 - logit_0 = a0 x0 - a1 x1
    dw = woh[1] - woh[0]
    print("   pieces of Delta(x0, x1) = a0 x0 - a1 x1 (sign of x0, active units at step 2):")
    for sgn, name in ((1, "x0>0"), (-1, "x0<0")):
        h1 = np.array([1.0, 0.0]) if sgn > 0 else np.array([0.0, -1.0])  # h_1 = h1 * x0
        zc0, zc1 = whh @ h1, wih  # z_2 = zc0 x0 + zc1 x1
        for mask in ((1, 1), (1, 0), (0, 1)):
            m = np.array(mask)
            a0, a1 = dw @ (m * zc0), -(dw @ (m * zc1))
            print(f"     {name}, active {mask}: a0 = {a0:10.5f}, a1 = {a1:10.5f}, a0/a1 = {a0 / a1:.6f}, |a0/a1 - 1| = 1 part in {1 / abs(a0 / a1 - 1):.0f}")
        print(f"     {name}: unit 0 active iff x1 > {-zc0[0]:.4f} x0; unit 1 active iff x1 < {zc0[1]:.4f} x0")
    x = rng.standard_normal((2**24, 2))
    e = 1 - acc(wih, whh, woh, x)
    print(f"   sampled error rate on N(0,1), 2^24 samples: {e:.3e} = 1 in {1 / e:.0f}   (ARC: about 1 in 13,000; their val log: 1 in 12,363)")


# ---------------------------------------------------------------- M_{4,3}
def part_m43():
    print("\n" + "=" * 30, "M_4,3 = zoo_2nd_argmax(4, 3), seed 0", "=" * 30)
    wih, whh, woh = load("2nd_argmax_4_3_1073741824_0.pth")
    print("D1 raw W_ih", wih, "\n   raw W_hh\n", whh, "\n   raw W_oh\n", woh)
    x = rng.standard_normal((2**22, 3))
    print(f"   accuracy on N(0,1), 2^22 samples: {acc(wih, whh, woh, x):.4f}   (ARC: 98.5%)")
    d = 1 / np.abs(wih)
    wih, whh, woh = d * wih, d[:, None] * whh / d[None, :], woh / d[None, :]
    U = np.array([[1, 0, 0, -1], [0, 1, -1, 0]], float)
    V = np.abs(U)
    pos, neg = np.where(wih > 0)[0], np.where(wih < 0)[0]
    best = None
    for p in ([pos[0], pos[1], neg[0], neg[1]], [pos[0], pos[1], neg[1], neg[0]]):  # unit order (a, b, -b, -a)
        wi, wh, wo = wih[p], whh[np.ix_(p, p)], woh[:, p]
        ri, rh, ro = U @ wi / 2, U @ wh @ U.T / 4, wo @ V.T / 2  # least-squares reduced weights
        res = [np.linalg.norm(wi - U.T @ ri), np.linalg.norm(wh - U.T @ rh @ U), np.linalg.norm(wo - ro @ V)]
        rel = [res[0] / np.linalg.norm(wi), res[1] / np.linalg.norm(wh), res[2] / np.linalg.norm(wo)]
        if best is None or sum(rel) < sum(best[0]):
            best = (rel, p, wi, wh, wo, ri, rh, ro)
    rel, p, wi, wh, wo, ri, rh, ro = best
    print("   standardised (|W_ih| = 1), units reordered as", list(p), "\n   W_ih", wi, "\n   W_hh\n", wh, "\n   W_oh\n", wo)
    print("   Robinson's symmetry W_ih ~ U^T w_ih', W_hh ~ U^T w_hh' U, W_oh ~ w_ho' V: relative residuals (ih, hh, oh) =", np.round(rel, 4))
    print("   reduced w_ih'", ri, "\n   reduced w_hh'\n", rh, "\n   reduced w_ho'\n", ro)
    sym = (U.T @ ri, U.T @ rh @ U, ro @ V)
    print(f"   accuracy of the exactly symmetric model: {acc(*sym, x):.4f}")
    fc1 = np.stack([rh @ rh @ ri, rh @ ri, ri], 1)  # logits = fc2 |fc1 x|
    fc2 = ro - ro.mean(0)
    print("   equivalent abs-network logits = fc2 |fc1 x|: fc1 (columns x0, x1, x2)\n", fc1, "\n   row sums of fc1:", fc1.sum(1))
    print("   fc2 (column means removed)\n", fc2)
    u = x @ fc1.T
    okr = (np.abs(u) @ ro.T).argmax(1) == np.argsort(x, 1)[:, -2]
    print(f"   accuracy of fc2 |fc1 x|: {okr.mean():.4f}")
    # Robinson's normal form for the argmedian abs-network: fc1 rows (1, -(lam+1), lam) and (1, lam, -(lam+1)),
    # fc2 columns (eps, -mu, mu-eps) and (eps, mu-eps, -mu). Rescale each abs-unit so its x0 coefficient is 1.
    k = np.abs(fc1[:, 0])
    n1, n2 = fc1 / fc1[:, :1], fc2 * k
    print("   normal form (x0 coefficient 1): fc1\n", n1, "\n   fc2\n", n2)
    lam, eps, mu = (n1[0, 2] - n1[0, 1] - 1 + n1[1, 1] - n1[1, 2] - 1) / 4, n2[0].mean(), -(n2[1, 0] + n2[2, 1]) / 2
    print(f"   averaged parameters: lambda = {lam:.3f}, eps = {eps:.3f}, mu = {mu:.3f}   (Robinson's abs-network: 7.808, 0.7, 17.79)")
    print(f"   Robinson's ratios: 6 eps/((2 lam+1)(2 mu-eps)) = {6 * eps / ((2 * lam + 1) * (2 * mu - eps)):.4f} (his: 0.007; width of the failure sliver),"
          f" 2(2 mu-eps)/(3 eps (2 lam+1)) = {2 * (2 * mu - eps) / (3 * eps * (2 * lam + 1)):.3f} (his: 'almost exactly 2')")
    v = x @ n1.T  # pre-activations in Robinson's normal form
    quad = (v[:, 0] > 0) * 2 + (v[:, 1] > 0)
    pred = (np.abs(u) @ ro.T).argmax(1)
    for q, nm in ((3, "(+,+)"), (2, "(+,-)"), (1, "(-,+)"), (0, "(-,-)")):
        m = quad == q
        print(f"   Robinson quadrant {nm}: probability {m.mean():.4f}, accuracy {okr[m].mean():.4f}, share of all errors {(~okr & m).sum() / (~okr).sum():.3f},"
              f" wrong answers by output position {np.bincount(pred[m & ~okr], minlength=3)}")
    # Hilton's a_j, b_j, c_j for the standardised, reordered model (units 0,1 positive input weight; 2,3 negative)
    aj, bj, cj = wh[:, 0] * wi[0] + wh[:, 1] * wi[1], wh[:, 2] * wi[2] + wh[:, 3] * wi[3], wi
    print("   Hilton's a_j", aj, " b_j", bj, " c_j", cj)
    print("   claims: a3 ~ -a0 > 0, a2 ~ -a1 > 0, b3 ~ -b0 > 0, b2 ~ -b1 > 0, c0 ~ -c3 > 0, c1 ~ -c2 > 0, a_j ~ b_j")
    al3, al2, be3, be2 = (aj[3] - aj[0]) / 2, (aj[2] - aj[1]) / 2, (bj[3] - bj[0]) / 2, (bj[2] - bj[1]) / 2
    g0, g1 = (cj[0] - cj[3]) / 2, (cj[1] - cj[2]) / 2
    print(f"   gamma0/alpha3 = {g0 / al3:.3f}, gamma0/beta3 = {g0 / be3:.3f} (claimed < 1); gamma1/alpha2 = {g1 / al2:.3f}, gamma1/beta2 = {g1 / be2:.3f} (claimed > 1)")
    lam = 100.0
    f1 = np.array([[1, -lam - 1, lam], [-1, -lam, lam + 1]])
    f2 = np.array([[1, 1], [-lam, lam + 1], [lam + 1, -lam]])
    hw = (U.T @ f1[:, 2], U.T @ f1[:, :2] @ np.linalg.inv(f1[:, 1:]) @ U, f2 @ V)
    print(f"   handcrafted_2nd_argmax(3) (lambda = 100) in float64: accuracy {acc(*hw, x):.5f}   (ARC: 99.99%)")


if __name__ == "__main__":
    part_m16()
    part_handcrafted()
    part_m22()
    part_m43()
