"""Spot-checks of numbers quoted in the program-route working paper
(agent-program-route_second-argmax-paper-v3.pdf) against the weights of M_16,10.

Companion to digest_program_paper.md (section 8).  Read-only on the weights; prints only.
Run:  python -B check_program_paper.py > check_program_paper_output.txt      (about 1-2 min, CPU)
"""
import sys
sys.dont_write_bytecode = True
import math
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                                   # workspace root
W = np.load(ROOT / "reverse-engineered-networks" / "1-algzoo-second-argmax-432" / "code" / "m16_10_weights.npz")
w, Whh, Wout = (W[k].astype(np.float64) for k in ("W_ih", "W_hh", "W_out"))
w = w[:, 0]
np.set_printoptions(precision=3, suppress=True, linewidth=220)
CMP, THIRTEEN = [1, 6, 7, 8], [i for i in range(16) if i not in (4, 5, 9)]
IDX = np.arange(10)


def forward(x, dtype=np.float64):
    """Pre-activations (N, 10, 16) and logits (N, 10)."""
    wi, wh = w.astype(dtype), Whh.astype(dtype)
    h, pre = np.zeros((len(x), 16), dtype), []
    for t in range(10):
        pre.append(x[:, t:t + 1].astype(dtype) * wi + h @ wh.T)
        h = np.maximum(pre[-1], 0)
    return np.stack(pre, 1), h @ Wout.astype(dtype).T


def before(x):
    """Maximum, runner-up and position of the maximum of the inputs before each step (-inf if absent)."""
    N = len(x)
    mb, sb, pm = np.full((N, 10), -np.inf), np.full((N, 10), -np.inf), np.zeros((N, 10), int)
    m, s, p = np.full(N, -np.inf), np.full(N, -np.inf), np.zeros(N, int)
    for t in range(10):
        mb[:, t], sb[:, t], pm[:, t] = m, s, p
        new = x[:, t] > m
        s, p, m = np.where(new, m, np.maximum(s, x[:, t])), np.where(new, t, p), np.maximum(m, x[:, t])
    return mb, sb, pm


def section(title):
    print("\n" + "=" * 100 + "\n" + title + "\n" + "=" * 100)


# ---------------------------------------------------------------------------------------------------
section("A. Numbers that can be read off the weights")
print("input weights w:", w)
print("  paper p.2: n4 +10.16, n1 -10.56, n6 -11.00, n8 -12.31, n7 -13.17")
print(f"W_hh[2,2] = {Whh[2, 2]:.4f}, W_hh[2,4] = {Whh[2, 4]:.4f}     (paper p.2: 0.97 and 1.73)")
print(f"W_out[9,4] = {Wout[9, 4]:.4f}, w4*W_out[9,4] = {w[4] * Wout[9, 4]:.2f}     (paper p.8: -4.99 and -50.7)")
rest = np.ones(16); rest[[4, 5, 9]] = 0
post = np.ones(16); post[4] = 0
A = rest[:, None] * Whh
ev = np.linalg.eigvals(A); ev = ev[np.argsort(-np.abs(ev))]
print("eigenvalues of the resting map A = G_rest W_hh (modulus, angle):   (paper Table 2)")
for e in ev[:13]:
    print(f"    {abs(e):.4f}  {np.degrees(np.angle(e)):+8.2f} deg")
send = Whh @ (rest * w)
print("W_hh G w, resting gates (per unit of an arriving input, next step):", send)
print(f"  to n5 {send[5]:.1f}, to n9 {send[9]:.1f}, largest other |.| {np.abs(np.delete(send, [5, 9])).max():.2f}   (paper p.6: 18.0, 84.3, at most 0.4)")
g8 = rest.copy(); g8[8] = 0
print("same with n8 off:", Whh @ (g8 * w), "  (paper Fig. 4a purple; n9 71)")
print(f"W_hh[5,j]*w_j for j = 1, 6, 7, 8: {Whh[5, CMP] * w[CMP]}   W_hh[5,4]*w4 = {Whh[5, 4] * w[4]:.2f}   W_hh[9,4]*w4 = {Whh[9, 4] * w[4]:.2f}")
grec = rest.copy(); grec[5] = 1
E = np.array([Wout @ (np.linalg.matrix_power(A, 8 - p) @ (grec * send)) for p in range(2, 9)])
print("Fig. 5(a) reconstruction  W_out A^(8-p) G_rec W_hh (G_rest w), rows p = 2..8, columns = logit position:")
print(np.round(E, 0))
print("  own position:", np.round(E[np.arange(7), np.arange(2, 9)], 1), "  (paper: 72 76 71 70 67 62 62)")
print("  one position later:", np.round(E[np.arange(6), np.arange(3, 9)], 1), "  (paper: about +20)")
pair = np.array([[1.2, 0.1, 0, 1.23, 1.26, 0, 0, 0, 0, 0], [1.2, 1.1, 0, 1.23, 1.26, 0, 0, 0, 0, 0]])
lp = forward(pair)[1]
print("constructed pair (p.11-12): answers", lp.argmax(1), " margin logit3 - logit0:", lp[:, 3] - lp[:, 0],
      f" difference {np.diff(lp[:, 0] - lp[:, 3])[0]:.2f}   (paper: answers 3 and 0, difference 5.5)")

# ---------------------------------------------------------------------------------------------------
section("B. Accuracy, kinds of mistake, margins, near ties (N = 2^20, fp32)")
rng = np.random.default_rng(0)
x = rng.standard_normal((2 ** 20, 10))
logits = forward(x, np.float32)[1].astype(np.float64)
N = len(x); r = np.arange(N)
order = np.argsort(-x, 1); xs = np.take_along_axis(x, order, 1)
ans = logits.argmax(1); ok = ans == order[:, 1]
third, top = ans == order[:, 2], ans == order[:, 0]
print(f"accuracy {ok.mean():.2%}   (paper 95.3%)")
print(f"answers third largest {third.mean():.2%}, maximum {top.mean():.2%}, smaller input {(~ok & ~third & ~top).mean():.2%}   (paper 2.97%, 0.95%, 0.76%)")
near = ~ok & (np.abs(x[r, ans] - xs[:, 1]) < 0.05); nerr = (~ok).sum()
print(f"errors with the answer within 0.05 of the runner-up: {near.sum() / nerr:.1%} of errors; third {(near & third).sum() / nerr:.1%}, "
      f"maximum {(near & top).sum() / nerr:.1%}, smaller {(near & ~third & ~top).sum() / nerr:.1%}   (paper 65%, 48%, 15%, 2%)")
srt = np.sort(logits, 1); margin = srt[:, -1] - srt[:, -2]
print(f"median logit margin: mistakes {np.median(margin[~ok]):.2f}, correct {np.median(margin[ok]):.2f}   (paper 1.0 and 8.5)")
r23, r12 = xs[:, 1] - xs[:, 2] < 0.05, xs[:, 0] - xs[:, 1] < 0.05
print(f"runner-up/third within 0.05: {r23.mean():.1%} of inputs, network right on {ok[r23].mean():.1%}   (paper 78%)")
print(f"no near tie among the top three: {(~r23 & ~r12).mean():.1%} of inputs, network right on {ok[~r23 & ~r12].mean():.2%}   (paper 98.4%)")
print(f"median logit[runner-up] - logit[maximum]: {np.median(logits[r, order[:, 1]] - logits[r, order[:, 0]]):.1f}   (paper +12)")
print(f"runner-up at the last position: {np.mean(order[:, 1] == 9):.1%}   (paper: about a tenth)")

# ---------------------------------------------------------------------------------------------------
section("C. Gates and thresholds (N = 2^18, float64)")
x = x[:2 ** 18]; N = len(x); r = np.arange(N)
order = np.argsort(-x, 1)
pre, logits = forward(x)
G = pre > 0
mb, sb, pm = before(x)
age = IDX[None, :] - pm
tt = np.broadcast_to(IDX[None, :], x.shape)
newmax, newrun, below = x > mb, (x < mb) & (x > sb), x < sb
eq = G[:, 1:, 9] == G[:, :-1, 4]
print(f"n9's gate equals n4's gate one step earlier, steps 1-9: {eq.mean():.2%}   (paper p.2: 99.0%)")
alloff = ~G[:, :, CMP].any(2)
print(f"a new maximum switches n1, n6, n7, n8 all off: arrivals at steps 1-9 {alloff[newmax & (tt >= 1)].mean():.1%}, steps 2-9 {alloff[newmax & (tt >= 2)].mean():.1%}   (paper p.8: 86%)")
passed = {4: G[:, :, 4], 1: ~G[:, :, 1], 6: ~G[:, :, 6], 7: ~G[:, :, 7], 8: ~G[:, :, 8]}
show = lambda sel: {j: f"{p[sel].mean():.2%}" for j, p in passed.items()}
print("a new maximum passes each threshold: steps 1-9", show(newmax & (tt >= 1)), " steps 2-9", show(newmax & (tt >= 2)), "  (paper p.6: 93-95%)")
print("an input below the runner-up so far (steps 2-9) passes:", show(below & (tt >= 2)), "  (paper p.6: at most 0.5% each)")
nr = newrun & (tt >= 2)
none = ~(passed[4] | passed[1] | passed[6] | passed[7] | passed[8])
print(f"new runners-up (steps 2-9): pass n8's threshold and not n7's {(passed[8] & ~passed[7])[nr].mean():.1%} (paper: about half); "
      f"pass none {none[nr].mean():.1%}; pass n4's (misfire) {passed[4][nr].mean():.1%}")
all13 = G[:, :, THIRTEEN].all(2); exact = all13 & ~G[:, :, [4, 5, 9]].any(2)
sel = below & (tt >= 2)
print(f"steps with the input below the runner-up so far: all thirteen on {all13[sel].mean():.1%}, exact resting pattern {exact[sel].mean():.1%}   (paper p.5: 34%, 17%)")
n5next = np.zeros_like(G[:, :, 5]); n5next[:, :9] = G[:, 1:, 5]
for name, ev_ in (("new maximum (steps 1-8)", newmax & (tt >= 1) & (tt <= 8)), ("new runner-up (steps 2-8)", nr & (tt <= 8))):
    print(f"n5 on one step after a {name}: all {n5next[ev_].mean():.1%}; input > 0: {n5next[ev_ & (x > 0)].mean():.1%}; input < 0: {n5next[ev_ & (x < 0)].mean():.1%}")
p2 = order[:, 1]; m8 = p2 <= 8
print(f"final runner-up arriving before the last step is recorded (n5 on at the next step): {G[r[m8], p2[m8] + 1, 5].mean():.1%}   (paper p.6: 99%)")
theta = x[:, :, None] - pre / w                              # theta_j = -W_hh[j].h_{t-1}/w_j
sel = (tt >= 2) & (mb - sb > 0.2)
print("median threshold position (theta_j - 2nd)/(max - 2nd) by age of the maximum 1..9 (arrivals at steps 2-9, gap > 0.2; Fig. 3 circles):")
with np.errstate(invalid="ignore"):
    pos = (theta - sb[:, :, None]) / (mb - sb)[:, :, None]
for j in (4, 7, 6, 1, 8):
    print(f"    n{j}:", np.array([np.median(pos[:, :, j][sel & (age == a)]) for a in range(1, 10)]))

# ---------------------------------------------------------------------------------------------------
section("D. Own effects as Jacobian entries d logit_q / d x_p along the actual gates (N = 2^18)")
Gf = G.astype(np.float64)
J = np.zeros((N, 10, 10))                                      # J[n, q, p]
Jgrp = {}                                                      # own effect of the final maximum split by neuron group
p1 = order[:, 0]
grp = {"n1 n6 n7 n8": CMP, "n4": [4], "n0 n9 n5": [0, 9, 5], "memory": [2, 3, 10, 11, 12, 13, 14, 15]}
own_split = {k: np.zeros(N) for k in grp}
for p in range(10):
    v = Gf[:, p] * w
    for t in range(p + 1, 10):
        v = Gf[:, t] * (v @ Whh.T)
    J[:, :, p] = v @ Wout.T
    m_ = p1 == p
    for k, i in grp.items():
        own_split[k][m_] = (v[m_][:, i] * Wout[p, i]).sum(1)
print(f"homogeneity (Euler) check, max |logits - J x| = {np.abs(logits - np.einsum('nqp,np->nq', J, x)).max():.1e}")
own = J[r, p1, p1]
print("the final maximum's own effect by its age 0..9: median [quartiles] mean, and with the commonest joint gate pattern at each step   (Fig. 6a)")
for a in range(10):
    p = 9 - a; m_ = p1 == p; j = own[m_]
    v = None
    for t in range(p, 10):
        code = Counter((Gf[m_, t] @ (2 ** np.arange(16))).astype(int).tolist()).most_common(1)[0][0]
        g = ((code >> np.arange(16)) & 1).astype(float)
        v = g * w if v is None else g * (Whh @ v)
    print(f"    age {a}: {np.median(j):6.1f} [{np.percentile(j, 25):6.1f}, {np.percentile(j, 75):6.1f}]  mean {j.mean():6.1f}   commonest gates {Wout[p] @ v:6.1f}")
m_ = p1 <= 7
print("mean own effect of a final maximum aged 2-9 by neuron group:", {k: round(float(v[m_].mean()), 1) for k, v in own_split.items()},
      f" total {own[m_].mean():.1f}   (Fig. 6c green: -72, 0, 0, +54, total -18)")
a_, b_, c_, d_ = J[r, p2, p2], J[r, p1, p2], J[r, p1, p1], J[r, p2, p1]
mar = logits[r, p2] - logits[r, p1]
t1, t2 = (d_ - c_) * x[r, p1], (a_ - b_) * x[r, p2]
print(f"medians: dL_2nd/dx_2nd {np.median(a_):.1f}, dL_max/dx_2nd {np.median(b_):.1f}, dL_max/dx_max {np.median(c_):.1f}, dL_2nd/dx_max {np.median(d_):.1f}; "
      f"P(dL_max/dx_2nd >= dL_2nd/dx_2nd) = {np.mean(b_ >= a_):.2f}")
print(f"margin L_2nd - L_max: median {np.median(mar):.1f} = terms in x_max {np.median(t1):.1f} + terms in x_2nd {np.median(t2):.1f} + other inputs {np.median(mar - t1 - t2):.1f} (each a median)")
print("  (paper p.8: 'the runner-up's trace raises the maximum's logit at least as much as its own'; median margin +12)")

# ---------------------------------------------------------------------------------------------------
section("E. Max-detector misfires on new runners-up within 0.05 of the maximum, by age of the maximum 1..9 (N = 2^21)")
x = np.random.default_rng(3).standard_normal((2 ** 21, 10))
h, g4 = np.zeros((len(x), 16)), []
for t in range(10):
    q = x[:, t:t + 1] * w + h @ Whh.T
    g4.append(q[:, 4] > 0); h = np.maximum(q, 0)
g4 = np.stack(g4, 1)
mb, sb, pm = before(x)
age = IDX[None, :] - pm; tt = np.broadcast_to(IDX[None, :], x.shape)
close = (x < mb) & (x > sb) & (mb - x < 0.05)
final = (mb == x.max(1, keepdims=True)) & (x == np.sort(x, 1)[:, [-2]])
for name, sel in (("every such event at steps 2-9", close & (tt >= 2)), ("only the sequence's final maximum and runner-up", close & final)):
    print(f"  {name}:", np.array([g4[sel & (age == a)].mean() for a in range(1, 10)]))
print("  (paper p.6: 27% at age 1 and 96-98% at ages 5-7)")

# ---------------------------------------------------------------------------------------------------
section("F. The Appendix B program exactly as printed, against the network (N = 2^15)")
ROT = complex(math.cos(.74), math.sin(.74))
COMPARATORS = [1, 6, 7, 8]
TAPE = [0, 2, 3, 10, 11, 12, 13, 14, 15]
GAIN = np.array([10.5, 11, 13, 12])
ALPHA = np.array([
    [0.57, -0.22, -0.18, 0.27],  # neuron 1
    [0.75, -0.04, 0.03, -0.74],  # neuron 6
    [0.76, 0.07, -0.01, -0.04],  # neuron 7
    [0.68, 0.04, -0.32, -0.47],  # neuron 8
])
EMIT = np.array([
    [10.5, -5.5, 0.5, -3.5, -2.0, -2.0, -2.0, 3.0, 0.0, 1.5, 3.0, 1.0],  # neuron 0
    [12.5, 7.5, -0.5, 1.0, -6.5, 0.5, 1.5, -0.5, 1.5, -2.0, 1.5, -1.0],  # neuron 2
    [3.0, 1.5, 1.0, 0.0, -3.0, 0.5, -0.5, 1.5, -3.0, 5.5, -0.5, -8.0],  # neuron 3
    [10.5, -5.5, 0.5, 4.5, -4.0, -0.5, 6.5, 2.5, -0.5, -0.5, -2.5, 3.0],  # neuron 10
    [10.0, -6.0, 3.5, 3.0, -6.5, 4.0, 3.0, -2.0, -3.0, 1.5, 1.5, 1.0],  # neuron 11
    [13.0, -4.0, 5.0, -1.5, -7.5, 6.0, -1.5, -1.0, 2.0, 2.5, -2.0, -3.5],  # neuron 12
    [5.5, 1.5, -0.5, 0.0, -2.0, -0.5, -1.0, 3.5, 0.5, 0.5, 0.0, -3.0],  # neuron 13
    [10.5, -1.5, 0.0, -3.5, -2.0, -2.5, -3.0, -2.5, 0.0, -0.5, -2.5, 8.5],  # neuron 14
    [8.0, -1.5, 3.0, -3.0, -6.0, 3.0, -3.5, 4.0, -4.5, -0.5, 0.0, -5.5],  # neuron 15
])


def step(carry, x):
    if carry is None:
        carry = (0., 0., 1., 1., 1., 0.)
    M, S, zM, zS, reset, increment = carry
    phase = np.array([1., zM.real, zM.imag, reset])
    threshold = S + (M-S)*(ALPHA @ phase)
    h = np.zeros(16)
    h[COMPARATORS] = GAIN * np.maximum(threshold-x, 0.)
    h[4] = 10*max(x-M, 0.)
    h[5] = 12*increment
    h[9] = 1.5*M*reset
    increment = max(x-S, 0.)
    zM, zS = zM*ROT, zS*ROT
    reset = runner_reset = 0.
    if x > M:
        M, S, zM, zS, reset = x, M, 1., zM, 1.
    elif x > S:
        S, zS, runner_reset = x, 1., 1.
    features = [M, S, M*zM.real, M*zM.imag, M*reset]
    for z in (zS, zS**2, zS**3):
        features.extend((S*z.real, S*z.imag))
    features.append(S*runner_reset)
    h[TAPE] = EMIT @ features
    h[14] -= 1.25*x
    return (M, S, zM, zS, reset, increment), np.maximum(h, 0.)


X = np.random.default_rng(1).standard_normal((2 ** 15, 10))
N = len(X); y = np.argsort(-X, 1)[:, 1]
Hn = np.maximum(forward(X)[0], 0)
net = (Hn[:, -1] @ Wout.T).argmax(1)
Hi = np.zeros((N, 10, 16))
for n in range(N):
    carry = (0., 0., complex(1.), complex(1.), 1., 0.)          # the printed initial carry, with complex clocks
    for t in range(10):
        carry, Hi[n, t] = step(carry, float(X[n, t]))
ide = (Hi[:, -1] @ Wout.T).argmax(1)
okn, oki = net == y, ide == y
print(f"network accuracy {okn.mean():.2%}, idealised accuracy {oki.mean():.2%}   (paper 95.3%, 96.6%)")
print(f"agreement {np.mean(net == ide):.2%}, network's mistakes kept {np.mean(ide[~okn] == net[~okn]):.1%}   (paper 93.2%, 6%)")
print(f"both right {np.mean(okn & oki):.2%}, only idealised {np.mean(~okn & oki):.2%}, only network {np.mean(okn & ~oki):.2%}   (paper 92.9%, 3.7%, 2.4%)")
xs = -np.sort(-X, 1); r23 = xs[:, 1] - xs[:, 2] < 0.05; nonear = ~r23 & (xs[:, 0] - xs[:, 1] >= 0.05)
print(f"runner-up/third near ties: idealised right {oki[r23].mean():.1%}, network {okn[r23].mean():.1%}; no near tie: idealised {oki[nonear].mean():.2%}, network {okn[nonear].mean():.2%}"
      "   (paper 91%, 78%; 97.3%, 98.4%)")
print("correlation of each emitted neuron with the network's, all steps pooled   (paper: comparators, pulses, n2 0.96-1.00; other tape 0.67-0.88):")
print("   ", {f"n{j}": round(float(np.corrcoef(Hi[:, :, j].ravel(), Hn[:, :, j].ravel())[0, 1]), 3) for j in range(16)})


def replace(idx):
    h = np.zeros((N, 16))
    for t in range(10):
        h = np.maximum(X[:, t:t + 1] * w + h @ Whh.T, 0)
        h[:, idx] = Hi[:, t, idx]
    return (h @ Wout.T).argmax(1)


print(f"its four comparators put in place of the network's together: agreement {np.mean(replace(COMPARATORS) == net):.1%}   (paper 28%)")
print("one neuron replaced at a time:", {f"n{j}": f"{np.mean(replace([j]) == net):.1%}" for j in COMPARATORS + [4, 5, 9]}, "  (paper Table 4: worst comparator 33%)")
print("numeric literals in the printed program: 2 (ROT) + 4 + 9 + 4 + 16 + 108 + 27 in step() = 170")
c0, rad = EMIT[:, 0], np.hypot(EMIT[:, 2], EMIT[:, 3])
print("idealised tape neurons, maximum alone (S = 0, no reset): h_i = M [c_i + rho_i cos(phi - phi_i)]_+ ; c_i =", c0, " rho_i =", np.round(rad, 2), " rho_i/c_i =", np.round(rad / c0, 2))
two = (np.sort(np.where(IDX[None, None, :] <= IDX[None, :, None], X[:, None, :], -np.inf), 2)[:, :, -2] > 0)   # two positive inputs so far
print("share of steps on which a tape neuron is clipped (activation 0), steps with two positive inputs so far; order n0 n2 n3 n10..n15:")
print("    idealised program:", np.round((Hi[:, :, TAPE] == 0)[two].mean(0), 3), f" all nine on: {(Hi[:, :, TAPE] > 0).all(2)[two].mean():.1%}")
print("    network:          ", np.round((Hn[:, :, TAPE] == 0)[two].mean(0), 3), f" all nine on: {(Hn[:, :, TAPE] > 0).all(2)[two].mean():.1%}")
