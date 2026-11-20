"""Reviewer R2, test 4.
(A) "In the wave units the circle is cut on purpose / by design" (summary, section 8a).  Remove the ReLU of the
    travelling-zero units, singly and jointly, on all steps and only on the step where the max has the matching age
    (unit 0: age 2, unit 12: age 4, unit 11: age 5, unit 10: age 6).  Accuracy and where the changed answers go.
(B) "The offset is needed" (section 8a; arm D's level removal).  Which gates carry the damage?  Level removed at
    t0 (f = 0, state not clipped), later inputs replaced by -|x_t| (arm D's 'negreal' continuation, so no later input
    can change the top two).  The continuation is run (i) freely, (ii) with the gates of the nine wave units frozen
    to those of the unpatched run, (iii) with the gates of the comparators and pulses frozen, (iv) all gates frozen.
python -B t4_cuts_and_level.py
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import LEFT, RIGHT

# ---------------------------------------------------------------- (A)
x = sample(1_000_000, 9501)
truth = second_argmax(x)
st = running_stats(x)
pM = x.argmax(1)
H0, Z0 = run(x, return_pre=True)
base = logits(H0).argmax(1)
a1 = st["a1"]                                         # age of the max of the prefix at each step
posmax = st["M1"] > 0

def run_linear(mask):
    """mask (B, T, 16) bool: where True the unit is linear (no ReLU) at that step."""
    B = len(x); h = np.zeros((B, N))
    for t in range(T):
        z = x[:, t, None] * w + h @ W.T
        h = np.where(mask[:, t], z, np.maximum(z, 0))
    return (h @ U.T).argmax(1)

def report(label, mask):
    a = run_linear(mask)
    ch = a != base
    print(f"  {label:58s}: accuracy {np.mean(a == truth):.4f} ({100 * (np.mean(a == truth) - np.mean(base == truth)):+.2f} pp)   "
          f"changed {ch.mean():.4f}   of changed: -> max {np.mean(a[ch] == pM[ch]):.3f}")

print(f"(A) baseline accuracy {np.mean(base == truth):.4f}, n = {len(x)}")
B = len(x)
age_of = {0: 2, 12: 4, 11: 5, 10: 6}
print(" share of steps (max > 0) on which the unit is cut, at its 'own' age / at all other ages >= 1:")
for u, a in age_of.items():
    own = (a1 == a) & posmax; oth = (a1 >= 1) & (a1 != a) & posmax
    print(f"   unit {u}: {np.mean(Z0[..., u][own] <= 0):.3f} / {np.mean(Z0[..., u][oth] <= 0):.3f};   median pre-activation / M when cut at own age: "
          f"{np.median((Z0[..., u] / st['M1'])[own & (Z0[..., u] <= 0)]):.2f}")
for units in ([0], [12], [11], [10], [10, 11, 12], [0, 10, 11, 12], [15], [3], [0, 3, 10, 11, 12, 13, 14, 15, 2]):
    m = np.zeros((B, T, N), bool); m[:, :, units] = True
    report(f"ReLU removed, units {units}, all steps", m)
for units in ([0], [12], [11], [10], [10, 11, 12], [0, 10, 11, 12]):
    m = np.zeros((B, T, N), bool)
    for u in units:
        m[:, :, u] = (a1 == age_of[u]) & posmax
    report(f"ReLU removed, units {units}, only at own age of the max", m)
for units in ([10, 11, 12],):
    m = np.zeros((B, T, N), bool)
    for u in units:
        m[:, :, u] = ~((a1 == age_of[u]) & posmax)
    report(f"ReLU removed, units {units}, all steps EXCEPT own age", m)

# ---------------------------------------------------------------- (B)
print("\n(B) level removed at t0, later inputs -> -|x_t| (arm D 'negreal'); scored on sequences with S(t0) > 0")
oi = np.array(ORD)
Ll, Rl = LEFT["level"].real, RIGHT["level"].real
xb = sample(300_000, 9502)
Hb = run(xb)
stb = running_stats(xb)
groups = {"free": [], "wave gates frozen": WAVE, "comparator + pulse gates frozen": COMP + PULSE, "all gates frozen": list(range(16))}

def cont(h0, xc, Gref, frozen):
    """continue from h0 on inputs xc; units in `frozen` use the gate of the reference run (Gref: (B, k, 16) bool)."""
    h = h0.copy(); nb = np.zeros(len(h0))
    fr = np.zeros(N, bool); fr[frozen] = True
    for k in range(xc.shape[1]):
        z = xc[:, k, None] * w + h @ W.T
        g = np.where(fr, Gref[:, k], z > 0)
        nb += ((z > 0) != Gref[:, k]).sum(1)
        h = np.where(g, z, 0.0)
    return h, nb

for t0 in (3, 5, 7):
    sel = (stb["a1"][:, t0] >= 2) & (stb["M1"][:, t0] > 0) & (stb["M2"][:, t0] > 0)
    xs, h_t0 = xb[sel], Hb[sel, t0]
    xc = -np.abs(xs[:, t0 + 1:])
    xfull = np.concatenate([xs[:, :t0 + 1], xc], 1)
    tr = second_argmax(xfull)
    Href, Zref = run(xc, return_pre=True, h0=h_t0)
    Gref = Zref > 0
    ref_ans = (Href[:, -1] @ U.T).argmax(1)
    mlev = h_t0[:, oi] @ Ll
    print(f"  t0 = {t0}: n = {sel.sum()}, reference accuracy {np.mean(ref_ans == tr):.4f}")
    for f in (0.0, 0.5):
        h0 = h_t0.copy(); h0[:, oi] += (f - 1) * mlev[:, None] * Rl
        for label, frozen in groups.items():
            h, nb = cont(h0, xc, Gref, frozen)
            ans = (h @ U.T).argmax(1)
            print(f"     level x {f:.1f}, {label:32s}: accuracy {np.mean(ans == tr):.4f}   agreement with reference {np.mean(ans == ref_ans):.4f}")
    # control: clock x 1.5 (the lead's 'a circle too large for its centre gets cut'), same gate freezing
    Lc, Rc = LEFT["clock"], RIGHT["clock"]
    zc = h_t0[:, oi] @ Lc
    for f in (1.5,):
        h0 = h_t0.copy(); h0[:, oi] += (f - 1) * 2 * np.real(zc[:, None] * Rc)
        for label, frozen in groups.items():
            h, nb = cont(h0, xc, Gref, frozen)
            ans = (h @ U.T).argmax(1)
            print(f"     clock x {f:.1f}, {label:32s}: accuracy {np.mean(ans == tr):.4f}   agreement with reference {np.mean(ans == ref_ans):.4f}")
    for f in (1.5,):
        h0 = h_t0.copy(); h0[:, oi] += (f - 1) * mlev[:, None] * Rl
        for label, frozen in groups.items():
            h, nb = cont(h0, xc, Gref, frozen)
            ans = (h @ U.T).argmax(1)
            print(f"     level x {f:.1f}, {label:32s}: accuracy {np.mean(ans == tr):.4f}   agreement with reference {np.mean(ans == ref_ans):.4f}")
