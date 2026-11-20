"""I6. Concrete failing inputs (three or fewer non-background values, background -1), with the step-by-step thresholds.
For each: the failing input, a neighbouring input that is answered correctly, the value at which the answer flips,
robustness to the background level, and whether the pulse interventions (b1), (c) repair it."""
from explain import *


def flip(mk, lo, hi, good, n=4000):
    """scan the free value v in [lo, hi]; return the sub-intervals where the answer is wrong (answer != good(v))"""
    vs = np.linspace(lo, hi, n + 1)
    X = np.array([mk(v) for v in vs])
    a = logits(run(X)).argmax(1)
    tgt = second_argmax(X)
    bad = a != tgt
    out = []; start = None
    for i, bflag in enumerate(bad):
        if bflag and start is None: start = vs[i]
        if not bflag and start is not None: out.append((start, vs[i - 1])); start = None
    if start is not None: out.append((start, vs[-1]))
    return out


def report(title, mk, v_bad, v_good, lo, hi, note=""):
    x = mk(v_bad)
    ans, lg = explain(x, title)
    print(f"interventions on the failing input: (b1) -> {answer_iv(x, {'b1'})}, (c) -> {answer_iv(x, {'c'})}, (d) -> {answer_iv(x, {'b1', 'b2', 'c'})}, (a) -> {answer_iv(x, {'a'})}")
    xg = mk(v_good)
    og = np.argsort(-xg, kind='stable')
    print(f"neighbouring input with the free value {v_good:g}: answer {answer(xg)}, correct {og[1]}")
    iv = flip(mk, lo, hi, None)
    print(f"free value scanned over [{lo}, {hi}]: wrong for v in " + (", ".join(f"[{a:.4f}, {b:.4f}]" for a, b in iv) if iv else "(nowhere)"))
    for b in (-0.3, -0.5, -2.0):
        xb = mk(v_bad, b); ob = np.argsort(-xb, kind='stable')
        print(f"   background {b}: answer {answer(xb)} (correct {ob[1]})", end=";")
    print()
    if note: print(note)


# 1. zero reference: a negative runner-up crosses no threshold
report("E1  negative runner-up (zero reference)", lambda v, b=-1.0: seq({2: 1.0, 4: v}, b), -0.2, 0.2, -0.9, 0.9)
# 2. false record: the runner-up arrives at the last step above the sagged threshold of unit 4
report("E2  false record at the last step (sagged theta_4)", lambda v, b=-1.0: seq({0: 1.0, 9: v}, b), 0.98, 0.94, 0.5, 1.5)
# 3. missed record: a new max 2% above the old one, 8 steps later
report("E3  missed record (new max too close to the old one)", lambda v, b=-1.0: seq({0: 1.0, 8: v}, b), 1.02, 1.06, 0.5, 1.5)
# 4. a later third above the low threshold of unit 5
report("E4  later third wins through the unit-5 pulse", lambda v, b=-1.0: seq({0: 1.0, 3: 0.5, 5: v}, b), 0.49, 0.47, 0.0, 0.9)
# 5. late runner-up fails to displace the earlier third
report("E5  late runner-up loses to an earlier third", lambda v, b=-1.0: seq({0: 1.0, 1: 0.5, 7: v}, b), 0.54, 0.58, 0.0, 0.9)
# 6. third at the last position crosses comparator thresholds that lie below the runner-up
report("E6  third at the last position beats an old runner-up", lambda v, b=-1.0: seq({0: 0.5, 6: 1.0, 9: v}, b), 0.45, 0.38, 0.0, 0.9)
# 7. adjacent: the runner-up right after the third
report("E7  runner-up immediately after the third", lambda v, b=-1.0: seq({0: 1.0, 3: 0.5, 4: v}, b), 0.55, 0.60, 0.0, 0.9)
# ---- supplementary numbers quoted in findings.md
print("\n## supplementary: thresholds of unit 5 and local slopes of the two competing logits")


def u5_threshold(mk, p, lo=-1.0, hi=1.2, n=22000):
    """smallest value of the free input at position p for which unit 5 fires at step p + 1"""
    vs = np.linspace(lo, hi, n + 1)
    H = run(np.array([mk(v) for v in vs]))
    on = H[:, p + 1, 5] > 0
    return vs[on.argmax()] if on.any() else np.nan


def slopes(mk, v, qa, qb, dv=1e-4):
    la = logits(run(np.array([mk(v), mk(v + dv)])))
    return (la[1, qa] - la[0, qa]) / dv, (la[1, qb] - la[0, qb]) / dv


for name, mk, p, v, qa, qb in (("E4", lambda v: seq({0: 1.0, 3: 0.5, 5: v}), 5, 0.49, 5, 3), ("E5", lambda v: seq({0: 1.0, 1: 0.5, 7: v}), 7, 0.54, 7, 1),
                               ("E7", lambda v: seq({0: 1.0, 3: 0.5, 4: v}), 4, 0.55, 4, 3)):
    sa, sb = slopes(mk, v, qa, qb)
    print(f"{name}: unit 5 fires at step {p+1} iff x_{p} > {u5_threshold(mk, p):.3f};  d logit[{qa}]/dx_{p} = {sa:.1f}, d logit[{qb}]/dx_{p} = {sb:.1f}")
for name, mk, p, v, qa, qb in (("E2", lambda v: seq({0: 1.0, 9: v}), 9, 0.98, 9, 0), ("E3", lambda v: seq({0: 1.0, 8: v}), 8, 1.02, 8, 0),
                               ("E6", lambda v: seq({0: 0.5, 6: 1.0, 9: v}), 9, 0.45, 9, 0)):
    sa, sb = slopes(mk, v, qa, qb)
    print(f"{name}: d logit[{qa}]/dx_{p} = {sa:.1f}, d logit[{qb}]/dx_{p} = {sb:.1f}")
# 8. no positive input (runner-up not at position 0)
x = np.array([-1.0, -0.6, -1.4, -0.1, -0.8, -0.3, -1.2, -0.5, -0.9, -0.7])
explain(x, "E8  no positive input (runner-up at 5)")
