"""A2: gate patterns by rank event of x_t, and conditioned on the events of the two previous steps.

Literal classes (t >= 1), prefix x_0..x_{t-1} with max M and second largest S (S = -inf at t = 1):
   R: x_t > M      N: S < x_t < M      L: x_t < S
Fine classes split these by sign and position (see FINE).  Positive-part classes ("effective"):
   R*: x_t > max(M, 0)     N*: max(S, 0) < x_t < M     L*: x_t < max(S, 0)      (t = 0: R* if x_0 > 0 else L*)
"""
from cc import *

FINE = ["R+  x>M>0", "R+0 x>0>=M (first positive)", "R-  0>x>M (all negative)",
        "N+  q in [0,.25)", "N+  q in [.25,.5)", "N+  q in [.5,.75)", "N+  q in [.75,1)",
        "N-  S<x<0<M", "N-  S<x<M<0",
        "L   r<0 (x<0<S)", "L   r in [0,.25)", "L   r in [.25,.5)", "L   r in [.5,.7)", "L   r in [.7,.9)", "L   r in [.9,1)",
        "L   x<S<=0<M", "L   x<S<M<0"]


def fine_class(x, ev):
    Mp, Sp = ev["Mp"], ev["Sp"]
    Sp0 = np.maximum(Sp, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = (x - Sp0) / (Mp - Sp0)
        r = x / Sp
    f = np.full(x.shape, -1, dtype=np.int8)
    R, Nn, L = x > Mp, (x < Mp) & (x > Sp), x < Sp
    f[R & (Mp > 0)] = 0
    f[R & (x > 0) & (Mp <= 0)] = 1
    f[R & (x < 0)] = 2
    for k, (a, b) in enumerate(((0, .25), (.25, .5), (.5, .75), (.75, 1.0001))):
        f[Nn & (x > 0) & (q >= a) & (q < b)] = 3 + k
    f[Nn & (x < 0) & (Mp > 0)] = 7
    f[Nn & (x < 0) & (Mp < 0)] = 8
    Lp = L & (Sp > 0)
    f[Lp & (r < 0)] = 9
    for k, (a, b) in enumerate(((0, .25), (.25, .5), (.5, .7), (.7, .9), (.9, 1.0001))):
        f[Lp & (r >= a) & (r < b)] = 10 + k
    f[L & (Sp <= 0) & (Mp > 0)] = 15
    f[L & (Mp < 0)] = 16
    f[:, 0] = -1
    return f


def eff_class(x, ev):
    """0 = R*, 1 = N*, 2 = L* for every t (t = 0 included)."""
    Mp0, Sp0 = np.maximum(ev["Mp"], 0), np.maximum(ev["Sp"], 0)
    return np.where(x > Mp0, 0, np.where(x > Sp0, 1, 2)).astype(np.int8)


EFF = ["R*", "N*", "L*"]

if __name__ == "__main__":
    fine, eff, lit, pid = [], [], [], []
    for seed, x, H, Z in batches():
        ev = events(x)
        fine.append(fine_class(x, ev)); eff.append(eff_class(x, ev)); lit.append(ev["cls"])
        pid.append(pattern_id(Z > 0).astype(np.int32))
    fine, eff, lit, pid = map(np.concatenate, (fine, eff, lit, pid))
    assert (fine[:, 1:] >= 0).all()
    nB = len(pid)
    bit = lambda i: (pid >> i) & 1 == 1
    g4 = bit(4)
    comp = sum(((pid >> j) & 1) << k for k, j in enumerate(COMP))           # bit k = comparator COMP[k] on
    cset = lambda v: "{" + ",".join(str(j) for k, j in enumerate(COMP) if not (v >> k) & 1) + "}"   # OFF set

    def top_patterns(mask, k=5):
        u, c = np.unique(pid[mask], return_counts=True)
        o = np.argsort(-c)[:k]
        return "; ".join(f"{pstr(int(u[i]))} {c[i] / mask.sum():.3f}" for i in o), len(u), c[o].sum() / mask.sum()

    def top_comp(mask, k=4):
        c = np.bincount(comp[mask], minlength=16)
        o = np.argsort(-c)[:k]
        return "; ".join(f"off{cset(v)} {c[v] / mask.sum():.3f}" for v in o if c[v] > 0)

    n1 = fine[:, 1:].size
    print("## A2.1 literal classes, steps t >= 1 (n = %d steps)\n" % n1)
    rows = []
    for q in (R_, N_, L_):
        m = lit == q
        rows.append([CLS[q], m.sum(), f"{m.sum() / n1:.4f}", f"{g4[m].mean():.4f}"] + [f"{bit(j)[m].mean():.4f}" for j in COMP]
                    + [top_comp(m)])
    print(md_table(["class", "n", "share", "P(4 on)", "P(1 on)", "P(6 on)", "P(7 on)", "P(8 on)", "comparator OFF sets (top 4)"], rows))

    print("\n## A2.2 fine classes, steps t >= 1\n")
    rows = []
    for f_ in range(len(FINE)):
        m = fine == f_
        rows.append([FINE[f_], m.sum(), f"{m.sum() / n1:.4f}", f"{g4[m].mean():.4f}"] + [f"{bit(j)[m].mean():.4f}" for j in COMP]
                    + [top_comp(m)])
    print(md_table(["fine class", "n", "share", "P(4 on)", "P(1 on)", "P(6 on)", "P(7 on)", "P(8 on)", "comparator OFF sets (top 4)"], rows))

    print("\n## A2.3 most common full patterns per fine class (t >= 1)\n")
    rows = []
    for f_ in range(len(FINE)):
        m = fine == f_
        s, nd, cov = top_patterns(m, 4)
        rows.append([FINE[f_], nd, f"{cov:.3f}", s.replace("; ", "<br>")])
    print(md_table(["fine class", "distinct patterns", "top-4 coverage", "top 4 patterns (freq within class)"], rows))

    print("\n## A2.3b most common full patterns per literal class (t >= 1)\n")
    rows = []
    for q in (R_, N_, L_):
        m = lit == q
        s, nd, cov = top_patterns(m, 6)
        rows.append([CLS[q], nd, f"{cov:.3f}", s.replace("; ", "<br>")])
    print(md_table(["class", "distinct patterns", "top-6 coverage", "top 6 patterns (freq within class)"], rows))

    # conditioning on the previous step(s): effective classes, and literal classes
    for name, c, labels, codes in (("effective (positive-part) classes", eff, EFF, (0, 1, 2)),
                                   ("literal classes", lit, ["R", "N", "L"], (R_, N_, L_))):
        print(f"\n## A2.4 conditioned on the previous step: {name}; steps t >= 2\n")
        rows = []
        for a in codes:
            for b in codes:
                m = np.zeros_like(c, bool)
                m[:, 2:] = (c[:, 1:-1] == a) & (c[:, 2:] == b)
                s, nd, cov = top_patterns(m, 3)
                rows.append([labels[codes.index(a)], labels[codes.index(b)], m.sum(), f"{g4[m].mean():.4f}", f"{bit(5)[m].mean():.4f}",
                             f"{bit(9)[m].mean():.4f}", f"{(comp[m] == 15).mean():.4f}", f"{(comp[m] == 0).mean():.4f}",
                             f"{1 - bit(0)[m].mean():.4f}", nd, s.replace("; ", "<br>")])
        print(md_table(["class t-1", "class t", "n", "P(4 on)", "P(5 on)", "P(9 on)", "P(all 4 comp on)", "P(all 4 comp off)",
                        "P(0 off)", "distinct", "top 3 full patterns"], rows))
        print(f"\n## A2.5 conditioned on the two previous steps: {name}; steps t >= 2 (t >= 3 for literal classes)\n")
        rows = []
        for a0 in codes:
            for a in codes:
                for b in codes:
                    m = np.zeros_like(c, bool)
                    m[:, 2:] = (c[:, :-2] == a0) & (c[:, 1:-1] == a) & (c[:, 2:] == b)
                    if m.sum() == 0:
                        continue
                    s, nd, cov = top_patterns(m, 3)
                    rows.append([labels[codes.index(a0)], labels[codes.index(a)], labels[codes.index(b)], m.sum(),
                                 f"{g4[m].mean():.4f}", f"{bit(5)[m].mean():.4f}", f"{bit(9)[m].mean():.4f}",
                                 f"{1 - bit(0)[m].mean():.4f}", f"{(comp[m] == 15).mean():.4f}", nd, f"{cov:.3f}", s.replace("; ", "<br>")])
        print(md_table(["class t-2", "class t-1", "class t", "n", "P(4 on)", "P(5 on)", "P(9 on)", "P(0 off)", "P(all comp on)",
                        "distinct", "top-3 cov", "top 3 full patterns"], rows))
