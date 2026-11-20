"""A7: how often and by how much each wave unit is clipped, c_t[i] = ReLU(-z_t[i]), by event history.

Event history: effective classes R*, N*, L* of steps t-2, t-1, t (see a02_events.py); steps since the last R* event
(sR; R* at t gives 0) and since the last N* event (sN).  Steps t >= 1.  'Regular' steps: M_t > 0 (some positive input so far).
"""
from cc import *
from a02_events import eff_class, EFF

WV = WAVE
acc = {}                    # key -> [n, n_clipped (9), sum_c (9)]
samples = {}                # key -> list of arrays of c (only clipped) per unit, subsampled
ratio = []                  # |c_wave| / |h_ord| per regular step
hmean = np.zeros(len(WV)); hcount = np.zeros(len(WV))
cM = {i: [] for i in WV}


def add(key, mask, C):
    n = mask.sum()
    if n == 0:
        return
    c = C[mask]
    a = acc.setdefault(key, [0, np.zeros(len(WV)), np.zeros(len(WV))])
    a[0] += n; a[1] += (c > 0).sum(0); a[2] += c.sum(0)
    s = samples.setdefault(key, [[] for _ in WV])
    for k in range(len(WV)):
        v = c[:, k]; v = v[v > 0]
        s[k].append(v)


for seed, x, H, Z in batches():
    ev = events(x); e = eff_class(x, ev)
    C = np.maximum(-Z[:, :, WV], 0)                         # (B, T, 9)
    B = len(x)
    tt = np.broadcast_to(np.arange(T), (B, T))
    lastR = np.maximum.accumulate(np.where(e == 0, tt, -1), 1)
    lastN = np.maximum.accumulate(np.where(e == 1, tt, -1), 1)
    sR = np.where(lastR >= 0, tt - lastR, -1); sN = np.where(lastN >= 0, tt - lastN, -1)
    t1 = tt >= 1
    add(("all", "t>=1"), t1, C)
    add(("all", "t>=1, regular (M_t>0)"), t1 & (ev["M"] > 0), C)
    for t in range(T):
        add(("t", t), tt == t, C)
    for a in range(-1, 10):
        add(("sR", a), t1 & (sR == a), C)
        add(("sN", a), t1 & (sN == a) & (ev["M"] > 0), C)
    for c0 in range(3):
        add(("cls", c0), t1 & (e == c0), C)
    for c2 in range(3):
        for c1 in range(3):
            for c0 in range(3):
                m = np.zeros((B, T), bool)
                m[:, 2:] = (e[:, :-2] == c2) & (e[:, 1:-1] == c1) & (e[:, 2:] == c0)
                add(("triple", (c2, c1, c0)), m, C)
    reg = t1 & (ev["M"] > 0.5)                 # running max above 0.5, so that ratios to M and to |h| are meaningful
    hn = np.linalg.norm(H[reg], axis=1)        # norm of the whole state (16 units)
    e1 = np.concatenate([np.full((B, 1), -1), e[:, :-1]], 1)
    ratio.append(np.stack([np.linalg.norm(C[reg], axis=1) / hn, e[reg], np.linalg.norm(C[reg][:, 1:], axis=1) / hn, e1[reg]], 1))
    on = Z[:, 1:, WV] > 0
    hmean += (H[:, 1:, WV] * on).sum((0, 1)); hcount += on.sum((0, 1))
    for k, i in enumerate(WV):
        m = reg & (C[:, :, k] > 0)
        cM[i].append(np.stack([C[:, :, k][m] / ev["M"][m], sR[m]], 1))

hmean /= hcount
qs = lambda v: " / ".join(f"{p:.2f}" for p in np.percentile(v, [50, 90, 99])) if len(v) else "-"


def row(key, label):
    n, nc, sc = acc[key]
    return [label, n] + [f"{nc[k] / n:.3f}" for k in range(len(WV))]


def row_c(key, label):
    n, nc, sc = acc[key]
    return [label, n] + [qs(np.concatenate(samples[key][k])) for k in range(len(WV))]


def row_mean(key, label):
    n, nc, sc = acc[key]
    return [label, n] + [f"{sc[k] / n:.3f}" for k in range(len(WV))]


hdr = ["condition", "n steps"] + [f"unit {i}" for i in WV]
print("## A7.1 overall\n")
print("mean activation of each wave unit when on (t >= 1), for scale:  " + ", ".join(f"unit {i}: {hmean[k]:.2f}" for k, i in enumerate(WV)) + "\n")
print("P(clipped)\n"); print(md_table(hdr, [row(k, k[1]) for k in (("all", "t>=1"), ("all", "t>=1, regular (M_t>0)"))]))
print("\nclipped amount c when clipped: 50% / 90% / 99%\n"); print(md_table(hdr, [row_c(k, k[1]) for k in (("all", "t>=1"), ("all", "t>=1, regular (M_t>0)"))]))
print("\nmean clipped amount over all steps (zeros included)\n"); print(md_table(hdr, [row_mean(k, k[1]) for k in (("all", "t>=1"), ("all", "t>=1, regular (M_t>0)"))]))
n, nc, sc = acc[("all", "t>=1")]
print("\nmean clip / mean activation when on: " + ", ".join(f"unit {i}: {sc[k] / n / hmean[k]:.4f}" for k, i in enumerate(WV)))

print("\n## A7.2 by step t\n")
print("P(clipped)\n"); print(md_table(hdr, [row(("t", t), f"t={t}") for t in range(T)]))
print("\nc when clipped: 50% / 90% / 99%\n"); print(md_table(hdr, [row_c(("t", t), f"t={t}") for t in range(T)]))

print("\n## A7.3 by steps since the last R* event (new positive maximum), t >= 1\n")
keys = [("sR", a) for a in range(-1, 10) if ("sR", a) in acc]
lab = lambda a: "no R* yet" if a == -1 else f"sR={a}"
print("P(clipped)\n"); print(md_table(hdr, [row(k, lab(k[1])) for k in keys]))
print("\nc when clipped: 50% / 90% / 99%\n"); print(md_table(hdr, [row_c(k, lab(k[1])) for k in keys]))
print("\nmean c over all steps of the row\n"); print(md_table(hdr, [row_mean(k, lab(k[1])) for k in keys]))
print("\nc / M_t when clipped, steps with running max M_t > 0.5: 50% / 90%\n")
rows = []
for a in range(10):
    r = [f"sR={a}"]
    for i in WV:
        d = np.concatenate(cM[i]); v = d[d[:, 1] == a, 0]
        r.append(" / ".join(f"{p:.3f}" for p in np.percentile(v, [50, 90])) if len(v) >= 200 else "-")
    rows.append(r)
print(md_table(["condition"] + [f"unit {i}" for i in WV], rows))

print("\n## A7.4 by steps since the last N* event (new positive runner-up), regular steps t >= 1\n")
keys = [("sN", a) for a in range(-1, 10) if ("sN", a) in acc]
lab = lambda a: "no N* yet" if a == -1 else f"sN={a}"
print("P(clipped)\n"); print(md_table(hdr, [row(k, lab(k[1])) for k in keys]))
print("\nc when clipped: 50% / 90% / 99%\n"); print(md_table(hdr, [row_c(k, lab(k[1])) for k in keys]))

print("\n## A7.5 by the classes of steps t-2, t-1, t (t >= 2)\n")
keys = [("triple", (a, b, c)) for a in range(3) for b in range(3) for c in range(3)]
lab = lambda k: f"{EFF[k[0]]} {EFF[k[1]]} {EFF[k[2]]}"
print("P(clipped)\n"); print(md_table(hdr, [row(k, lab(k[1])) for k in keys]))
print("\nc when clipped: 50% / 90% / 99%\n"); print(md_table(hdr, [row_c(k, lab(k[1])) for k in keys]))
print("\nby the class of step t alone (t >= 1): P(clipped)\n"); print(md_table(hdr, [row(("cls", c), EFF[c]) for c in range(3)]))

ratio = np.concatenate(ratio)
print("\n## A7.6 size of the wave clip relative to the state, steps t >= 1 with M_t > 0.5:  rho = |c_wave|_2 / |h_t|_2  (h_t: all 16 units)\n")
rows = []
for label, m, col in (("all classes", np.ones(len(ratio), bool), 0), ("R* at t", ratio[:, 1] == 0, 0), ("N* at t", ratio[:, 1] == 1, 0),
                      ("L* at t", ratio[:, 1] == 2, 0), ("L* at t-1 and t", (ratio[:, 1] == 2) & (ratio[:, 3] == 2), 0),
                      ("all classes, unit 0 left out of c", np.ones(len(ratio), bool), 2)):
    r = ratio[m, col]
    rows.append([label, len(r), f"{(r == 0).mean():.4f}", f"{(r < 0.01).mean():.4f}", f"{(r < 0.03).mean():.4f}", f"{(r < 0.1).mean():.4f}",
                 f"{np.median(r):.4f}", f"{np.percentile(r, 90):.4f}", f"{np.percentile(r, 99):.4f}", f"{r.max():.3f}"])
print(md_table(["steps", "n", "P(rho = 0)", "P(rho < 0.01)", "P(rho < 0.03)", "P(rho < 0.1)", "median", "90%", "99%", "max"], rows))
