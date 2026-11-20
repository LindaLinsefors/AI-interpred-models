"""V6 cross-check: per-(aM, aS) c1 from the NATURAL D1 events (seeds 5000..5009) against the
controlled-sampling table (seed 5100). Also the natural-sample D4 split for the D = 7, 8 configurations."""
from v6_data import *

out = open(HERE / "a11_crosscheck.txt", "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(v) for v in a)
    print(s)
    out.write(s + "\n")


e = dict(np.load(HERE / "d1_events.npz"))
y = e["yk"][:, 0]; gap = e["M0"] - e["S0"]; M0 = e["M0"]
aM, aS = e["t"] - e["pM"], e["t"] - e["pS"]
reg = np.load(HERE / "a2_configs.npz")["reg"]
nat, con, ns = [], [], []
P("rows aM = 2..9, columns aS = 1..9: 'natural / controlled' c1 (natural cells with n >= 2000)")
for a in range(2, 10):
    row = []
    for b in range(1, 10):
        m = (aM == a) & (aS == b)
        if a == b or m.sum() < 2000:
            row.append("            -")
            continue
        c_n = np.linalg.lstsq(np.column_stack([gap[m], M0[m]]), y[m], rcond=None)[0][0]
        c_c = np.mean([reg[t, t - a, t - b, 0] for t in range(max(a, b), 10)])
        nat.append(c_n); con.append(c_c); ns.append(m.sum())
        row.append(f"{c_n:6.1f}/{c_c:6.1f}")
    P(f"  aM={a}: " + " ".join(row))
nat, con = np.array(nat), np.array(con)
P(f"{len(nat)} cells: corr = {np.corrcoef(nat, con)[0, 1]:.4f}, median |diff| = {np.median(np.abs(nat - con)):.2f}, max |diff| = {np.abs(nat - con).max():.1f}, "
  f"median relative diff = {np.median(np.abs(nat - con) / np.abs(con)):.3f}")

d = dict(np.load(HERE / "d4_seqs.npz"))
rg = (d["S"] - d["T"]) / d["S"]
edges = [0, .03, .08, .2, .5]
cnt = np.load(HERE / "a2_configs.npz")["cnt"]; nbin = np.load(HERE / "a2_configs.npz")["nbin"]
P("\nD = 7, 8 configurations, P(answer = pT) per bin: natural sample (n) | controlled sample")
for c in ((0, 8, 9), (8, 0, 9), (0, 7, 9), (7, 0, 9), (1, 8, 9), (8, 1, 9), (0, 7, 8), (7, 0, 8)):
    m = (d["pS"] == c[0]) & (d["pT"] == c[1]) & (d["pM"] == c[2])
    cells = []
    for b in range(4):
        mm = m & (rg > edges[b]) & (rg <= edges[b + 1])
        cells.append(f"{np.mean(d['ans'][mm] == d['pT'][mm]):.3f} ({mm.sum()})")
    P(f"  (pS,pT,pM)={c}: " + " ".join(cells) + " | " + " ".join(f"{cnt[c[2], c[0], c[1], b, 1] / nbin[b]:.3f}" for b in range(4)))
out.close()
