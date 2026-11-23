"""Figures for the measured section of tape.md.
    LHS = (s_tape_t[9 - a] - s_tape_t[9 - a_S]) / M_t,     u = (S_t - x_(t-a)) / M_t
  ../tape_eq11.png   LHS against -Phi(u), (a) with one Phi for all entries, (b) with one Phi per
                     configuration (a, a_M, a_S) and per third of S_t/M_t.
  ../tape_phi.png    the single Phi(u) with the spread of the entries around it.
Entries (no filter on sequences): every step t = 4..9, age a >= 2, input below S_t, maximum and runner-up of
x_0..x_t both at least two steps old, S_t > 0.  A state at step t is the final state of (0,..,0,x_0..x_t).
Run: python -B hide/plot_tape_eq11.py"""
from matplotlib.colors import LogNorm, LinearSegmentedColormap

from fig_mechanism import *                        # style helpers, colours, plt, common
from modes import RIGHT

B = 300_000
x = sample(B, 3838); Uo = U[:, ORD]; ar = np.arange(B)
rows = []
for t in range(4, 10):
    xr = np.zeros((B, 10)); xr[:, 9 - t:] = x[:, :t + 1]
    c = coords(run(xr)[:, 9])
    G = np.real(np.multiply.outer(c["alt"], Uo @ RIGHT["alt"]))
    for n in ("short", "h2", "h3"):
        G = G + 2 * np.real(np.multiply.outer(c[n], Uo @ RIGHT[n]))
    pre = x[:, :t + 1]; order = np.argsort(-pre, 1); pM, pS = order[:, 0], order[:, 1]
    M, S = pre[ar, pM], pre[ar, pS]; aM, aS = t - pM, t - pS
    for a in range(2, t + 1):
        q = (aM >= 2) & (aS >= 2) & (S > 0) & (pre[:, t - a] < S)
        rows.append(np.stack([np.full(q.sum(), a), aM[q], aS[q], (S[q] - pre[q, t - a]) / M[q], S[q] / M[q],
                              (G[q, 9 - a] - G[ar[q], 9 - aS[q]]) / M[q]], 1))
a_, aM_, aS_, u, sm, lhs = np.concatenate(rows).T
R2 = lambda y, yh: 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()

edges = np.concatenate([[0], np.geomspace(0.005, 6, 60)]); mid = 0.5 * (edges[:-1] + edges[1:]); k = np.digitize(u, edges) - 1
phi = np.array([-np.median(lhs[k == j]) for j in range(len(mid))])
rhs1 = -np.interp(u, mid, phi)
ub = np.concatenate([[0], np.geomspace(0.01, 6, 24)])
cell = ((a_ * 100 + aM_ * 10 + aS_) * 10 + np.digitize(sm, [0.4, 0.7])).astype(int)
code = np.unique(cell * 100 + (np.digitize(u, ub) - 1), return_inverse=True)[1]
o = np.argsort(code, kind="stable"); cs = code[o]; st_ = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
rhs2 = np.empty_like(lhs); rhs2[o] = np.repeat([np.median(v) for v in np.split(lhs[o], st_[1:])], np.diff(np.r_[st_, len(cs)]))
print(f"{len(u)} entries; one Phi: R2 {R2(lhs, rhs1):.3f}; Phi per configuration: R2 {R2(lhs, rhs2):.3f}")

cmap = LinearSegmentedColormap.from_list("seq", ["#d6e6fa", "#2a78d6", "#0c2f5e"])
fig, axs = plt.subplots(1, 2, figsize=(13, 6.2), facecolor=SURF)
for ax, rhs, title, xl in ((axs[0], rhs1, "a  One Φ for all entries", "−Φ(u)"),
                           (axs[1], rhs2, "b  A separate Φ for each configuration", "−Φ(u) of the entry's configuration (a, a_M, a_S, third of S/M)")):
    style(ax, f"{title}:  R² = {R2(lhs, rhs):.2f}")
    lim = (-60, 12) if ax is axs[1] else (-22, 1)
    hb = ax.hist2d(rhs, lhs, bins=[110, 160], range=[lim, (-70, 30)], cmap=cmap, norm=LogNorm(vmin=1), cmin=1)
    ax.plot(lim, lim, color=INK, lw=1.2, label="left side = right side")
    qe = np.linspace(lim[0], lim[1], 30); qm = 0.5 * (qe[:-1] + qe[1:]); kb = np.digitize(rhs, qe) - 1
    qs = np.array([np.percentile(lhs[kb == j], [25, 75]) if (kb == j).sum() > 300 else [np.nan] * 2 for j in range(len(qm))])
    ax.plot(qm, qs[:, 0], color=ORANGE, lw=1.6, label="quartiles of the left side"); ax.plot(qm, qs[:, 1], color=ORANGE, lw=1.6)
    ax.set_xlabel(f"right side without ε:  {xl}   (units of M)")
axs[0].set_ylabel("left side:  ( s_tape[9 − a] − s_tape[9 − a_S] ) / M")
legend(axs[0], loc="upper left")
cb = fig.colorbar(hb[3], cax=fig.add_axes([0.92, 0.3, 0.012, 0.45])); cb.set_label("entries per bin", color=INK2, fontsize=8.5)
cb.ax.tick_params(colors=INK2, labelsize=8, length=0); cb.outline.set_visible(False)
fig.text(0.02, 0.012, f"{len(u):,} tape entries from {B:,} random sequences, steps 4–9: age at least 2, input below the runner-up so far, maximum and runner-up so far both at least two steps old,\n"
         "runner-up positive.  u = (S − x)/M.  Each Φ is a median curve, so the medians lie on the line by construction.", fontsize=8, color=INK2, va="bottom")
fig.subplots_adjust(left=0.06, right=0.9, top=0.93, bottom=0.15, wspace=0.12)
fig.savefig(OUT / "tape_eq11.png", dpi=150, facecolor=SURF)
print("wrote", OUT / "tape_eq11.png")

# ---- Phi itself (line), with the spread of single entries around it (bands)
pq = np.array([np.percentile(-lhs[k == j], [10, 25, 75, 90]) for j in range(len(mid))])
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8), facecolor=SURF)
for ax, umax, title in ((axs[0], 2.5, "a  Φ(u), with the spread of single entries around it"), (axs[1], 0.2, "b  The same, zoomed in on u < 0.2")):
    style(ax, title)
    s_ = mid <= umax
    ax.fill_between(mid[s_], pq[s_, 0], pq[s_, 3], color=BLUE, alpha=0.10, lw=0, label="10% to 90% of single entries")
    ax.fill_between(mid[s_], pq[s_, 1], pq[s_, 2], color=BLUE, alpha=0.22, lw=0, label="quartiles of single entries")
    ax.plot(mid[s_], phi[s_], color=BLUE, lw=2.2, label="Φ(u): the median")
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xlim(0, umax); ax.set_ylim(-12, 60 if umax > 1 else 30)
    ax.set_xlabel("u = (S − x) / M")
axs[0].set_ylabel("Φ(u)   and   −(left side of eq. 11) / M")
legend(axs[0], loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "tape_phi.png", dpi=150, facecolor=SURF)
print("wrote", OUT / "tape_phi.png; slope of Phi for u < 0.1:", np.polyfit(u[u < 0.1], -lhs[u < 0.1], 1).round(2))
