"""Linear probes for cos(w*a) and sin(w*a), a = steps since the max (a_1 = t - p_1) or the 2nd max
(a_2 = t - p_2), window x_0..x_t, for each rotation frequency w of W_hh's 13 non-pulse units.
States with a = 0 are excluded. One affine least-squares probe per target on all 16 units, shared by
all t. Data: iid N(0,1), 200k train / 50k test sequences (linear-probes/linear_probes.data, seed 0).
Writes rotate/clock_probes.png and rotate/clock_probes.md. The .md also has two HTML tables with one
column per unit: the probe weights (unit_table) and the eigenmodes of W_hh (mode_rows)."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np

HIDE = Path(__file__).resolve().parent
ROOT = HIDE.parent.parent
sys.path.insert(0, str(HIDE.parent / "linear-probes"))
sys.path.insert(0, str(HIDE.parent))
from linear_probes import data, T, WEIGHTS
from plot_weights import INK, MUTED, ORDER, UNIT_COLOR, plt

FREQS = {"main clock": 42.75, "2nd harmonic": 83.27, "3rd harmonic": 124.78, "short-lived clock": 42.98,
         "alternating": 180.0}  # alternating: cos(180° a) = (-1)^a, and sin = 0, so only the cos probe exists
SKIP = "short-lived clock"      # left out of the tables (still in the figure): same probes as the main clock
FIG_FREQS = list(FREQS)[:4]     # the figure keeps its four panels per row
ANCHORS = {"max": 3, "2nd max": 4}                     # index of age_k in targets(): Y[0, 3 + k]
FRAC = 0.25     # unit u takes part in a row if |z_u| >= FRAC * max_u |z_u| in that row
PULSE = (4, 5, 9)
MODES = {"main clock": (0.951, 42.75), "2nd harmonic": (0.937, 83.27), "3rd harmonic": (0.837, 124.78),
         "short-lived clock": (0.735, 42.98), "alternating": (0.928, 180), "max level": (1.0098, 0),
         "runner-up level": (0.9736, 0), "fast decay": (0.231, 0), "null": (0.001, 0)}   # (|lambda|, angle°)


def dial(phi, col):
    """Inline SVG: circle with an arrow at angle phi (radians), 0 = right, 90° = up."""
    c, s = np.cos(phi), -np.sin(phi)                   # SVG y points down
    tip, base = (12 + 9 * c, 12 + 9 * s), (12 + 4 * c, 12 + 4 * s)
    head = [tip, (base[0] - 3.5 * s, base[1] + 3.5 * c), (base[0] + 3.5 * s, base[1] - 3.5 * c)]
    return (f'<svg width="24" height="24" viewBox="0 0 24 24">'
            f'<circle cx="12" cy="12" r="11" fill="none" stroke="{col}" stroke-width="1"/>'
            f'<line x1="12" y1="12" x2="{base[0]:.1f}" y2="{base[1]:.1f}" stroke="{col}" stroke-width="2"/>'
            f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in head)}" fill="{col}"/></svg>')


def unit_table(head, rows):
    """HTML table: the columns in head, then one column per unit (grouped order). rows: (leading
    cells, z) with one complex z_u per unit (NaN = unit not in the system, shown as –).
    Cell colour: |z_u| / max_u |z_u| of that row; arrow: arg z_u; blank below FRAC of the row max."""
    th = "".join(f"<th>{h}</th>" for h in head) + "".join(f'<th style="color:{UNIT_COLOR[u]}">{u}</th>' for u in ORDER)
    out = ["<table>", f"<tr>{th}</tr>"]
    for lead, z in rows:
        f, cells = np.abs(z) / np.nanmax(np.abs(z)), [f"<td>{c}</td>" for c in lead]
        for u in ORDER:
            if not f[u] >= FRAC:
                cells.append('<td style="text-align:center">–</td>' if np.isnan(f[u]) else "<td></td>")
                continue
            bg = "#%02x%02x%02x" % tuple(int(255 + f[u] * (c - 255)) for c in (0x8a, 0x2b, 0x0e))
            fg = "white" if f[u] > 0.5 else INK
            cells.append(f'<td style="background:{bg};text-align:center">{dial(np.angle(z[u]), fg)}</td>')
        out.append("<tr>" + "".join(cells) + "</tr>")
    return out + ["</table>"]


def importance_table(tab):
    """HTML table: per probe, the R^2 lost when one unit (or a whole group) is left out and the probe
    is refit. Number: loss in units of 0.001; colour: loss relative to the largest single-unit loss."""
    th = ("".join(f"<th>{h}</th>" for h in ["frequency", "steps since", "R²"])
          + "".join(f'<th style="color:{UNIT_COLOR[u]}">{u}</th>' for u in ORDER)
          + "".join(f"<th>{g}</th>" for g in GROUP_DROPS))
    out = ["<table>", f"<tr>{th}</tr>"]
    for r in tab:
        loss = r["r2"] - r["drop"]
        cells = [f"<td>{r['freq']}</td><td>{r['anchor']}</td><td><b>{r['r2']:.3f}</b></td>"]
        for u in ORDER:
            f = max(loss[u], 0) / loss.max()
            bg = "#%02x%02x%02x" % tuple(int(255 + f * (c - 255)) for c in (0x8a, 0x2b, 0x0e))
            cells.append(f'<td style="background:{bg};color:{"white" if f > 0.5 else INK};text-align:right">'
                         f'{1000 * loss[u]:.0f}</td>')
        cells += [f'<td style="text-align:right">{1000 * (r["r2"] - d):.0f}</td>' for d in r["drop_group"]]
        out.append("<tr>" + "".join(cells) + "</tr>")
    return out + ["</table>"]


def mode_rows(H, a):
    """Eigenmodes of W_hh on the 13 non-pulse units, as rows for unit_table: z = conj(v).
    H: states (n, 16); a: their steps since the max. The overall phase (or sign) of each v is fixed so
    that the mode coordinate c = l^H h (l: left eigenvector) has mean phase w*a: arg <c e^(-i w a)> = 0."""
    keep = [u for u in range(16) if u not in PULSE]
    lam, V = np.linalg.eig(np.load(WEIGHTS)["W_hh"][np.ix_(keep, keep)])
    C = H[:, keep] @ np.linalg.inv(V).T                # C[:, m]: coordinate of mode m
    out = []
    for name, (r, deg) in MODES.items():
        m = np.argmin(np.abs(lam - r * np.exp(1j * np.radians(deg))))
        g = (C[:, m] * np.exp(-1j * np.angle(lam[m]) * a)).mean()       # <c e^(-i w a)>
        v = np.full(16, np.nan, complex)
        v[keep] = V[:, m] * g / abs(g)
        out.append(([name, f"{np.degrees(np.angle(lam[m])):.2f}°", f"{abs(lam[m]):.4f}",
                     f"{abs(g) / np.abs(C[:, m]).mean():.2f}"], v.conj()))
    return out

(_, _), (Htr, Hte), (Ytr, Yte) = data(200_000, 0)


def rows(H, Y, j):
    """Stack states with defined a >= 1: features (with bias), a, t."""
    a = Y[0, j]                                        # (T, n), incl window
    m = ~np.isnan(a) & (a >= 1)
    tt = np.broadcast_to(np.arange(T)[:, None], a.shape)
    return np.c_[H[m], np.ones(m.sum())], a[m], tt[m]


def r2(y, p):
    """R^2 over all columns of y together; NaN for a constant target (sin(180° a) = 0)."""
    sst = ((y - y.mean(0)) ** 2).sum()
    return 1 - ((y - p) ** 2).sum() / sst if sst > 1e-9 else np.nan


def r2_without(drop, G, g, Ge, ge, yy, sst):
    """Test R^2 (cos and sin together) of the probe refit without the units in drop, from the Gram
    matrices: train G = F'F, g = F'Y; test Ge, ge, yy = sum(Y^2). SSE = yy - 2 b.ge + b.Ge.b."""
    S = [i for i in range(len(G)) if i not in drop]
    b = np.linalg.solve(G[np.ix_(S, S)], g[S])
    return 1 - (yy - 2 * (b * ge[S]).sum() + (b * (Ge[np.ix_(S, S)] @ b)).sum()) / sst


GROUP_DROPS = {"all comparators": (1, 6, 7, 8), "all pulses": PULSE,
               "all wave units": tuple(u for u in range(16) if u not in (1, 6, 7, 8) + PULSE)}
res, preds = [], {}
for an, j in ANCHORS.items():
    Ftr, atr, _ = rows(Htr, Ytr, j)
    Fte, ate, tte = rows(Hte, Yte, j)
    sig = Ftr[:, :-1].std(0)                           # spread of each unit over the kept training states
    G, Ge = Ftr.T @ Ftr, Fte.T @ Fte
    for fn, deg in FREQS.items():
        w = np.radians(deg)
        Ytrg = np.c_[np.cos(w * atr), np.sin(w * atr)]
        Yteg = np.c_[np.cos(w * ate), np.sin(w * ate)]
        B = np.linalg.lstsq(Ftr, Ytrg, rcond=None)[0]
        P = Fte @ B
        preds[an, fn] = (P, ate)
        gram = (G, Ftr.T @ Ytrg, Ge, Fte.T @ Yteg, (Yteg ** 2).sum(), ((Yteg - Yteg.mean(0)) ** 2).sum())
        rec = dict(anchor=an, freq=fn, deg=deg, n=len(ate), r2=r2(Yteg, P),
                   r2_cos=r2(Yteg[:, 0], P[:, 0]), r2_sin=r2(Yteg[:, 1], P[:, 1]),
                   z=(B[:-1, 0] + 1j * B[:-1, 1]) * sig,
                   drop=np.array([r2_without((u,), *gram) for u in range(16)]),      # R^2 without unit u
                   drop_group=[r2_without(d, *gram) for d in GROUP_DROPS.values()])
        ang_err = np.degrees(np.angle(np.exp(1j * (np.arctan2(P[:, 1], P[:, 0]) - w * ate))))
        rec["med_abs_angle_err"] = np.median(np.abs(ang_err))
        for ai in range(1, T):
            rec[f"err_a{ai}"] = np.median(np.abs(ang_err[ate == ai]))
        res.append(rec)

# Figure: predicted (cos, sin) per state, coloured by a; true points on the unit circle.
fig, axes = plt.subplots(2, 4, figsize=(13, 7))
rng = np.random.default_rng(0)
for (i, an), (k, fn) in [(x, y) for x in enumerate(ANCHORS) for y in enumerate(FIG_FREQS)]:
    ax = axes[i, k]; P, a = preds[an, fn]; w = np.radians(FREQS[fn])
    s = rng.choice(len(a), 4000, replace=False)
    sc = ax.scatter(P[s, 0], P[s, 1], c=a[s], cmap="viridis", vmin=1, vmax=9, s=3, alpha=0.5, lw=0)
    th = np.linspace(0, 2 * np.pi, 200); ax.plot(np.cos(th), np.sin(th), color=MUTED, lw=0.6)
    for ai in range(1, 10):
        m = a == ai
        ax.plot(*np.median(P[m], 0), "o", ms=6, mfc=plt.cm.viridis((ai - 1) / 8), mec=INK, mew=0.8)
        ax.annotate(str(ai), (np.cos(w * ai), np.sin(w * ai)), ha="center", va="center", fontsize=7, color=INK,
                    xytext=(1.18 * np.cos(w * ai), 1.18 * np.sin(w * ai)), textcoords="data")
    r = next(r for r in res if r["anchor"] == an and r["freq"] == fn)
    ax.set_title(f"{an}, {fn} ({FREQS[fn]}°)\n$R^2$ cos {r['r2_cos']:.2f}, sin {r['r2_sin']:.2f}", fontsize=9, color=INK)
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45); ax.set_aspect("equal")
    ax.axhline(0, color=MUTED, lw=0.4); ax.axvline(0, color=MUTED, lw=0.4)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
for ax in axes[-1]: ax.set_xlabel("predicted cos(ωa)")
for ax in axes[:, 0]: ax.set_ylabel("predicted sin(ωa)")
fig.suptitle("Linear probes for the clock phase ωa, a = steps since max / 2nd max (a ≥ 1, shared over t, test set)\n"
             "dots: states; circles: median per a; numbers: true phase of a on the unit circle", color=INK)
fig.colorbar(sc, ax=axes, fraction=0.015, pad=0.02, label="a (steps since)")
fig.savefig(ROOT / "rotate" / "clock_probes.png", dpi=150, bbox_inches="tight")

# Markdown table
L = ["# Clock-phase probes", "",
     "Affine least-squares probes on $h_t$ (all 16 units), one per target shared by all $t$. "
     "Targets $\cos(\omega a)$, $\sin(\omega a)$ with $a = t - p_k$ (window $x_0..x_t$), states with $a = 0$ excluded. "
     "iid $\mathcal N(0,1)$ inputs, 200k train / 50k test sequences. Angle error: median $|\operatorname{arg}(\hat c + i\hat s) - \omega a|$. "
     r"Script: `hide/rotate/clock_probes.py`.", "",
     r"The short-lived clock (42.98°) has no rows: for $a \le 9$ its targets are within $9 \times 0.23° \approx 2°$ "
     "of the main clock's (42.75°), so its probes and scores are the same. "
     r"The alternating mode (180°) has only a cos probe: $\cos(180° a) = (-1)^a$ and $\sin(180° a) = 0$. "
     "Its angle error is 0° when the sign of the prediction is right and 180° when it is wrong.", "",
     r"**How good each probe is:** $R^2$ is the held-out $R^2$ of the cos and sin predictions together, "
     r"$1 - \sum |\hat y - y|^2 / \sum |y - \bar y|^2$ with $y = e^{i \omega a}$ (1 = perfect, 0 = no better than a constant). "
     "It is repeated in the per-unit tables below.", "",
     "| frequency | $\omega$ | steps since | $R^2$ | $R^2$ cos | $R^2$ sin | median angle error |", "|---|---|---|---|---|---|---|"]
tab = [r for fn in FREQS if fn != SKIP for r in res if r["freq"] == fn]     # by frequency, then max / 2nd max
for r in tab:
    L.append(f"| {r['freq']} | {r['deg']}° | {r['anchor']} | **{r['r2']:.3f}** | {r['r2_cos']:.3f} | {r['r2_sin']:.3f} | "
             f"{r['med_abs_angle_err']:.1f}° |".replace("nan", "–"))
L += ["", "## Which units each probe uses", "",
      r"The cos and sin probes of a row are combined into one complex weight per unit, "
      r"$z_u = (w^{\cos}_u + i\, w^{\sin}_u)\, \sigma_u$, with $\sigma_u$ the std of $h_u$ over the training states. "
      r"The prediction is then $\hat c + i \hat s = \sum_u z_u (h_u - \mu_u)/\sigma_u + \text{const}$, "
      r"so when $h_u$ is one std above its mean it adds $|z_u|$ in the direction $\arg z_u$ to the predicted $e^{i\omega a}$. "
      r"**Colour:** amplitude $|z_u|$ relative to the largest in its row (darkest = the row's value in the max |z| column). "
      r"**Arrow:** phase $\arg z_u$, the clock phase $\omega a$ the unit votes for (0° points right, 90° up, "
      r"counter-clockwise like $e^{i\omega a}$ in the $(\cos, \sin)$ plane). "
      f"Blank: $|z_u| < {FRAC:g} \max_{{u'}} |z_{{u'}}|$ in that row. "
      "Columns in group order (blue: comparators, orange: pulses, grey: wave units).", ""]
L += unit_table(["frequency", "steps since", "R²", "max |z|"],
                [([r["freq"], r["anchor"], f"<b>{r['r2']:.3f}</b>", f"{np.abs(r['z']).max():.2f}"], r["z"]) for r in tab])
L += ["", "## How important each unit is for each probe", "",
      r"Each probe is refit without unit $u$ and scored again. The cell is the loss "
      r"$R^2_{\text{all 16}} - R^2_{\text{without } u}$ on the held-out set, **in units of 0.001** (so 25 means 0.025). "
      "**Colour:** the loss relative to the largest single-unit loss in its row. "
      "The last three columns leave out a whole group at once. "
      "A unit can have a large weight above and a small loss here, when other units carry the same signal; "
      "a group can then cost far more than its units do one at a time.", ""]
L += importance_table(tab)
Fm, am, _ = rows(Htr, Ytr, ANCHORS["max"])
L += ["", "## Eigenmodes of $W_{hh}$", "",
      r"Eigendecomposition of $W_{hh}$ restricted to the 13 non-pulse units, $A v_m = \lambda_m v_m$, one row per mode "
      r"(a complex-conjugate pair is one row). While the pulse units are 0 and no unit is clipped, "
      r"$h = \sum_m c_m v_m$ and each step multiplies $c_m$ by $\lambda_m$, so a complex mode moves unit $u$ as "
      r"$2\,\mathrm{Re}(c_m v_{m,u}) = 2\,|c_m|\,|v_{m,u}| \cos(\arg c_m + \arg v_{m,u})$, with $\arg c_m$ advancing by "
      r"$\omega_m = \arg \lambda_m$ per step.", "",
      r"**Colour:** $|v_{m,u}|$ relative to the largest in its row. "
      r"**Arrow:** $-\arg v_{m,u}$, the mode phase $\arg c_m$ at which the mode makes unit $u$ largest. "
      r"**Phase convention:** the overall phase of $v_m$ is arbitrary. It is fixed so that the mode coordinate "
      r"$c_m = \ell_m^\dagger h$ ($\ell_m$: left eigenvector) has mean phase $\omega_m a_1$, that is "
      r"$\arg \langle c_m e^{-i \omega_m a_1} \rangle = 0$, over the training states with $a_1 \ge 1$ "
      r"($a_1$: steps since the max). So an arrow is the clock phase $\omega_m a_1$ at which the unit peaks, "
      r'which makes these rows comparable with the "max" rows of the probe table above. '
      r"**Coherence** $= |\langle c_m e^{-i \omega_m a_1} \rangle| / \langle |c_m| \rangle$: near 1, the mode's phase is "
      r"locked to $a_1$; where it is low, only the relative directions within the row mean anything "
      r"(for the modes with $\omega_m = 0$ it only says how consistently $c_m$ keeps one sign). "
      r"For the real modes the arrow is a sign: right, the unit rises with $c_m$; left, it falls; $c_m$ is positive on "
      r"average (for the alternating mode, at even $a_1$). "
      f"Blank: $|v_{{m,u}}| < {FRAC:g} \max_{{u'}} |v_{{m,u'}}|$. –: the pulse units are not part of the 13 × 13 block.", ""]
L += unit_table(["mode", "arg λ", "|λ|", "coherence with a₁"], mode_rows(Fm[:, :-1], am))
L += ["", "## Median angle error by $a$ (degrees)", "",
      "| frequency | steps since | " + " | ".join(f"a={a}" for a in range(1, T)) + " |", "|---|---|" + "---|" * (T - 1)]
for r in tab:
    L.append(f"| {r['freq']} | {r['anchor']} | " + " | ".join(f"{r[f'err_a{a}']:.0f}" for a in range(1, T)) + " |")
(ROOT / "rotate" / "clock_probes.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
