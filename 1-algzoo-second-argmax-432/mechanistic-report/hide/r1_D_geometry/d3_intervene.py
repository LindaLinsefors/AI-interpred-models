"""D3 (causal). Scale one mode coordinate of h_t0 by f, keep every other mode coordinate and the pulses, continue the run.

  level:  h'_t0[ORD] = h_t0[ORD] + (f - 1) m_level RIGHT_level
  clock:  h'_t0[ORD] = h_t0[ORD] + (f - 1) 2 Re(z_clock RIGHT_clock)
  variant "pure": entries of h' may go negative (pure coordinate scaling);  variant "relu": h' clipped at 0 afterwards.
Continuations:  real  = the real inputs x_{t0+1..9}
                zero  = all later inputs 0            (autonomous flow)
                neg   = all later inputs -2 M         (comparators held on by the input)
                negreal = later inputs -|x_t|         (no later input can be a new max or runner-up; answers stay defined)
Selection: N(0,1), seed 1021, 200k sequences; at t0 the max has age a1 >= 2 and M > 0.
Clock / level comparisons (real inputs) use only sequences whose max is not replaced after t0.
Reference for 'zero' and 'neg' = the unpatched state continued with the same later inputs.
python -B d3_intervene.py
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *

SEED, NB = 1021, 200_000
FS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
x = sample(NB, SEED)
H, Z = run(x, return_pre=True)
st = running_stats(x)
truth = second_argmax(x)
base_ans = logits(H).argmax(1)
print(f"seed {SEED}, n = {NB}; baseline accuracy {np.mean(base_ans == truth):.4f}")
Lc, Rc, Ll, Rl = LEFT["clock"], RIGHT["clock"], LEFT["level"].real, RIGHT["level"].real
oi = np.array(ORD)
med = np.median
q = lambda v: f"{med(v):5.2f}[{np.percentile(v, 25):5.2f},{np.percentile(v, 75):5.2f}]"
qa = lambda v: f"{med(v):+5.0f}[{np.percentile(v, 25):+5.0f},{np.percentile(v, 75):+5.0f}]"


def experiment(t0, which, variant, mode, verbose_units=False):
    sel = (st["a1"][:, t0] >= 2) & (st["M1"][:, t0] > 0)
    xs, Hs = x[sel], H[sel]
    Msel = st["M1"][sel, t0]
    n, k_steps = sel.sum(), 9 - t0
    nomax = st["p1"][sel, 9] == st["p1"][sel, t0]
    Spos = st["M2"][sel, t0] > 0
    h_t0 = Hs[:, t0]
    mlev, zc = h_t0[:, oi] @ Ll, h_t0[:, oi] @ Lc
    xc = {"real": xs[:, t0 + 1:], "zero": np.zeros((n, k_steps)), "neg": np.repeat(-2 * Msel[:, None], k_steps, 1),
          "negreal": -np.abs(xs[:, t0 + 1:])}[mode]
    xfull = np.concatenate([xs[:, :t0 + 1], xc], 1)
    tr = second_argmax(xfull)                                   # truth for the sequence actually fed
    ref_ans = None
    Href, Zref = run(xc, return_pre=True, h0=h_t0)
    zref, lref = Href[..., oi] @ Lc, Href[..., oi] @ Ll
    use = nomax if mode == "real" else np.ones(n, bool)
    ref_ans = (Href[:, -1] @ U.T).argmax(1)
    scored = mode in ("real", "negreal")
    print(f"\n   --- scale {which}; variant {variant}; continuation {mode}; n = {n} (clock/level columns: {use.sum()})")
    print("      f  | neg entries | newly OFF ord units k=1, mean k>=2 | newly ON | unit4 extra, missed |"
          " clock |z'/z| and phase diff (deg): k=1 ; k=2 ; last k | level'/level k=1, last"
          + (" | accuracy  agree(ref)  acc on K  acc on K without extra unit-4 firing (share)   [K = max kept (real) / S(t0) > 0 (negreal)]" if scored else ""))
    for f in FS:
        h0 = h_t0.copy()
        if which == "level":
            h0[:, oi] += (f - 1) * mlev[:, None] * Rl
        else:
            h0[:, oi] += (f - 1) * 2 * np.real(zc[:, None] * Rc)
        neg = (h0[:, oi] < -1e-9).sum(1).mean()
        if variant == "relu":
            h0 = np.maximum(h0, 0)
        Hp, Zp = run(xc, return_pre=True, h0=h0)
        on_ref, on_p = Zref[..., oi] > 0, Zp[..., oi] > 0
        off_new = (on_ref & ~on_p).sum(-1).mean(0)
        on_new = (~on_ref & on_p).sum(-1).mean()
        f4r, f4p = Zref[..., 4] > 0, Zp[..., 4] > 0
        extra4, missed4 = (f4p & ~f4r).any(1), (~f4p & f4r).any(1)
        zp, lp = Hp[..., oi] @ Lc, Hp[..., oi] @ Ll
        cols = []
        for k in (0, 1, k_steps - 1):
            r = zp[use, k] / zref[use, k]
            cols.append(q(abs(r)) + qa(np.degrees(np.angle(r))))
        lv = f"{med(lp[use, 0] / lref[use, 0]):5.2f} {med(lp[use, -1] / lref[use, -1]):5.2f}"
        tail = ""
        if scored:
            ans = (Hp[:, -1] @ U.T).argmax(1)
            keep = nomax if mode == "real" else Spos          # negreal: only sequences whose runner-up at t0 is positive
            k2 = keep & ~extra4
            tail = (f" | {np.mean(ans == tr):.4f}   {np.mean(ans == ref_ans):.4f}   {np.mean(ans[keep] == tr[keep]):.4f}"
                    f"   {np.mean(ans[k2] == tr[k2]):.4f} ({k2.sum() / keep.sum():.2f})")
        print(f"   {f:5.2f} | {neg:5.2f} | {off_new[0]:5.2f} {off_new[1:].mean():5.2f} | {on_new:5.2f} | {np.mean(extra4):.3f} {np.mean(missed4):.3f} | "
              + " ; ".join(cols) + " | " + lv + tail)
        if verbose_units and f == 0.0:
            print("           newly OFF at k=1, per unit " + str(ORD) + ": " + " ".join(f"{v:.2f}" for v in (on_ref & ~on_p)[:, 0].mean(0)))


for t0 in (3, 5, 7):
    sel = (st["a1"][:, t0] >= 2) & (st["M1"][:, t0] > 0)
    Ho = H[sel, t0][:, oi]
    mlev = Ho @ Ll
    print("\n" + "=" * 150)
    print(f"t0 = {t0}: {sel.sum()} sequences with a1(t0) >= 2, M > 0;  baseline accuracy on them {np.mean(base_ans[sel] == truth[sel]):.4f}")
    print(f"   level coordinate / (10.157 M): median {med(mlev / (10.157 * st['M1'][sel, t0])):.3f};  mean share of |h_t0[ORD]|^2 carried by the level part:"
          f" {np.mean(1 - ((Ho - mlev[:, None] * Rl) ** 2).sum(1) / (Ho ** 2).sum(1)):.3f}")
    for variant in ("pure", "relu"):
        for mode in ("real", "negreal", "zero", "neg"):
            experiment(t0, "level", variant, mode, verbose_units=(variant == "pure"))
    for mode in ("real", "negreal", "zero"):
        experiment(t0, "clock", "pure", mode)
