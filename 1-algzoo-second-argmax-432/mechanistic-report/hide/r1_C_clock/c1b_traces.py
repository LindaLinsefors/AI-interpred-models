"""C1b. Exact-trace test of F6.  z_t / K4 = sum_p zeta_{t,p} x_p exactly, zeta_{t,p} = LEFT_clock . J[t,p,ORD] / K4.
F6 (z = K4 (M - S+) rho e^{i phi(a1)}) says: zeta of the max = rho e^{i phi(a1)}, zeta of the runner-up = -(that) when S > 0, all other traces ~ 0.
200k N(0,1) sequences in 8 batches, seeds 1700-1707."""
from cl import *
acc = {}
def add(key, **kw):
    d = acc.setdefault(key, {k: [] for k in kw})
    for k, v in kw.items():
        d[k].append(v)
tot_ss = 0.0; tot_res_ms = 0.0; tot_res_m = 0.0
for seed in range(1700, 1708):
    x = sample(25_000, seed)
    Hh, J = traces(x)
    st = running_stats(x)
    zeta = J[..., ORD] @ LC / K4                                   # (B, T, T)
    del J
    z = Hh[..., ORD] @ LC / K4
    assert np.abs((zeta * x[:, None, :]).sum(-1) - z).max() < 1e-9
    B = x.shape[0]; bi = np.arange(B)
    for t in range(1, 10):
        M, S2 = st["M1"][:, t], st["M2"][:, t]
        P1, P2, P3 = st["p1"][:, t], st["p2"][:, t], st["p3"][:, t]
        A1, A2 = st["a1"][:, t], st["a2"][:, t]
        zm, zs_ = zeta[bi, t, P1], zeta[bi, t, P2]
        rest = z[:, t] - zm * M - zs_ * S2                         # everything that is not the trace of M or S (S may be negative here)
        ok = M > 0
        for a in range(0, t + 1):
            for b in range(0, t + 1):
                if a == b: continue
                q = ok & (A1 == a) & (A2 == b)
                if not q.any(): continue
                qp = q & (S2 > 0)
                add((a, b), zm=zm[q], rS=(-zs_[qp] / zm[qp]) if a >= 1 else np.zeros(0, complex), rest=rest[q], z=z[q, t],
                    sneg=(zs_ * S2)[q & (S2 <= 0)], gap=(M - np.maximum(S2, 0))[q])
cells = {k: {kk: np.concatenate(vv) for kk, vv in v.items()} for k, v in acc.items()}

cmed = lambda v: np.median(v.real) + 1j * np.median(v.imag)          # component-wise median (robust to the ~1% of states with a huge pulse-9 gain)

def table(title, f, width=17):
    print()
    print(title)
    print("a1/a2 " + "".join(f"{b:>{width}d}" for b in range(0, 10)))
    for a in range(1, 10):
        print(f"  {a}   " + "".join(f"{(f(cells[(a, b)]) if (a, b) in cells and cells[(a, b)]['zm'].size >= 300 else '.'):>{width}s}" for b in range(0, 10)))

table("trace of the max, zeta_M: component-wise median as |.|@phase(deg)  [F6: rho e^{i phi(a1)}, the same along each row]", lambda c: fmt(cmed(c["zm"])))
table("spread of zeta_M inside the cell: IQR of |zeta_M| / IQR of its phase (deg) / share of states with |zeta_M| > 2", lambda c: f"{np.subtract(*np.percentile(abs(c['zm']), [75, 25])):.2f}/{np.subtract(*np.percentile(np.degrees(np.angle(c['zm'] / cmed(c['zm']))), [75, 25])):.0f}/{np.mean(abs(c['zm']) > 2):.3f}")
table("runner-up trace relative to the max trace, -zeta_S/zeta_M (states with S > 0): component-wise median as re+im  [F6: 1+0i]", lambda c: f"{cmed(c['rS']).real:+.2f}{cmed(c['rS']).imag:+.2f}i")
table("same: quartiles of Re(-zeta_S/zeta_M)", lambda c: "/".join(f"{v:+.2f}" for v in np.percentile(c["rS"].real, [25, 50, 75])), 19)
table("share of states (S > 0) with |zeta_S/zeta_M + 1| < 0.25  (runner-up trace within 25% of F6)", lambda c: f"{np.mean(abs(c['rS'] - 1) < 0.25):.3f}", 9)
table("share of states (S > 0) with |zeta_S/zeta_M| < 0.1  (the runner-up left no trace in the clock)", lambda c: f"{np.mean(abs(c['rS']) < 0.1):.3f}", 9)
table("rms of everything except the traces of M and S, relative to rms |z|  (third largest, older inputs, current input)", lambda c: f"{np.sqrt(np.mean(abs(c['rest']) ** 2) / np.mean(abs(c['z']) ** 2)):.3f}", 9)

print()
print("Pooled over cells, by a1: median zeta_M; -zeta_S/zeta_M for the old-max runner-up (a2 > a1) and for a runner-up that arrived after the max (a2 < a1): component-wise median, share within 25% of F6, share with no trace")
for a in range(1, 10):
    zm = np.concatenate([cells[(a, b)]["zm"] for b in range(10) if (a, b) in cells])
    old = np.concatenate([cells[(a, b)]["rS"] for b in range(a + 1, 10) if (a, b) in cells] or [np.zeros(0, complex)])
    yng = np.concatenate([cells[(a, b)]["rS"] for b in range(0, a) if (a, b) in cells] or [np.zeros(0, complex)])
    f = lambda r: f"{cmed(r).real:+.3f}{cmed(r).imag:+.3f}i, within 25%: {np.mean(abs(r - 1) < 0.25):.3f}, no trace: {np.mean(abs(r) < 0.1):.3f}, n={r.size}" if r.size else "-"
    print(f"  a1={a}: zeta_M {fmt(cmed(zm))} (IQR of |.| {np.subtract(*np.percentile(abs(zm), [75, 25])):.3f}) | old max as runner-up: {f(old)} | younger runner-up: {f(yng)}")
print()
print("Younger runner-up, by the age of the max when the runner-up arrived (lag = a1 - a2) and by the time since its arrival a2:")
print("  entries: median Re(-zeta_S/zeta_M) / share with no trace (|ratio| < 0.1)   [1 = radius fully updated to M - S]")
print(" lag/a2 " + "".join(f"{b:>13d}" for b in range(0, 8)))
for lag in range(1, 10):
    row = []
    for b in range(0, 8):
        a = lag + b
        row.append(f"{np.median(cells[(a, b)]['rS'].real):+.2f}/{np.mean(abs(cells[(a, b)]['rS']) < 0.1):.2f}".rjust(13) if (a, b) in cells and cells[(a, b)]["rS"].size >= 300 else f"{'.':>13s}")
    print(f"   {lag}    " + "".join(row))
print("same, median Im(-zeta_S/zeta_M) (tangential part)")
for lag in range(1, 10):
    row = []
    for b in range(0, 8):
        a = lag + b
        row.append(f"{np.median(cells[(a, b)]['rS'].imag):+8.2f}" if (a, b) in cells and cells[(a, b)]["rS"].size >= 300 else f"{'.':>8s}")
    print(f"   {lag}    " + "".join(row))
sn = np.concatenate([c["sneg"] for (a, b), c in cells.items() if a >= 1]); zz = np.concatenate([c["z"] for (a, b), c in cells.items() if a >= 1])
print()
print(f"negative runner-up (S <= 0, F6 sets S+ = 0): rms of its actual contribution zeta_S S = {np.sqrt(np.mean(abs(sn) ** 2)):.4f}  (rms |z| over all a1 >= 1 states: {np.sqrt(np.mean(abs(zz) ** 2)):.4f})")
