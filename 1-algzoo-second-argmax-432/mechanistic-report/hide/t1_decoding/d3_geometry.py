"""d3: how are the ages laid out in the tape coordinates?  Circular statistics on quiet final states (t = 9).
For a complex tape coordinate z_m (eigenvalue rho_m e^{i w_m}) and an age a_X (X = runner-up S, maximum M, third T):
    u = z_m / |z_m|                                 unit phasor
    R(a)   = | mean of u over states with a_X = a |  resultant length (1 = one phase, 0 = no preferred phase)
    phi(a) = arg of that mean                        mean phase
    lock   = | mean over all states of u e^{-i w_m a_X} |   (1 = the phase is  w_m a_X + const: a rotating clock for a_X)
    acc    = accuracy of "a_X = the age whose mean phase phi(a) is nearest to arg z_m" (fitted on train, scored on test)
Output: d3_geometry.txt"""
from dec import *

out = Tee(__file__.replace(".py", ".txt"))
x, h, st, D = data()
B = len(x); n_train = int(0.75 * B); tr = np.arange(B) < n_train
c = tape_coords(h); M = st["M"]
z = {"short": c[:, 1] + 1j * c[:, 2], "h2": c[:, 3] + 1j * c[:, 4], "h3": c[:, 5] + 1j * c[:, 6]}
zc = coords(h)["clock"]
deg = lambda v: np.degrees(np.angle(v))
sel = {"all": np.ones(B, bool), "S before M": D["S_before_M"], "S after M": ~D["S_before_M"]}
out(f"quiet final states {B}; S before M {D['S_before_M'].mean():.3f}")
out("rms of each tape coordinate / M:", np.round(np.sqrt(((c / M[:, None]) ** 2).mean(0)), 2))


def circ(zm, a, w, mask):
    u = zm / np.abs(zm)
    ages = np.unique(a[mask])
    mean = np.array([u[mask & tr & (a == k)].mean() for k in ages])
    lock = np.abs((u * np.exp(-1j * w * a))[mask].mean())
    te = mask & ~tr
    pred = ages[np.real(u[te, None] * np.conj(mean / np.abs(mean))[None]).argmax(1)]
    vals, cnt = np.unique(a[mask & tr], return_counts=True)
    return ages, mean, lock, (pred == a[te]).mean(), (a[te] == vals[cnt.argmax()]).mean()


for X, a in (("a_S", st["aS"]), ("a_M", st["aM"]), ("a_T", st["aT"])):
    out(f"\n================ age {X} ================")
    for nm, zm in list(z.items()) + [("clock (not tape)", zc)]:
        w = np.angle(LAM[nm.split()[0]])
        for sn, mask in sel.items():
            ages, mean, lock, acc, base = circ(zm, a, w, mask)
            out(f"{nm:17s} {sn:11s} lock {lock:.3f}  phase-only accuracy {acc:.3f} (majority {base:.3f})   w = {np.degrees(w):.1f} deg/step")
            if sn != "all":
                out("     age:        " + " ".join(f"{k:6d}" for k in ages))
                out("     R(a):       " + " ".join(f"{np.abs(v):6.2f}" for v in mean))
                out("     phi(a) deg: " + " ".join(f"{deg(v):6.0f}" for v in mean))
                out("     phi - w a:  " + " ".join(f"{deg(v * np.exp(-1j * w * k)):6.0f}" for k, v in zip(ages, mean)))
                out("     med |z|/M:  " + " ".join(f"{np.median((np.abs(zm) / M)[mask & (a == k)]):6.2f}" for k in ages))

out("\n================ alternating coordinate zeta_alt / M ================")
for X, a in (("a_S", st["aS"]), ("a_M", st["aM"]), ("a_T", st["aT"])):
    for sn, mask in sel.items():
        ages = np.unique(a[mask]); v = c[:, 0] / M
        out(f"{X} {sn:11s} age:    " + " ".join(f"{k:6d}" for k in ages))
        out(f"{'':15s} median: " + " ".join(f"{np.median(v[mask & (a == k)]):6.2f}" for k in ages))
        out(f"{'':15s} P(>0):  " + " ".join(f"{(v[mask & (a == k)] > 0).mean():6.2f}" for k in ages))
    par = ((-1.0) ** a) * np.sign(c[:, 0])
    out(f"{X}: P(sign zeta_alt = (-1)^a) = {(par > 0).mean():.3f}   (S before M {(par[D['S_before_M']] > 0).mean():.3f}, S after M {(par[~D['S_before_M']] > 0).mean():.3f})")
