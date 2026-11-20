"""E6. Is the readout blind to the level and clock modes on data?
Mode decomposition of the final logits:  L = sum_k C_k + U[:, PULSE] p_9,  C_k[p] = R_k[p] m_k  (real modes) or
2 Re(R_k[p] z_k) (rotating pairs), with m = LEFT . h_9[ORD] and R_k = U[:, ORD] RIGHT_k.
seeds 1060..1067 (8 x 65536 = 2^19 N(0,1) sequences)."""
from src import *

R = {n: Uo @ RIGHT[n] for n in NAMES}
print("[EXACT] readout vectors R_k = U[:,ORD] RIGHT_k (per unit mode coordinate; pairs: 2|R| @ phase in deg)")
for nme in ("level", "runner", "clock"):
    v = R[nme]
    if IS_COMPLEX[nme]:
        print(f"  {nme:6s}: " + " ".join(f"{2 * abs(u):5.2f}@{np.degrees(np.angle(u)):4.0f}" for u in v))
    else:
        print(f"  {nme:6s}: " + " ".join(f"{u.real:7.3f}" for u in v) + f"   | sd over positions 0..8: {v.real[:9].std():.3f}; value at 9 minus mean(0..8): {v.real[9] - v.real[:9].mean():.3f}")

stats = {}
parts_keys = NAMES + ["pulses@9"]
acc = dict(n=0)
rows = []
for sd in range(1060, 1068):
    x = sample(65536, sd)
    H, Z = run(x, return_pre=True)
    L = logits(H)
    m = coords(H[:, 9])
    C = {}
    for nme in NAMES:
        C[nme] = 2 * np.real(np.multiply.outer(m[nme], R[nme])) if IS_COMPLEX[nme] else np.multiply.outer(m[nme], R[nme].real)
    C["pulses@9"] = H[:, 9][:, PULSE] @ U[:, PULSE].T
    tot = sum(C.values())
    err = np.abs(tot - L).max()
    allon = (Z[:, 9][:, ORD] > 0).all(1)
    order = np.argsort(-x, axis=1)
    p1, p2 = order[:, 0], order[:, 1]
    Lm = L.copy(); Lm[np.arange(len(x)), p2] = -np.inf; q = Lm.argmax(1)
    rows.append(dict(C={k: v.astype(np.float32) for k, v in C.items()}, L=L, allon=allon, p1=p1, p2=p2, q=q, err=err, x=x,
                     lev=m["level"], clk=np.abs(m["clock"])))
print(f"\nmax |sum of mode contributions + pulses - logits| = {max(r['err'] for r in rows):.2e}")
L = np.concatenate([r["L"] for r in rows]); allon = np.concatenate([r["allon"] for r in rows])
p1 = np.concatenate([r["p1"] for r in rows]); p2 = np.concatenate([r["p2"] for r in rows]); q = np.concatenate([r["q"] for r in rows])
x = np.concatenate([r["x"] for r in rows])
C = {k: np.concatenate([r["C"][k] for r in rows]).astype(np.float64) for k in parts_keys}
lev = np.concatenate([r["lev"] for r in rows]); clk = np.concatenate([r["clk"] for r in rows])
n = len(L); rr = np.arange(n)
print(f"n = {n}; all 13 ordinary units on at t = 9: {allon.mean():.4f} of sequences; accuracy {np.mean(L.argmax(1) == p2):.4f} "
      f"(all-on: {np.mean((L.argmax(1) == p2)[allon]):.4f}, otherwise: {np.mean((L.argmax(1) == p2)[~allon]):.4f})")
print(f"size of the coordinates: mean level coordinate {lev.mean():.2f} (rms {np.sqrt((lev**2).mean()):.2f}); mean |clock| {clk.mean():.2f} (rms {np.sqrt((clk**2).mean()):.2f})")


def cen(V, sl=slice(0, 9)):
    return V[:, sl] - V[:, sl].mean(1, keepdims=True)


def report(mask, title):
    print(f"\n--- {title} (n = {mask.sum()}) ---")
    L0 = cen(L[mask]); tot_rms = np.sqrt((L0 ** 2).mean())
    L10 = cen(L[mask], slice(0, 10)); tot_rms10 = np.sqrt((L10 ** 2).mean())
    mg_tot = (L[rr, p2] - L[rr, q])[mask]; mg1_tot = (L[rr, p2] - L[rr, p1])[mask]
    print(f"total: rms of logits centred over positions 0..8 = {tot_rms:.2f}; over 0..9 = {tot_rms10:.2f}; margin p2-best: mean {mg_tot.mean():.2f}, rms {np.sqrt((mg_tot**2).mean()):.2f}")
    print("part        rms(centred 0..8)  ratio   rms(centred 0..9)  ratio | margin p2-best: mean    rms   | margin p2-p1: mean    rms  | share of sum of squares (0..8)  agree if removed (argmax over 0..9)  agree if removed (argmax over 0..8)")
    full_ans = L[mask].argmax(1); full_ans8 = L[mask][:, :9].argmax(1)
    for k in parts_keys + ["level+clock", "level+clock+runner", "h2+h3+short+trans"]:
        if k == "level+clock": V = C["level"] + C["clock"]
        elif k == "level+clock+runner": V = C["level"] + C["clock"] + C["runner"]
        elif k == "h2+h3+short+trans": V = C["h2"] + C["h3"] + C["short"] + C["trans"]
        else: V = C[k]
        Vm = V[mask]
        c0 = cen(Vm); c10 = cen(Vm, slice(0, 10))
        mg = (V[rr, p2] - V[rr, q])[mask]; mg1 = (V[rr, p2] - V[rr, p1])[mask]
        r2drop = ((c0) ** 2).sum() / (L0 ** 2).sum()            # = 1 - R2 of (L - V) vs L, centred
        ag = np.mean((L[mask] - Vm).argmax(1) == full_ans)
        ag8 = np.mean((L[mask] - Vm)[:, :9].argmax(1) == full_ans8)
        print(f"{k:20s} {np.sqrt((c0**2).mean()):8.3f}  {np.sqrt((c0**2).mean())/tot_rms:6.3f}   {np.sqrt((c10**2).mean()):8.3f}  {np.sqrt((c10**2).mean())/tot_rms10:6.3f} |"
              f" {mg.mean():12.3f} {np.sqrt((mg**2).mean()):7.3f} | {mg1.mean():10.3f} {np.sqrt((mg1**2).mean()):7.3f} | {r2drop:12.4f}   {ag:14.4f}   {ag8:14.4f}")


report(np.ones(n, bool), "ALL sequences")
report(allon, "all 13 ordinary units ON at t = 9")
report(~allon, "at least one ordinary unit clipped at t = 9")
# finer: answers among positions 0..8 only (exclude sequences where the network answers 9 or the truth is 9)
old = (p2 < 9) & (p1 < 9)
report(old & allon, "all 13 ON at t = 9, and neither max nor runner-up is the last input")
report(old & ~allon, "some unit clipped at t = 9, and neither max nor runner-up is the last input")
