"""B6. Causal checks.
(i)  scale the level coordinate of h_t0 by f (other mode coordinates and the pulses unchanged), run on.
(ii) replace unit 4's activation by the oracle 10.157*max(x_t - Mp_{t-1}, 0) at every step.
Data: N(0,1); (i) seed 1060 (400k); (ii) seeds 1061, 1062 (400k each).
"""
from blib import *

L = LEFT["level"].real
R = RIGHT["level"].real
lam = LAM["level"].real

print("=== [EXACT] what one unit of level coordinate (per 10.157) does at the next step ===")
print(f"  unit 4 threshold: d theta_4 = {C4['level'].real * W4:+.4f} per unit of level/10.157")
kap = {j: -(W[j, ORD] @ R) / w[j] * W4 for j in COMP}
for j in COMP:
    print(f"  comparator {j} threshold: d theta_{j} = {kap[j]:+.4f} per unit of level/10.157   (W[{j},ORD].R_level = {W[j, ORD] @ R:+.4f} = lam*R_level[{j}] = {lam * R[OI[j]]:+.4f})")
for q in (5, 9):
    print(f"  pulse {q} pre-activation: {W[q, ORD] @ R:+.4f} per unit of level  ({(W[q, ORD] @ R):+.4f} x 10.157 per unit of level/10.157)")
ro = U[:, ORD] @ R
print("  readout U[:, ORD].R_level by position:", np.round(ro, 3), " -> position 9 minus mean of the others:", round(ro[9] - ro[:9].mean(), 3))


def perturb(h, f):
    lev = h[:, ORD] @ L
    h2 = h.copy()
    h2[:, ORD] += ((f - 1) * lev)[:, None] * R[None, :]
    return h2


x = sample(400_000, 1060)
n = len(x)
H, Z = run(x, return_pre=True)
st = prefix_stats(x)
base_ans = logits(H).argmax(1)
srt = np.argsort(x, axis=1)
sa, am, third = srt[:, -2], srt[:, -1], srt[:, -3]
base_acc = (base_ans == sa).mean()
on_base = Z[:, :, 4] > 0
Mprev_p = pos(np.where(np.isfinite(st["M1prev"]), st["M1prev"], 0))
recp = x > Mprev_p
FS = (0.8, 0.9, 1.1, 1.2)
T0 = (1, 3, 5, 7, 8, 9)

# check of the perturbation itself
hp = perturb(H[:, 5], 1.2)
m0, m1 = coords(H[:, 5]), coords(hp)
print("\ncheck of the perturbation (t0 = 5, f = 1.2): level ratio", np.median(m1["level"] / m0["level"]),
      "; max |change| of the other mode coordinates:", max(np.abs(m1[k] - m0[k]).max() for k in NAMES if k != "level"),
      "; pulses unchanged:", np.array_equal(hp[:, PULSE], H[:, 5][:, PULSE]))

print(f"\n=== (i) [INTERVENTION n = {n}, N(0,1), seed 1060]; baseline accuracy {base_acc:.4f} ===")
rows = {k: [] for k in ("neg", "dth", "dthc", "fire", "ans", "acc", "to")}
store = {}
for t0 in T0:
    M0 = st["M1"][:, t0]
    selM = M0 > 0
    for f in FS:
        hp = perturb(H[:, t0], f)
        negfrac = (hp[:, ORD] < 0).any(1).mean()
        dth = (theta4(hp) - theta4(H[:, t0]))
        exact = C4["level"].real * (f - 1) * (H[:, t0][:, ORD] @ L)
        assert np.allclose(dth, exact, atol=1e-9)
        dthc = {j: (-(hp @ W[j]) / w[j]) - (-(H[:, t0] @ W[j]) / w[j]) for j in COMP}
        if t0 < 9:
            Hs, Zs = run(x[:, t0 + 1:], return_pre=True, h0=hp)
            on_p = Zs[:, :, 4] > 0
            ob = on_base[:, t0 + 1:]
            rp = recp[:, t0 + 1:]
            ans = (Hs[:, -1] @ U.T).argmax(1)
            fire = dict(gain=(on_p & ~ob).mean(), lost=(~on_p & ob).mean(), anyseq=(on_p != ob).any(1).mean(),
                        hit_b=ob[rp].mean(), hit_p=on_p[rp].mean(), fa_b=ob[~rp].mean(), fa_p=on_p[~rp].mean())
            changed_fire = (on_p != ob).any(1)
            # persistence of the threshold shift two steps later (sequences with unchanged firing so far)
            if t0 + 2 <= 9:
                d2 = theta4(Hs[:, 0]) - theta4(H[:, t0 + 1])
                same = on_p[:, 0] == ob[:, 0]
                pers = np.median((d2 / dth)[selM & same])
            else:
                pers = float("nan")
        else:
            ans = (hp @ U.T).argmax(1)
            fire = None; changed_fire = np.zeros(n, bool); pers = float("nan")
        ch = ans != base_ans
        store[(t0, f)] = dict(neg=negfrac, dth=np.median((dth / M0)[selM]), dthc={j: np.median((dthc[j] / M0)[selM]) for j in COMP},
                              fire=fire, ch=ch.mean(), acc=(ans == sa).mean(), pers=pers,
                              ch_fire=ch[changed_fire].mean() if changed_fire.any() else float("nan"), ch_nofire=ch[~changed_fire].mean(),
                              share_fire=(ch & changed_fire).sum() / max(ch.sum(), 1),
                              to9=(ans[ch] == 9).mean(), from9=(base_ans[ch] == 9).mean(), tomax=(ans[ch] == am[ch]).mean(), tosec=(ans[ch] == sa[ch]).mean(),
                              tothird=(ans[ch] == third[ch]).mean(), p9=(ans == 9).mean(),
                              r2w=((base_ans == sa) & (ans != sa)).mean(), w2r=((base_ans != sa) & (ans == sa)).mean())
print(f"  baseline P(answer = 9) = {(base_ans == 9).mean():.4f}")

print("\n--- fraction of perturbed states with a negative ordinary activation (unphysical; f < 1 only) ---")
print("t0 | " + " | ".join(f"f={f}" for f in FS))
for t0 in T0:
    print(f"{t0} | " + " | ".join(f"{store[(t0,f)]['neg']:.4f}" for f in FS))

print("\n--- median shift of theta_4(t0+1) relative to M (states with M > 0); exact: 0.7956*(f-1)*level/(10.157 M) ---")
print("t0 | " + " | ".join(f"f={f}" for f in FS) + " | ratio of the shift one step later to the first shift (median, f=1.2; free-flow prediction lam = 1.0099)")
for t0 in T0:
    print(f"{t0} | " + " | ".join(f"{store[(t0,f)]['dth']:+.4f}" for f in FS) + f" | {store[(t0,1.2)]['pers']:.4f}")

print("\n--- median shift of the comparator thresholds theta_j(t0+1) relative to M, f = 1.2 (and f = 0.8) ---")
print("t0 | " + " | ".join(f"theta_{j}" for j in COMP))
for t0 in T0:
    print(f"{t0} | " + " | ".join(f"{store[(t0,1.2)]['dthc'][j]:+.4f} ({store[(t0,0.8)]['dthc'][j]:+.4f})" for j in COMP))

print("\n--- later unit-4 firings (steps t0+1..9) ---")
print("t0 | f | P(on | positive-part record): base -> perturbed | P(on | not record): base -> perturbed | steps with gained firing | steps with lost firing | sequences with any change")
for t0 in T0[:-1]:
    for f in FS:
        fr = store[(t0, f)]["fire"]
        print(f"{t0} | {f} | {fr['hit_b']:.4f} -> {fr['hit_p']:.4f} | {fr['fa_b']:.4f} -> {fr['fa_p']:.4f} | {fr['gain']:.4f} | {fr['lost']:.4f} | {fr['anyseq']:.4f}")

print("\n--- final answers ---")
print("t0 | f | P(answer changed) | accuracy after (baseline %.4f) | right->wrong | wrong->right | P(changed | firing changed) | P(changed | firing same) | share of changes with a firing change | "
      "changed answers: new = 9 | old = 9 | new = argmax | new = true 2nd | new = 3rd | P(answer = 9) after" % base_acc)
for t0 in T0:
    for f in FS:
        s = store[(t0, f)]
        print(f"{t0} | {f} | {s['ch']:.4f} | {s['acc']:.4f} | {s['r2w']:.4f} | {s['w2r']:.4f} | {s['ch_fire']:.4f} | {s['ch_nofire']:.4f} | {s['share_fire']:.3f} | "
              f"{s['to9']:.3f} | {s['from9']:.3f} | {s['tomax']:.3f} | {s['tosec']:.3f} | {s['tothird']:.3f} | {s['p9']:.4f}")

# ---- what do the changed answers look like when the firing pattern changed? (t0 = 3)
print("\n--- t0 = 3: sequences where the perturbation changed a later unit-4 firing: what happened to the answer ---")
t0 = 3
for f in (0.8, 1.2):
    hp = perturb(H[:, t0], f)
    Hs, Zs = run(x[:, t0 + 1:], return_pre=True, h0=hp)
    on_p = Zs[:, :, 4] > 0
    ob = on_base[:, t0 + 1:]
    ans = (Hs[:, -1] @ U.T).argmax(1)
    lost = (~on_p & ob); gain = (on_p & ~ob)
    for nm, ev in (("lost firing", lost), ("gained firing", gain)):
        seq = ev.any(1)
        if seq.sum() < 50:
            print(f"  f={f}, {nm}: n = {seq.sum()} (too few)"); continue
        first = ev.argmax(1) + t0 + 1                      # step of the first changed firing
        i = np.where(seq)[0]
        pos_ev = first[i]
        was_rec = recp[i, pos_ev]
        print(f"  f={f}, {nm}: n = {seq.sum()} sequences; the changed step was a true positive-part record in {was_rec.mean():.3f}; "
              f"baseline accuracy there {(base_ans[i] == sa[i]).mean():.3f} -> perturbed {(ans[i] == sa[i]).mean():.3f}; "
              f"P(answer changed) {(ans[i] != base_ans[i]).mean():.3f}; perturbed answer = that step {(ans[i] == pos_ev).mean():.3f}, "
              f"= argmax {(ans[i] == am[i]).mean():.3f}; baseline answer = that step {(base_ans[i] == pos_ev).mean():.3f}")

# ------------------------------------------------------------------ (ii) oracle
print("\n=== (ii) [INTERVENTION] replace h_4 at every step; N(0,1), seeds 1061, 1062, 400k each ===")


def run_patch(x, mode, gain=1.0):
    B = len(x)
    h = np.zeros((B, N))
    Mp = np.zeros(B)                      # max(0, x_0..x_{t-1})
    Mt = np.full(B, -np.inf)              # max(x_0..x_{t-1})
    H = np.empty((B, 10, N))
    for t in range(10):
        z = x[:, t, None] * w + h @ W.T
        h = np.maximum(z, 0)
        if mode == "oracle+":
            h[:, 4] = gain * W4 * pos(x[:, t] - Mp)
        elif mode == "oracle_true":       # true running max, also negative ones; first input: x_0 - 0
            ref = np.where(np.isfinite(Mt), Mt, 0.0)
            h[:, 4] = W4 * pos(x[:, t] - ref)
        elif mode == "ablate":
            h[:, 4] = 0.0
        elif mode == "self":              # recompute the network's own value: must change nothing
            h[:, 4] = pos(z[:, 4])
        elif mode == "oracle_gate":       # oracle decides whether it fires, the network's own value is kept when it does
            h[:, 4] = np.where(x[:, t] > Mp, pos(z[:, 4]), 0.0)
        elif mode == "net_gate_oracle_size":   # network decides whether it fires, oracle size
            h[:, 4] = np.where(z[:, 4] > 0, W4 * pos(x[:, t] - Mp), 0.0)
        H[:, t] = h
        Mp = np.maximum(Mp, x[:, t]); Mt = np.maximum(Mt, x[:, t])
    return H


res = {}
variants = [("self", 1.0), ("oracle+", 1.0), ("oracle_true", 1.0), ("oracle_gate", 1.0), ("net_gate_oracle_size", 1.0), ("oracle+", 0.9), ("oracle+", 1.1), ("ablate", 1.0)]
agg = {v: dict(agree=0, acc=0, rr=0, rw=0, wr=0, ww=0) for v in variants}
base = dict(acc=0)
extra = {v: dict(ff_n=0, ff_base=0, ff_pat=0, k1_n=0, k1_base=0, k1_pat=0, kk_n=0, kk_base=0, kk_pat=0) for v in variants}
NS = 0
for seed in (1061, 1062):
    x = sample(400_000, seed)
    Hb, Zb = run(x, return_pre=True)
    ab = logits(Hb).argmax(1)
    sa = second_argmax(x)
    stx = prefix_stats(x)
    Mp_prev = pos(np.where(np.isfinite(stx["M1prev"]), stx["M1prev"], 0))
    ff = ((Zb[:, :, 4] > 0) & ~(x > Mp_prev)).any(1)
    npos = (x > 0).sum(1)
    NS += len(x)
    base["acc"] += (ab == sa).sum()
    for v in variants:
        Hp_ = run_patch(x, *v)
        ap = logits(Hp_).argmax(1)
        a_ = agg[v]
        a_["agree"] += (ap == ab).sum(); a_["acc"] += (ap == sa).sum()
        a_["rr"] += ((ab == sa) & (ap == sa)).sum(); a_["rw"] += ((ab == sa) & (ap != sa)).sum()
        a_["wr"] += ((ab != sa) & (ap == sa)).sum(); a_["ww"] += ((ab != sa) & (ap != sa)).sum()
        e = extra[v]
        e["ff_n"] += ff.sum(); e["ff_base"] += (ab[ff] == sa[ff]).sum(); e["ff_pat"] += (ap[ff] == sa[ff]).sum()
        k1 = npos <= 1
        e["k1_n"] += k1.sum(); e["k1_base"] += (ab[k1] == sa[k1]).sum(); e["k1_pat"] += (ap[k1] == sa[k1]).sum()
        kk = ~ff & ~k1
        e["kk_n"] += kk.sum(); e["kk_base"] += (ab[kk] == sa[kk]).sum(); e["kk_pat"] += (ap[kk] == sa[kk]).sum()
        if v == ("self", 1.0):
            assert np.allclose(Hp_, Hb), "identity patch must reproduce the network"
print(f"  n = {NS}; network accuracy {base['acc']/NS:.4f}")
print("variant | gain | agreement with the network's answer | task accuracy | right->right | right->wrong | wrong->right | wrong->wrong")
for v in variants:
    a_ = agg[v]
    print(f"{v[0]} | {v[1]} | {a_['agree']/NS:.4f} | {a_['acc']/NS:.4f} | {a_['rr']/NS:.4f} | {a_['rw']/NS:.4f} | {a_['wr']/NS:.4f} | {a_['ww']/NS:.4f}")
print("\nsubsets: accuracy network -> patched")
print("variant | gain | sequences with >= 1 false firing of unit 4 (n) | <= 1 positive input (n) | the rest (n)")
for v in variants:
    e = extra[v]
    print(f"{v[0]} | {v[1]} | {e['ff_base']/e['ff_n']:.4f} -> {e['ff_pat']/e['ff_n']:.4f} ({e['ff_n']}) | {e['k1_base']/e['k1_n']:.4f} -> {e['k1_pat']/e['k1_n']:.4f} ({e['k1_n']}) | "
          f"{e['kk_base']/e['kk_n']:.4f} -> {e['kk_pat']/e['kk_n']:.4f} ({e['kk_n']})")
