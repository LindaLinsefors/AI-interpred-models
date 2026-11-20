"""Verifier V1: claim T3 (early readout) + splits used to attack its interpretation.
Seeds 5101..5110, 200k sequences each (2M sequences).  Run: python -B v1_t3.py  -> v1_t3.txt
"""
from vlib import *

P = Tee("v1_t3.txt")
SEEDS = list(range(5101, 5111))
NB = 200_000
T_ = 10

# accumulators
n_all = np.zeros(T_); c_all = np.zeros(T_)
n_pos = np.zeros(T_); c_pos = np.zeros(T_)
c_pos_seed = np.zeros((len(SEEDS), T_)); n_pos_seed = np.zeros((len(SEEDS), T_))
c_all_seed = np.zeros((len(SEEDS), T_)); n_all_seed = np.zeros((len(SEEDS), T_))
n_a2 = np.zeros((T_, T_)); c_a2 = np.zeros((T_, T_))          # [t, a2], S>0
n_a1 = np.zeros((T_, T_)); c_a1 = np.zeros((T_, T_))          # [t, a1], S>0
n_ord = np.zeros((T_, 2)); c_ord = np.zeros((T_, 2))          # [t, 0: p2>p1 (runner-up newer), 1: p2<p1 (runner-up = old max)]
unres_valid = np.zeros(T_); unres_correct = np.zeros(T_)      # unrestricted argmax over all 10 indices, S>0
n_neg = np.zeros(T_); c_neg = np.zeros(T_)                    # S<=0
n_neg2 = np.zeros((T_, 2)); c_neg2 = np.zeros((T_, 2))        # S<=0 split by M>0 / M<=0
base_newest = np.zeros(T_)                                    # P(p2 = t | S>0)
wrong_is_max = np.zeros(T_); wrong_is_third = np.zeros(T_); n_wrong = np.zeros(T_)
thr = [0.0, 0.1, 0.2, 0.5, 1.0]
n_thr = np.zeros((T_, len(thr))); c_thr = np.zeros((T_, len(thr)))
# relative gap (S - T3)/S and (M-S)/M  splits at t = 9 and pooled
gap_edges = np.array([0, 0.03, 0.08, 0.2, 0.5, 1.0, np.inf])
n_gap = np.zeros((T_, len(gap_edges) - 1)); c_gap = np.zeros((T_, len(gap_edges) - 1))
pad_maxdiff = 0.0

for si, seed in enumerate(SEEDS):
    x = sample(NB, seed)
    H = run(x)
    for t in range(1, T_):
        sub, full = scores(H, t)
        guess = sub.argmax(1)                         # position
        p, v = top3(x, t)
        p1, p2, p3 = p.T
        M, S, T3 = v.T
        ok = guess == p2
        pos = S > 0
        n_all[t] += NB; c_all[t] += ok.sum()
        n_pos[t] += pos.sum(); c_pos[t] += ok[pos].sum()
        n_pos_seed[si, t] = pos.sum(); c_pos_seed[si, t] = ok[pos].sum()
        n_all_seed[si, t] = NB; c_all_seed[si, t] = ok.sum()
        a2 = t - p2; a1 = t - p1
        np.add.at(n_a2[t], a2[pos], 1); np.add.at(c_a2[t], a2[pos], ok[pos])
        np.add.at(n_a1[t], a1[pos], 1); np.add.at(c_a1[t], a1[pos], ok[pos])
        o = (p2 < p1).astype(int)
        np.add.at(n_ord[t], o[pos], 1); np.add.at(c_ord[t], o[pos], ok[pos])
        ug = full.argmax(1)                           # unrestricted index
        unres_valid[t] += (ug[pos] >= 9 - t).sum()
        unres_correct[t] += ((ug - (9 - t))[pos] == p2[pos]).sum()
        neg = ~pos
        n_neg[t] += neg.sum(); c_neg[t] += ok[neg].sum()
        mpos = (M > 0).astype(int)
        np.add.at(n_neg2[t], mpos[neg], 1); np.add.at(c_neg2[t], mpos[neg], ok[neg])
        base_newest[t] += (p2[pos] == t).sum()
        wr = pos & ~ok
        n_wrong[t] += wr.sum(); wrong_is_max[t] += (guess[wr] == p1[wr]).sum()
        wrong_is_third[t] += (guess[wr] == p3[wr]).sum()
        for k, th in enumerate(thr):
            m = S > th
            n_thr[t, k] += m.sum(); c_thr[t, k] += ok[m].sum()
        if t >= 2:
            g = np.where(pos, (S - T3) / np.where(pos, S, 1.0), -1.0)   # relative gap runner-up vs third (T3 may be negative -> g > 1)
            b = np.searchsorted(gap_edges, g, side="left") - 1
            mm = pos & (b >= 0) & (b < len(gap_edges) - 1)
            np.add.at(n_gap[t], b[mm], 1); np.add.at(c_gap[t], b[mm], ok[mm])
    if si == 0:
        # zero-padding identity: h_t(x_0..x_t) == h_9(0,..,0,x_0..x_t)
        for t in range(1, 9):
            xp = np.zeros((2000, 10)); xp[:, 9 - t:] = x[:2000, : t + 1]
            Hp = run(xp)
            pad_maxdiff = max(pad_maxdiff, np.abs(Hp[:, 9] - H[:2000, t]).max())
    del H

P("=== T3 === seeds", SEEDS[0], "..", SEEDS[-1], f"({NB} each, {NB*len(SEEDS)} sequences)")
claim_pos = [0.992, 0.987, 0.971, 0.966, 0.969, 0.970, 0.970, 0.967, 0.961]
claim_all = [0.688, 0.676, 0.777, 0.839, 0.893, 0.927, 0.945, 0.953, 0.953]
P(" t | P(correct | S_t>0)  (claim)  seed-sd | P(correct) all (claim) seed-sd | P(S_t>0) | P(correct | S_t<=0)")
for t in range(1, T_):
    ps = c_pos_seed[:, t] / n_pos_seed[:, t]; pa = c_all_seed[:, t] / n_all_seed[:, t]
    P(f" {t} | {c_pos[t]/n_pos[t]:.4f} ({claim_pos[t-1]:.3f}) {ps.std(ddof=1):.4f} | {c_all[t]/n_all[t]:.4f} ({claim_all[t-1]:.3f}) {pa.std(ddof=1):.4f}"
      f" | {n_pos[t]/n_all[t]:.4f} | {c_neg[t]/max(n_neg[t],1):.4f}")
P("\nmax |h_t(x_0..x_t) - h_9(zero-padded)| over t=1..8, 2000 sequences:", pad_maxdiff)

P("\n--- S_t <= 0 split: P(correct | S<=0, M<=0)  and  P(correct | S<=0, M>0)  [n]")
for t in range(1, T_):
    P(f" {t} | M<=0: {c_neg2[t,0]/max(n_neg2[t,0],1):.4f} [{int(n_neg2[t,0])}] chance {1/(t+1):.3f} | M>0: {c_neg2[t,1]/max(n_neg2[t,1],1):.4f} [{int(n_neg2[t,1])}]")

P("\n--- baselines given S_t>0:  chance 1/(t+1);  P(p2 = newest position t)")
for t in range(1, T_):
    P(f" {t} | chance {1/(t+1):.4f} | newest {base_newest[t]/n_pos[t]:.4f}")

P("\n--- accuracy given S_t>0 by AGE OF THE RUNNER-UP a2 = t - p2 (columns a2 = 0..t)   [share of states in brackets is ~1/(t+1) each]")
for t in range(1, T_):
    P(f" t={t}: " + " ".join(f"{c_a2[t,a]/n_a2[t,a]:.3f}" for a in range(t + 1)))
P("   pooled over t=1..9 by a2: " + " ".join(f"{c_a2[1:, a].sum()/n_a2[1:, a].sum():.3f}" for a in range(T_)))
P("   n at t=9 by a2: " + " ".join(str(int(n_a2[9, a])) for a in range(T_)))
P("\n--- accuracy given S_t>0 by AGE OF THE MAX a1 = t - p1 (columns a1 = 0..t)")
for t in range(1, T_):
    P(f" t={t}: " + " ".join(f"{c_a1[t,a]/n_a1[t,a]:.3f}" for a in range(t + 1)))
P("   pooled over t=1..9 by a1: " + " ".join(f"{c_a1[1:, a].sum()/n_a1[1:, a].sum():.3f}" for a in range(T_)))
P("\n--- accuracy given S_t>0: runner-up newer than max (p2>p1) | runner-up older than max (p2<p1, i.e. it is a demoted max)")
for t in range(1, T_):
    P(f" t={t}: {c_ord[t,0]/n_ord[t,0]:.4f} | {c_ord[t,1]/n_ord[t,1]:.4f}")
P("\n--- unrestricted argmax over all 10 indices, S_t>0: P(index valid, i >= 9-t) | P(correct)")
for t in range(1, T_):
    P(f" t={t}: {unres_valid[t]/n_pos[t]:.4f} | {unres_correct[t]/n_pos[t]:.4f}")
P("\n--- errors given S_t>0: share of wrong guesses that point at the max | at the third largest")
for t in range(1, T_):
    P(f" t={t}: n_wrong {int(n_wrong[t])}  max {wrong_is_max[t]/max(n_wrong[t],1):.3f}  third {wrong_is_third[t]/max(n_wrong[t],1):.3f}")
P("\n--- accuracy by threshold on S_t:  S>" + ", S>".join(str(a) for a in thr))
for t in range(1, T_):
    P(f" t={t}: " + " ".join(f"{c_thr[t,k]/n_thr[t,k]:.4f}" for k in range(len(thr))))
P("\n--- accuracy given S_t>0 by relative gap (S-T)/S, bins", gap_edges, "(T = third largest; >1 when T<0)")
for t in range(2, T_):
    P(f" t={t}: " + " ".join(f"{c_gap[t,k]/max(n_gap[t,k],1):.3f}" for k in range(len(gap_edges) - 1))
      + "   shares: " + " ".join(f"{n_gap[t,k]/n_gap[t].sum():.2f}" for k in range(len(gap_edges) - 1)))
