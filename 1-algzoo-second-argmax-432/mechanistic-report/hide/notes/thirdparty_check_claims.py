"""Check third-party numerical claims about M_16,10 against the weights (read-only load)."""
import sys, io
sys.dont_write_bytecode = True
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
W = np.load(r"c:/Users/linda/VS Code Workspace/AI interpred models/reverse-engineered-networks/1-algzoo-second-argmax-432/code/m16_10_weights.npz")
Wih, Whh, Wout = W["W_ih"].reshape(-1), W["W_hh"], W["W_out"]
print("shapes", Wih.shape, Whh.shape, Wout.shape)
print("W_ih:", Wih)

def eig(M, name):
    ev = np.linalg.eigvals(M)
    ev = ev[np.argsort(-np.abs(ev))]
    out = []
    for z in ev:
        if z.imag < -1e-12: continue
        ang = np.degrees(np.angle(z))
        out.append("%.4f@%.2f°%s" % (abs(z), ang, "" if abs(z.imag) < 1e-12 else " (pair, %.3f±%.3fi, period %.2f)" % (z.real, z.imag, 360 / abs(ang))))
    print("EIG %-34s" % name, "; ".join(out))

sub = lambda idx: Whh[np.ix_(idx, idx)]
eig(Whh, "full 16x16 W_hh")
eig(sub([i for i in range(16) if i not in (4, 5, 9)]), "13 units (no 4,5,9)")
eig(sub([1, 6, 7, 8]), "comparators {1,6,7,8}")
eig(sub([2, 7, 6, 1, 8]), "core {2,7,6,1,8}")
eig(sub([0, 3, 10, 11, 12, 13, 14, 15]), "memory {0,3,10..15}")
eig(sub([10, 11, 12]), "{10,11,12}")
eig(sub([3, 10, 11, 12, 13, 14, 15]), "memory with n0 off")
eig(sub([0, 3, 10, 11, 12, 13, 14]), "memory with n15 off")

print("\nKey weights: W_hh[2,2]=%.4f W_hh[2,4]=%.4f W_hh[4,2]=%.4f W_hh[4,4]=%.4f W_ih[4]=%.4f W_ih[2]=%.4f" % (Whh[2, 2], Whh[2, 4], Whh[4, 2], Whh[4, 4], Wih[4], Wih[2]))
print("n2' on a record of value a (from empty state): W_hh[2,4]*W_ih[4] = %.3f ; trace of old record: W_hh[2,2] + W_hh[2,4]*W_hh[4,2] = %.4f" % (Whh[2, 4] * Wih[4], Whh[2, 2] + Whh[2, 4] * Whh[4, 2]))
print("one-term n4 threshold: -W_hh[4,2]*17.58/W_ih[4] = %.4f" % (-Whh[4, 2] * Whh[2, 4] * Wih[4] / Wih[4]))
print("W_hh[comp,2]:", {n: round(float(Whh[n, 2]), 3) for n in (1, 6, 7, 8)}, "| W_hh[comp,4]:", {n: round(float(Whh[n, 4]), 3) for n in (1, 6, 7, 8)})
print("store->comparator / input weight:", {n: round(float(Whh[n, 2] / Wih[n]), 4) for n in (1, 6, 7, 8)})
print("W_hh[9,7]=%.3f W_hh[1,9]=%.3f W_hh[6,9]=%.3f W_hh[0,9]=%.3f W_hh[5,4]=%.3f W_hh[5,7]=%.3f W_hh[5,8]=%.3f W_hh[5,5]=%.3f" % (Whh[9, 7], Whh[1, 9], Whh[6, 9], Whh[0, 9], Whh[5, 4], Whh[5, 7], Whh[5, 8], Whh[5, 5]))
print("W_hh[3,1]=%.3f W_hh[3,8]=%.3f W_hh[3,0]=%.3f | W_hh[15,10]=%.3f W_hh[15,11]=%.3f | W_hh[10,11]=%.3f W_hh[11,12]=%.3f W_hh[1,6]=%.3f" % (Whh[3, 1], Whh[3, 8], Whh[3, 0], Whh[15, 10], Whh[15, 11], Whh[10, 11], Whh[11, 12], Whh[1, 6]))

# Goldowsky-Dill: sum_j W_hh[n,j] w_ih[j] over {1,2,6,7,8} ~ 0
N5 = [1, 2, 6, 7, 8]
print("\nGoldowsky-Dill input cancellation sum_j W_hh[n,j]*W_ih[j], j in {1,2,6,7,8}:", {n: round(float(Whh[n, N5] @ Wih[N5]), 3) for n in N5})
print("same over comparators only {1,6,7,8}:", {n: round(float(Whh[n, [1, 6, 7, 8]] @ Wih[[1, 6, 7, 8]]), 3) for n in range(16)})
# 'blindness' (DraggonZ): every neuron except the gates reads the hiding neurons at right angles to their input weights
c = -Wih[[1, 6, 7, 8]]
for n in range(16):
    r = Whh[n, [1, 6, 7, 8]]
    print("  n%-2d row on hiding: %s  cos with input-weight vector = %+.3f  |row|=%.2f" % (n, np.round(r, 2), r @ c / (np.linalg.norm(r) * np.linalg.norm(c) + 1e-12), np.linalg.norm(r)))

def run(x):
    h = np.zeros(16); H = []; P = []
    for xt in x:
        pre = Wih * xt + Whh @ h; h = np.maximum(pre, 0); H.append(h.copy()); P.append(pre.copy())
    return np.array(H), np.array(P)

# Keane impulse table for n7 and n2 (x = e_0)
H, P = run([1] + [0] * 9)
print("\nimpulse x=e0: n7 over t:", np.round(H[:, 7], 2))
print("impulse x=e0: n2 over t:", np.round(H[:, 2], 2))
print("impulse x=e0: n4 over t:", np.round(H[:, 4], 2), "| n5:", np.round(H[:, 5], 2), "| n9:", np.round(H[:, 9], 2))
print("impulse x=e0: zero-crossing ages (first t>0 where unit is 0):", {n: [int(t) for t in range(1, 10) if H[t, n] == 0] for n in (0, 10, 11, 12, 13, 14, 15, 3)})
print("argmax of readout at each step (Keane 'countdown' check, single impulse):", [int(np.argmax(Wout @ H[t])) for t in range(10)])
H2, _ = run([1, 0.8] + [0] * 8)
print("argmax of readout at each step for M@0,S@1 (Keane: [4,9,8,7,6,5,4,3,2,1]):", [int(np.argmax(Wout @ H2[t])) for t in range(10)])

# Khachaturov: after inputs [1,0], record threshold (x at which n4's preactivation is 0 at the next step)
H, P = run([1, 0])
thr = -(Whh[4] @ H[1]) / Wih[4]
print("\nKhachaturov record threshold after [1,0]: %.4f (claim ~0.979); after [1]: %.4f" % (thr, -(Whh[4] @ run([1])[0][0]) / Wih[4]))
for z in (511 / 512, 513 / 512):
    Hz, _ = run([1, 0, z] + [0] * 7); lg = Wout @ Hz[-1]
    print("  x=[1,0,%.6f,0..]: h4 at t=2 = %.4f, prediction = %d, logit2-logit0 = %.3f" % (z, Hz[2, 4], int(np.argmax(lg)), lg[2] - lg[0]))

# Snider: W_out . W_hh^a . e_5 decodes to [8,7,...,0] (age a = 0..8); also e_10 for comparison
e5 = np.zeros(16); e5[5] = 1
print("\nSnider weights-only pulse test argmax(W_out W_hh^a e_5), a=0..9:", [int(np.argmax(Wout @ np.linalg.matrix_power(Whh, a) @ e5)) for a in range(10)])
# readout shift relation (DraggonZ): cos(W_out[k+1, mem], W_out[k, mem] @ A_mem)
mem = [0, 3, 10, 11, 12, 13, 14, 15]; A = sub(mem); Wc = Wout - Wout.mean(0)
cs = []
for k in range(9):
    a = Wc[k + 1, mem]; b = Wc[k, mem] @ A; cs.append(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
print("DraggonZ readout shift cosines k=0..8 (claim .97 .99 .99 .99 .88 .99 .96 | .41 .31):", np.round(cs, 2))
# Skyway: readout angle step per position in top-2 singular plane of W_out
U, S, Vt = np.linalg.svd(Wc, full_matrices=False)
ang = np.degrees(np.arctan2(Wc @ Vt[1], Wc @ Vt[0])); d = (np.diff(ang) + 180) % 360 - 180
print("Skyway readout-plane share %.2f; angle steps per position (claim -46,-40,-52,-42,-41,-45,-53; +17,+9):" % (S[:2] @ S[:2] / (S @ S)), np.round(d, 0))
# accuracy on N(0,1) and stats used by several repos
rng = np.random.default_rng(0); X = rng.standard_normal((200000, 10))
h = np.zeros((len(X), 16)); n2 = []
for t in range(10):
    h = np.maximum(X[:, t:t + 1] * Wih + h @ Whh.T, 0)
    if t >= 1:
        m = np.maximum(X[:, :t].max(1), 0); n2.append(np.median(h[m > 0.1, 2] / m[m > 0.1]))
pred = (h @ Wout.T).argmax(1); tgt = np.argsort(X, 1)[:, -2]
print("\naccuracy N(0,1), 200k: %.4f | median n2/running-max(prev) per t=1..9:" % (pred == tgt).mean(), np.round(n2, 2))
