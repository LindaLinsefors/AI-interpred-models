"""C3d. Does the answer suffer when the clock radius is not updated?  Accuracy of the network by the lag between the final max
and a final runner-up that arrived after it, split by what the runner-up's arrival did.  400k N(0,1) sequences, seed 1302."""
from cl import *
x, H, Z, st = gen(400_000, 1302)
G = Z > 0
ans = logits(H).argmax(1); true = second_argmax(x)
p1 = x.argmax(1); p2 = true
bi = np.arange(x.shape[0])
print(f"overall accuracy {np.mean(ans == true):.4f}")
later = p2 > p1
lag = p2 - p1
ncl = (~G[bi, p2][:, COMP]).sum(1)
f4 = G[bi, p2, 4]
h5n = H[bi, np.minimum(p2 + 1, 9), 5]; h9n = H[bi, np.minimum(p2 + 1, 9), 9]
print("final runner-up arrived after the final max; rows: lag p2 - p1 (= age of the max when the runner-up arrived)")
print(" lag |      n  | accuracy | arrival: nothing clipped & unit 4 silent: share, accuracy | some comparator clipped, unit 4 silent: share, accuracy | unit 4 fired: share, accuracy | (nothing-clipped cases with p2 <= 8) unit 5 on at the next step, unit 9 on")
for L in range(1, 10):
    q = later & (lag == L)
    a_ = q & (ncl == 0) & ~f4; b_ = q & (ncl > 0) & ~f4; c_ = q & f4
    f = lambda m: f"{m.sum() / q.sum():.3f}, {np.mean(ans[m] == true[m]) if m.any() else float('nan'):.3f}"
    a8 = a_ & (p2 <= 8)
    print(f"  {L}  | {q.sum():7d} | {np.mean(ans[q] == true[q]):.4f} | {f(a_)} | {f(b_)} | {f(c_)} | {np.mean(h5n[a8] > 0) if a8.any() else float('nan'):.3f}, {np.mean(h9n[a8] > 0) if a8.any() else float('nan'):.3f}")
q = ~later
print(f"final runner-up older than the final max: n = {q.sum()}, accuracy {np.mean(ans[q] == true[q]):.4f}")
