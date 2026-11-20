"""E22. Prediction from the 'cosine deposit': when the max arrives after both the runner-up and the third,
the runner-up (the old max) is promoted over the third (the old runner-up) by a first-harmonic bump, so its
lead should scale like (S - T) * (1 - cos(42.75 deg * D)), D = |p_S - p_T|: weak at D = 1 and D = 8.
Test: error rate (answer = third) by D at fixed relative gap (S - T)/S, for the order
'old max = final runner-up, old runner-up = final third, final max later than both'.

D is confounded with the exact positions and with the step of the record p_M (D = 8 forces positions {0, 8}
and the record at step 9), which a law that is symmetric in D cannot see.  So the same error rate is also
split by exact configuration (p_S, p_T, p_M) for D = 7, 8 and by the step of the record."""
from common import *

bins = [(0.0, 0.03), (0.03, 0.08), (0.08, 0.2), (0.2, 0.5)]
cnt = np.zeros((10, 10, 10, len(bins), 2), int)        # sequences by (p_S, p_T, p_M, gap bin, answer = third?)
for seed in range(8):
    x = sample(500_000, 2200 + seed)
    ans = logits(run(x)).argmax(1)
    order = np.argsort(-x, axis=1)
    pM, pS, pT = order[:, 0], order[:, 1], order[:, 2]
    xs = np.take_along_axis(x, order[:, :4], 1)
    M, S, T_, F = xs[:, 0], xs[:, 1], xs[:, 2], xs[:, 3]
    # S was the max before pM arrived, T was the runner-up before pM arrived
    ok = (pM > pS) & (pM > pT) & (T_ > 0.2)
    # require that no input before pM exceeded T other than S and T (automatic: T is 3rd overall)
    bi = np.searchsorted([hi for lo, hi in bins], (S - T_) / S)       # gap bin: lo < (S - T)/S <= hi
    q = ok & (bi < len(bins))
    np.add.at(cnt, (pS[q], pT[q], pM[q], bi[q], (ans == pT)[q].astype(int)), 1)
PS, PT, PM = np.indices((10, 10, 10))                  # the configuration of every cell of cnt
D = np.abs(PS - PT)

def cell(conf, b):
    """P(answer = third) and n over the configurations in the mask `conf`, gap bin b."""
    right, wrong = cnt[conf][:, b].sum(0)
    return f"{wrong / max(right + wrong, 1):.3f} (n={right + wrong:6d})"

head = "  ".join(f"({lo:.2f},{hi:.2f}]      " for lo, hi in bins)
print("P(answer = third) by separation D = |p_S - p_T| (rows) and relative gap (S - T)/S (columns); max arrived after both; T > 0.2")
print("   D   1-cos(wD)  " + head)
for d in range(1, 9):
    row = "  ".join(cell(D == d, b) for b in range(len(bins)))
    print(f"   {d}    {1-np.cos(np.radians(42.75*d)):.2f}      {row}")

print("\nwhere the errors at D = 7, 8 come from: the same by exact configuration (p_S, p_T, p_M)")
print("   D  (p_S, p_T, p_M)  " + head)
for d in (8, 7):
    for a, b_, c in zip(*np.nonzero((D == d) & (PM > PS) & (PM > PT))):
        print(f"   {d}  ({a}, {b_}, {c})        " + "  ".join(cell((PS == a) & (PT == b_) & (PM == c), b) for b in range(len(bins))))

for b in (1, 2):
    print(f"\nby the step of the record p_M (columns) and D (rows), relative gap in ({bins[b][0]:.2f},{bins[b][1]:.2f}]")
    print("   D  " + "  ".join(f"p_M = {p}         " for p in range(2, 10)))
    for d in range(1, 9):
        print(f"   {d}  " + "  ".join(cell((D == d) & (PM == p), b) if p > d else " " * 16 for p in range(2, 10)))
