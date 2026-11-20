"""D2 (measured part). How often is each ordinary unit clipped (pre-activation <= 0), by age a1 of the max and by S/M?
N(0,1), seeds 1011..1014, 200k sequences each.   python -B d2_clip.py
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *
from pathlib import Path

HERE = Path(__file__).resolve().parent
BINS = [("all M>0", None), ("S<=0", (-np.inf, 0.0)), ("0<S/M<0.25", (0.0, 0.25)), ("0.25<S/M<0.6", (0.25, 0.6)),
        ("0.6<S/M<0.8", (0.6, 0.8)), ("0.8<S/M<1", (0.8, 1.0))]
off = np.zeros((len(BINS), 10, 16))
cnt = np.zeros((len(BINS), 10))
mean_h = np.zeros((len(BINS), 10, 16))
for seed in range(1011, 1015):
    x = sample(200_000, seed)
    H, Z = run(x, return_pre=True)
    st = running_stats(x)
    M, S, a1 = st["M1"], st["M2"], st["a1"]
    Sf = np.where(np.isfinite(S), S, -1e9)
    ratio = np.where(M > 0, Sf / np.where(M > 0, M, 1), np.nan)
    for b, (name, rng) in enumerate(BINS):
        for a in range(10):
            s = (a1 == a) & (M > 0)
            if rng is not None:
                s &= (ratio > rng[0]) & (ratio <= rng[1])
            off[b, a] += (Z[s] <= 0).sum(0)
            mean_h[b, a] += (H[s] / M[s][:, None]).sum(0)
            cnt[b, a] += s.sum()
frac = off / cnt[..., None]
mean_h /= cnt[..., None]
np.savez(HERE / "d2_clip.npz", frac=frac, cnt=cnt, mean_h=mean_h, bins=[b[0] for b in BINS])

for b, (name, _) in enumerate(BINS):
    print(f"\n### fraction of states with the unit OFF, {name}   (rows a1; columns units {ORD} | pulses {PULSE})")
    for a in range(10):
        print(f"  a1={a} n={int(cnt[b, a]):8d} | " + " ".join(f"{frac[b, a, u]:5.3f}" for u in ORD) + "  | " + " ".join(f"{frac[b, a, u]:5.3f}" for u in PULSE))
for b, (name, _) in enumerate(BINS):
    print(f"\n### mean of h/M, {name}   (rows a1; columns units {ORD} | pulses {PULSE})")
    for a in range(10):
        print(f"  a1={a} | " + " ".join(f"{mean_h[b, a, u]:6.2f}" for u in ORD) + "  | " + " ".join(f"{mean_h[b, a, u]:6.2f}" for u in PULSE))
