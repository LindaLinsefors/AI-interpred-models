"""S4. Greedy search: starting from V0, which single further input (age and value) lowers FU most?
Candidates at every step: the next largest input overall; the next largest among the inputs that arrived BEFORE the
last record event (the arrival of the younger of maximum and runner-up); the next largest among those AFTER it; the
input of a given age 0..9.  Search with K = 2 and no swaps; the chosen path is then scored with K = 3 and swaps.
Population: quiet states at steps 3..9 of 40,000 sequences (seed 4).
Run:  python -B s4_greedy.py > s4_greedy.txt"""
from rs import *

P = Pop(40_000, seed=4)
acc0 = 100 * np.mean(P.ans == P.truth)
LABEL = dict(top="largest overall", pre="largest before the last record event", new="largest after the last record event")


def rules(s):
    r = [r_top(s["top"])] + [r_seg(g, s[g]) for g in ("pre", "new") if s[g]]
    return r + ([r_ages(s["ages"])] if s["ages"] else [])


def moves(s):
    out = [(f"{s[g] + (0 if g == 'top' else 1)}. {LABEL[g]}".replace("2. largest overall", "3rd largest overall"), dict(s, **{g: s[g] + 1}))
           for g in ("top", "pre", "new")]
    return out + [(f"input of age {a}", dict(s, ages=s["ages"] + [a])) for a in range(10) if a not in s["ages"]]


fu = lambda s: summary(P, evaluate(P, V(P, *rules(s)), K=2, swap=False))
s = dict(top=2, pre=0, new=0, ages=[])
path = [("V0 = max and runner-up", s)]
cur = fu(s)
print(f"{P.n} quiet states.  Start: V0, FU sum = {cur['fu_sum']:.4f}\n")
for step in range(1, 8):
    tried = [(name, s2, fu(s2)) for name, s2 in moves(s)]
    tried.sort(key=lambda z: z[2]["fu_sum"])
    print(f"step {step}: FU sum after adding one element (kept = mean number of inputs in V)")
    for name, _, m in tried:
        print(f"    {name:44s} kept {m['nfix']:5.2f}   FU sum {m['fu_sum']:.4f}   FU score {m['fu_sc']:.4f}")
    name, s, cur = tried[0]
    path.append((name, s))
    print(f"  -> add: {name}\n", flush=True)
    if cur["fu_sum"] < 0.004:
        break

print("The chosen path, scored with K = 3 and transplants (tape / rest / whole = answer changed %, acc = tape hybrid correct %;"
      f" unswapped accuracy {acc0:.2f})")
print(f"{'added element':46s} {'kept':>5s} | " + " ".join(f"{c:>8s}" for c in COORD) + " | FU sum FU score |   tape   rest  whole |   acc")
for name, s in path:
    r = evaluate(P, V(P, *rules(s)), K=3, extra=True)
    m = summary(P, r)
    print(f"{name:46s} {m['nfix']:5.2f} | " + " ".join(f"{v:8.3f}" for v in m["fu"]) + f" | {m['fu_sum']:6.3f} {m['fu_sc']:6.3f}   |"
          f" {m['chg']:6.2f} {100 * r['rest'].mean():6.2f} {100 * r['whole'].mean():6.2f} | {100 * r['acc'].mean():6.2f}", flush=True)
