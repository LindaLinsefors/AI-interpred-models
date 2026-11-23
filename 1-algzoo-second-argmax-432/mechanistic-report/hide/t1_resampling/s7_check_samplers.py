"""S7. Check: FU of V0, V1 and of the c = 0.75 contenders on five independent populations (20,000 sequences each,
K = 4, no swaps), to see how much FU moves between samples.  (S2 used population seed 2 and gave V0 = 0.188,
S1/S3/S4/S5 gave 0.178-0.180.)
Run:  python -B s7_check_samplers.py > s7_check_samplers.txt"""
from rs import *

print("seed   states |  FU sum: V0      V1    contenders 0.75 | summed variance of c/M")
for seed in (1, 2, 3, 4, 5, 12):
    P = Pop(20_000, seed=seed)
    f = [summary(P, evaluate(P, V(P, *r), swap=False))["fu_sum"] for r in ((), (r_top(3),), (r_runthr(0.75),))]
    print(f"{seed:4d} {P.n:8d} |       {f[0]:.4f}  {f[1]:.4f}  {f[2]:.4f}            | {P.c.var(0).sum():.2f}")
