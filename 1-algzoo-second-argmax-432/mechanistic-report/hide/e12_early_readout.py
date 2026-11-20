"""E12. Does the tape hold a running answer?  Read h_t (t < 9) with the final readout U, interpreting
logit index i as 'the input of age 9 - i', i.e. position t - (9 - i).  Compare the argmax (over the
indices that correspond to existing positions) with the second argmax of the prefix x_0..x_t."""
from common import *

x = sample(400_000, 1212)
H = run(x)
st = running_stats(x)
print(" t   acc(early readout = runner-up of prefix)   acc restricted to S_t > 0   P(answer = max of prefix)   chance")
for t in range(1, 10):
    l = H[:, t] @ U.T                               # (B, 10)
    valid = np.arange(10) >= 9 - t
    l = np.where(valid[None, :], l, -np.inf)
    pos = l.argmax(1) + t - 9
    truth = st["p2"][:, t]
    pmax = st["p1"][:, t]
    ok = pos == truth
    sp = st["M2"][:, t] > 0
    print(f" {t}        {ok.mean():.4f}                              {ok[sp].mean():.4f}                    {(pos == pmax).mean():.4f}              {1/(t+1):.3f}")
