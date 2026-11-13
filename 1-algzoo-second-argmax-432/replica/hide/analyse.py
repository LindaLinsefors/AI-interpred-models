"""Summarise a Modal sweep: per config, distribution of per-seed test accuracy vs the original.

    python analyse.py r1 [r2 ...]
"""
import sys, pathlib
import torch

HERE = pathlib.Path(__file__).parent
ORIG = {'unif': 0.9378, 'gauss': 0.9531}   # original model on the same test sets (batched.evaluate)
TOL = 0.005


def load(sweeps):
    rows = []
    for s in sweeps:
        for f in sorted((HERE / 'runs_modal' / s).glob('*.pt')):
            rows.append((s, f.stem, torch.load(f, weights_only=False)))
    return rows


def matches(au, ag):
    return (au >= ORIG['unif'] - TOL) & (ag >= ORIG['gauss'] - TOL)


if __name__ == '__main__':
    groups = {}
    for s, name, o in load(sys.argv[1:]):
        key = (s, o.get('bias', 'none'), o['batch_size'], o.get('epochs', 1000), o.get('lr', 0.005), o.get('gamma', 0.995), o.get('tag', ''))
        g = groups.setdefault(key, {'u': [], 'g': [], 'cu': []})
        g['u'].append(o['acc_unif']); g['g'].append(o['acc_gauss']); g['cu'].append(o['ce_unif'])
    print(f"{'sweep':6} {'bias':7} {'bs':>4} {'ep':>5} {'lr':>6} {'gamma':>6} {'tag':8} {'n':>4} "
          f"{'med unif':>8} {'p90':>6} {'max unif':>8} {'gauss@max':>9} {'minCE':>6} {'#match':>6}")
    for key, g in sorted(groups.items(), key=lambda kv: kv[0][:3]):
        u, ga, cu = torch.cat(g['u']), torch.cat(g['g']), torch.cat(g['cu'])
        i = u.argmax()
        print(f"{key[0]:6} {key[1]:7} {key[2]:>4} {key[3]:>5} {key[4]:>6} {key[5]:>6} {key[6]:8} {len(u):>4} "
              f"{u.median():8.4f} {u.quantile(.9):6.4f} {u[i]:8.4f} {ga[i]:9.4f} {cu.min():6.3f} {int(matches(u, ga).sum()):>6}")
