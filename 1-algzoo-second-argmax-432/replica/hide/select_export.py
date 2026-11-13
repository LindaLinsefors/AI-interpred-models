"""Pick the best seed of each final run and export it.

Selection: highest accuracy on a VALIDATION set (2^20 Unif(-1,1) inputs, seed 999),
separate from the test sets used for reporting (seed 12345, uniform and Gaussian).
Exports hide/models/<name>.pth (nn.RNN / nn.Linear state_dict names, like m16_10.pth)
and hide/models/<name>.npz, and prints a markdown table.
"""
import pathlib
import numpy as np
import torch
import batched

HERE = pathlib.Path(__file__).parent
ORIG = HERE.parents[2] / 'reverse-engineered-networks' / '1-algzoo-second-argmax-432' / 'code' / 'm16_10.pth'
NAMES = {'none': 'replica_nobias', 'hh': 'replica_bias_hh', 'hh_out': 'replica_bias_hh_out'}
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def to_state_dict(p):
    sd = {'rnn.weight_ih_l0': p['W_ih'], 'rnn.weight_hh_l0': p['W_hh']}
    if 'b_h' in p:   # one hidden bias b: stored as bias_hh, with bias_ih fixed at 0
        sd['rnn.bias_ih_l0'] = torch.zeros_like(p['b_h'])
        sd['rnn.bias_hh_l0'] = p['b_h']
    sd['linear.weight'] = p['W_out']
    if 'b_out' in p:
        sd['linear.bias'] = p['b_out']
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


@torch.no_grad()
def agreement(p, q, n=2**20, seed=12345):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = batched.sample(1, n, 'gauss', dev, g)
    return (batched.forward(p, x).argmax(-1) == batched.forward(q, x).argmax(-1)).float().mean().item()


if __name__ == '__main__':
    sd = torch.load(ORIG, weights_only=True)
    orig = {'W_ih': sd['rnn.weight_ih_l0'][None], 'W_hh': sd['rnn.weight_hh_l0'][None], 'W_out': sd['linear.weight'][None]}
    orig = {k: v.to(dev) for k, v in orig.items()}
    rows = [('original (ARC example_2nd_argmax)', '-', *[float(v) for v in (*batched.evaluate(orig, 'unif'), *batched.evaluate(orig, 'gauss'))], 1.0)]
    (HERE / 'models').mkdir(exist_ok=True)
    for f in sorted((HERE / 'runs_modal' / 'final').glob('*.pt')):
        o = torch.load(f, weights_only=False)
        p = {k: v.to(dev) for k, v in o['params'].items()}
        val_acc, _ = batched.evaluate(p, 'unif', seed=999)
        k = int(val_acc.argmax())
        best = {n: v[k:k + 1] for n, v in p.items()}
        name = NAMES[o['bias']]
        state = to_state_dict({n: v[0] for n, v in best.items()})
        torch.save(state, HERE / 'models' / f'{name}.pth')
        np.savez(HERE / 'models' / f'{name}.npz', **{n: v[0].cpu().numpy() for n, v in best.items()})
        rows.append((name, f"{o['seed']}/{k} (best of {o['K']}, val {val_acc[k]:.4f})",
                     float(o['acc_unif'][k]), float(o['ce_unif'][k]), float(o['acc_gauss'][k]), float(o['ce_gauss'][k]),
                     agreement(best, orig)))
        n_match = int(((o['acc_unif'] >= 0.9378 - 0.005) & (o['acc_gauss'] >= 0.9531 - 0.005)).sum())
        print(f"{name}: {n_match}/{o['K']} seeds within the match bar; median unif acc {o['acc_unif'].median():.4f}")
    print('\n| model | seed (base/index) | acc uniform | CE uniform | acc Gaussian | CE Gaussian | agreement with original (Gaussian) |')
    print('|---|---|---|---|---|---|---|')
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]:.2%} | {r[3]:.3f} | {r[4]:.2%} | {r[5]:.3f} | {r[6]:.1%} |")
