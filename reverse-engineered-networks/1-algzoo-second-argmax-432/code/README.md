# Code for the 432-parameter second-argmax RNN

Everything here runs with Python 3 and NumPy. No GPU or PyTorch is needed.

## The network

- `m16_10_weights.npz`: the network's weights as NumPy arrays: `W_ih` (16×1), `W_hh` (16×16) and `W_out` (10×16). The RNN has no biases:
  `h_t = ReLU(W_ih·x_t + W_hh·h_(t−1))`, `h_(−1) = 0`, `logits = W_out·h_9`.
- `m16_10.pth`: the same weights as the original PyTorch checkpoint (`rnn.weight_ih_l0`, `rnn.weight_hh_l0`, `linear.weight`). The model comes from ARC's AlgZoo collection: https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/

## The three programs

Each program defines `step(carry, prefix, upstream)`. It reads `x = prefix.values[prefix.position]`, updates its own state `carry`, and returns the 16-unit hidden state it predicts for that step. `upstream` is unused. Each file is self-contained: it needs NumPy only and does not read the network's weights.

| file | what it is | size |
|---|---|---|
| `idealised_account.py` | The core algorithm written literally: exact running maximum and runner-up, unit-circle clocks reset on replacement, comparator thresholds `S + (M − S)·f(clock)`, ideal pulses, and a fitted half-integer mix for the nine wave units. Solves the task well; not a faithful copy of the units. | 171 numeric literals |
| `full_account.py` | The network's own recurrence in the eigenbasis of its recurrent matrix, with the pulse units 4, 5, 9 kept separate, and constants rounded to fixed grids (1/64 to 1/2048). Faithful (about 99.6% agreement), but not shorter than the network's 272 pre-readout weights, and most of its coefficients are uninterpreted. | 334 numeric literals, 285 of them coefficients |
| `exact_rewrite.py` | The network's recurrence re-coded into named quantities (levels, clock, a shifting score register, pulses) at 5 significant digits. A long docstring explains each quantity; its statistics come from the authors' own samples and differ slightly from ours (for example 95.2% network accuracy, and 193° rather than our 182° median clock angle at age 4). This checks that the re-coding is right; it is not simpler than the weights. | 13 KB source, mostly docstring |

`accounts_working_notes.md` is the original write-up of the idealised and full programs by the team that produced them. It includes the ablation table behind the paper's Figure 5 and the threshold-substitution experiment. It is a working document, with some local vocabulary.

## Checking the programs against the network

```
python verify.py [n_rows] [seed]      # default 20000 rows, seed 0
```

For each program it reports task accuracy, agreement with the network, the fraction of the network's mistakes reproduced, KL divergence, and single-unit replacement agreement. `verify_output_131072_rows.txt` is our run with `python verify.py 131072 1`; it takes several minutes on one CPU core:

| program | task accuracy | agreement | mistakes reproduced | KL (nats) | worst single unit |
|---|---|---|---|---|---|
| network | 95.29% | — | — | — | — |
| idealised | 96.46% | 93.18% | 5.2% | 1.80 | 33.5% (unit 7) |
| full | 95.25% | 99.64% | 96.1% | 6.7e-4 | 99.55% (unit 7) |
| exact rewrite | 95.29% | 99.98% | 99.7% | 1.8e-6 | 99.94% (unit 7) |

## Figures

- `analysis.py` computes the worked example, the clock-angle statistics (4,000 inputs) and the near-tie statistics (20,000 inputs), and writes them to `analysis.json`.
- `readout_stats.py` computes how the comparators and the other units vote at the maximum's and runner-up's positions (20,000 inputs), and writes `readout_stats.json`.
- `figures.py` draws the PNG figures in `../figures/` from `analysis.json`. It needs matplotlib.

The removal-experiment figure uses numbers from `accounts_working_notes.md`, on 19,776 inputs that were held out from fitting the idealised account but used to choose the full account's rounding. We did not re-run those experiments.
