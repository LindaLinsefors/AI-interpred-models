# Replica of ARC's 432-parameter 2nd-argmax RNN, plus two bias variants

The original is ARC's `example_2nd_argmax()` (M₁₆,₁₀), the model in `reverse-engineered-networks/1-algzoo-second-argmax-432/code/m16_10.pth`. The three new models are in `hide/models/`, as `.pth` (same key names as `m16_10.pth`) and `.npz`:

| model | architecture | acc uniform | CE uniform | acc Gaussian | CE Gaussian | agrees with original (Gaussian inputs) |
|---|---|---|---|---|---|---|
| original | $h_t=\mathrm{ReLU}(W_{ih}x_t+W_{hh}h_{t-1})$, $\ell=W_{out}h_9$ | 93.78% | 0.188 | 95.31% | 0.147 | — |
| `replica_nobias` | same as the original | 95.63% | 0.149 | 96.10% | 0.133 | 93.9% |
| `replica_bias_hh` | $h_t=\mathrm{ReLU}(W_{ih}x_t+W_{hh}h_{t-1}+b_h)$ | 91.15% | 0.243 | 91.81% | 0.285 | 90.4% |
| `replica_bias_hh_out` | as above, plus $\ell=W_{out}h_9+b_{out}$ | 92.60% | 0.210 | 92.29% | 0.257 | 90.5% |

Test sets: 2²⁰ inputs each, iid Unif(−1,1) or iid N(0,1), length 10. The bias variants have a single hidden bias $b_h$, stored as `rnn.bias_hh_l0` with `rnn.bias_ih_l0` = 0.

## How the original was trained

The example model is not one of the official AlgZoo models. ARC's blog post says it "was trained before we standardized our codebase" by Zihao Chen. His notebook (Colab `1yRv18_Gp0twoZ-kptMygVhAq-azj2bBq`, which loads `one_layer_16_hidden_94_acc_2nd_argmax.pth`) records this setup:

- `nn.RNN(1, 16, nonlinearity='relu', bias=False)` + `nn.Linear(16, 10, bias=False)`, PyTorch default init
- inputs iid **Unif(−1,1)** (not Gaussian), a fresh batch at every step; cross-entropy on the logits from the final hidden state
- Adam(lr = 0.005, weight_decay = 1e−14), ExponentialLR(γ = 0.995) stepped once per epoch
- 1000 epochs × 100,000 samples
- **Not recorded:** batch size (the variable is undefined in the notebook) and seed

The official zoo recipe is different: AdamW, Gaussian inputs, 2³⁰ sequences, batch 1024. It was not used here.

## The recipe I settled on

The notebook setup above, with **batch size 64**, keeping the **best of 32 seeds**. "Best" means highest accuracy on a separate Unif(−1,1) validation set.

The batch size comes from a sweep of 128 seeds per batch size, all with the notebook setup. A seed "matches" if it is within 0.5 percentage points of the original, or above it, on both test sets:

| batch size | median acc uniform | 90th percentile | best | seeds that match |
|---|---|---|---|---|
| 1000 (3 seeds, CPU) | 77.97% | — | 83.66% | 0/3 |
| 512 (3 seeds, CPU) | 82.14% | — | 82.91% | 0/3 |
| 256 | 87.06% | 91.47% | 94.18% | 1/128 |
| 128 | 89.53% | 92.71% | 94.83% | 6/128 |
| **64** | **90.66%** | **93.37%** | **95.74%** | **11/128** |
| 32 | 89.69% | 93.09% | 95.52% | 9/128 |
| 16 | 88.52% | 92.36% | 94.56% | 4/128 |

So the original is roughly a top-10% seed of this recipe, and best-of-32 at batch 64 matches about 94% of the time (1 − 0.914³²). The final no-bias run used fresh seeds: 2 of its 32 seeds match, and the best one (95.6%) is 1.8 points *above* the original. The three CPU runs that used PyTorch's own `nn.RNN` at batch 64 reached 92.5%, 93.5% and 91.4%, consistent with the batched GPU trainer.

## Bias variants

Same recipe, best of 32 seeds, with no other changes. Both train noticeably worse than the bias-free model. Over the 32 seeds, median uniform accuracy is 75.3% (W_hh bias) and 73.4% (W_hh + W_out bias), against 90.5% without bias, and no seed reaches the original. The bias variants also do worse on Gaussian than on uniform inputs, unlike the bias-free models. That fits a loss of positive homogeneity: without biases the network is scale-equivariant, so training on Unif(−1,1) covers every input scale; with biases it is not.

## Files (`hide/`)

- `batched.py`: trains K independent copies at once on one GPU, with the whole training step captured in a CUDA graph. It is exactly K independent runs of the recipe.
- `modal_sweep.py`: fans configurations out over Modal H100s. `runs_modal/` holds every trained seed (weights + accuracies).
- `train_replica.py`: a direct copy of the notebook's trainer (CPU, `nn.RNN`); `runs/` holds its 15 runs.
- `analyse.py` (sweep tables), `select_export.py` (best-seed selection + export to `models/`).
