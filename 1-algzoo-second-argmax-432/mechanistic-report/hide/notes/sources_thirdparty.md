# Third-party (non-ARC) material on AlgZoo's 2nd-argmax RNN $M_{16,10}$

Compiled 2026-11-19 by a search agent. Scope: analyses by people outside ARC of `example_2nd_argmax()` (16 ReLU units, no biases, 10 inputs, 432 parameters), of other AlgZoo 2nd-argmax RNNs, and closely related literature.

**How it was gathered.** LessWrong GraphQL and site search; GitHub repository search and shallow clones read as text (no third-party code was run); X's public syndication endpoint; WebFetch and WebSearch; arXiv listing search. Text in quotation marks is verbatim from the URL given. "(own check)" means I recomputed the number from `m16_10_weights.npz` with the scripts `thirdparty_check_*.py` in this folder (outputs beside them). "(inference)" marks my own reading.

**Conventions that differ between sources.**
- ARC and most third parties write $h_0 = 0$, $h_{t+1} = \mathrm{ReLU}(W^{hh} h_t + W^{hi} x_t)$, logits $= W^{oh} h_{10}$. This workspace writes $h_t = \mathrm{ReLU}(W_{ih} x_t + W_{hh} h_{t-1})$, logits $= W_{out} h_9$. Their $h_{t+1}$ is our $h_t$. Neuron indices 0–15 are the same everywhere (own check: $W_{ih}[4] = 10.157$).
- "Age" is off by one between authors. Goldowsky-Dill's $\delta = t - p_t \ge 1$ is measured on the recurrent drive $W_{hh} h_{t-1}$, before $x_t$ is read.
- Inputs: $\mathcal N(0,1)$ in ARC's post and most repos; Unif$(-1,1)$ in Goldowsky-Dill (and Chen's ARC notes); Keane's docs mostly use two-impulse inputs (1.0 and 0.8, all others 0) and U[0,1] (89% accuracy there).
- Almost every October 2026 repository says it was written with AI agents (Claude, GPT). Treat their numbers as claims; §6 lists the ones I checked.

**Why there was a burst in mid-October 2026 (inference from dates and from what Snider and Skyway1111 say about their motivation).** Nick Cammarata's Goodfire essay "Neural Networks Are Already Clean" is dated 2026-10-15 in Goodfire's research index (essay: https://already-clean.goodfire.com/). On 2026-10-16 Jacob Hilton (ARC) posted on X, in a thread about that essay: "The challenge of complete understanding is well exhibited by my 432-parameter model, which fits in a tweet and remains beyond current techniques." (https://x.com/JacobHHilton/status/2110915999356817588). Ten of the repositories below were created between 2026-10-16 and 2026-10-30.

## 1. Coverage at a glance

| # | Source | Author | Date | Subject | Mechanistic value |
|---|---|---|---|---|---|
| 3.1 | LW comment, gist, repo `NixGD/rnn-explaining` | Nicholas Goldowsky-Dill | 2026-01-28 to 02-10 | comparators 1, 6, 7, 8 of $M_{16,10}$ | high: $a_n(\delta) M + b_n(\delta) S^+$ fit, transition algebra |
| 3.2 | LW comments, gist, fork `mrsirrisrm/alg-zoo` (docs 01–49, `estimate/`) | Martin Keane | 2026-01-27 to 03-17; 10-17 to 10-27 | whole $M_{16,10}$ | high, but a research log with many retractions |
| 3.3 | `DraggonZ/algzoo-interp-public` | GitHub user DraggonZ (git author "Zhaslan") | 2026-10-16 to 10-22 | whole $M_{16,10}$, hand-built twins, 37 sister networks | very high: most complete account found |
| 3.4 | LW post, repo `jsnider3/algzoo-432-interp`, alg-zoo issue #1 | Josh Snider | 2026-10-17 to 10-26 | whole $M_{16,10}$, zoo seeds, 8,000 retrains | high: clocks, write pointer n5 |
| 3.5 | `Skyway1111/RNN432`, site "The runner-up" | GitHub user Skyway1111 | 2026-10-19 | whole $M_{16,10}$ | high: core/satellites, phase wheel, votes |
| 3.6 | "Inside a second-argmax network" | David Khachaturov | 2026-10-26 | whole $M_{16,10}$ as a 16-coordinate program | medium-high: exact local proofs, false-record error |
| 3.7 | `panaali/beyond-sampling`, `m16/FINDINGS.md` | Ali Panahi | 2026-10-16 to 10-21 | whole $M_{16,10}$ and estimators | high: Fourier-memory table, separable code, failed hypotheses |
| 3.8 | Preprint in `datasci3nce/causal-geometry-algzoo` | Kishore Kumar Mariappan | 2026-03-08 | $M_{16,10}$ | low |
| 3.9 | `mogar/algzoo_interp` | GitHub user mogar (git author "morgan") | 2026-10-21/22 | mainly 32×10; a neuron dictionary for $M_{16,10}$ | medium |
| 3.10 | `krandder/arc-mean-estimation`, lane `algzoo-noisy` | git author "kas.eth" | 2026-10-29 | noisy fine-tunes of $M_{16,10}$ | low-medium |
| 3.11 | `dinamicoplus/algzoo-legibility`; `WFJKK/Mech-Interp-Agent-level1`; "phoenix" comments | Marcos Brito; WFJKK; LW user phoenix | 2026-10-16; 11-19; 2026-06 to 08 | $M_{16,10}$ | none to low (scripts, scores, or unverifiable) |
| 4 | Other AlgZoo models: mogar 32×10 and 4×3; `kiankyars/alg-zoo`; `jc1122/...circuit-atlas`; `Xabilimon1/leeuwenhoek` | various | 2026-10/11 | other sizes, argmedian | context only |

An arXiv search for "AlgZoo" (titles, abstracts and metadata, not full text) returns nothing (§9). Jacob Hilton, 2026-08-17: "I don't recall seeing any specific effort on these models in the last six months outside of the comments on this post. To me, that progress is fairly marginal and we are not much closer to succeeding at the challenge given at the end of the post." (https://www.lesswrong.com/posts/x8BbjZqooS4LFXS8Z/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/comment/ibaSPBJRTRC8jgHTq)

## 2. Vocabulary map

| This workspace | ARC post | Goldowsky-Dill | Keane | DraggonZ | Skyway1111 | Khachaturov | Snider | Panahi |
|---|---|---|---|---|---|---|---|---|
| comparators 1, 6, 7, 8 | leave-one-out-max features (1, 6, 7) | negative-input neurons | comps | hiding neurons, "cancellers" | comparators (core) | comparisons | comparators | $x_9$ comparators |
| unit 2 | running max | max register | n2 latch | store | running max, "the record" | carry | value register | integrator |
| pulse 4 | increment of the running max | new-max detector | n4 relay / impulse detector | detector | increment, "the jump" | record | new-max detector, max marker | record detector |
| pulse 5 | – | – | bridge; "single-impulse handler" | keep gate | timer for position 8 | candidate | write pointer | near-record |
| pulse 9 | – | – | secondary relay | late gate | timer (faint) | relay | max marker | near-record |
| wave units 0, 3, 10–15 | – | – | waves 0, 10, 11, 12, 14; bridges 3, 13, 15 | memory | satellites, timers | carriers | clock neurons | clocks 0, 10, 11, 12, 14, 15; harmonics 3, 13 |

## 3. Sources about $M_{16,10}$

### 3.1 Nicholas Goldowsky-Dill: comparators as $a_n(\delta) M + b_n(\delta) S^+$

- Comment, 2026-01-28: https://www.lesswrong.com/posts/x8BbjZqooS4LFXS8Z/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/comment/Y6PSmNSP6sqbBev66
- Gist `neurons_n1_n6_n7_n8_circuit.py`, 2026-01-28: https://gist.github.com/NixGD/2651be79bfd22bcb13d1f609f23439a4
- Repository, last commit 2026-02-10 (`801dee7`): https://github.com/NixGD/rnn-explaining

Claims, verbatim from the comment:
- "I think I understand the role of {N1, N6, N7, N8} reasonably well. The activations post-$W_{hh}$ are well approximated by the linear model $W_{hh} h_{n, t} \approx a_n(\delta) \cdot M_t + b_n(\delta) \cdot \text{max}(S_t, 0)$ where $M_t$ is the running max, $S_t$ is the second running max, and $\delta$ represents how long ago the max-value occurred. The coefficients change with delta in pleasing patterns".
- "This model fits the activations well ($R^2 = 0.992$)", footnote "though more like 0.95 for some subsets".
- Limits he states: "I only have a partial mechanistic understanding of how the weights lead to this behavior." and "There are large parts of the model I haven't looked at at all, e.g. the other 10 neurons. There are also parts of the task that I don't know how the model does, e.g. tracking the current position of the 2nd-maximum value)."

From the gist (Unif$(-1,1)$ inputs, samples with $M > 0.1$, $t \ge 2$):
- Notation: $c_{n,t} = (W_{hh} h_{t-1})_n$, $M_t = \max(x_0, \dots, x_{t-1})$, $\delta = t - p_t$, and "$S_t^+ = \max(0, S_t)$ | clipped 2nd-max (N2 cannot represent negative values)".
- Case 1 ($x_t$ small). With $\mathcal N = \{1, 6, 7, 8, 2\}$: "$\sum_{j \in \mathcal{N}} W_{hh}[n,j] \cdot w_{ih}[j] \approx 0 \quad \text{for all } n \in \mathcal{N}$", so the input term drops out and "$\mathbf{a}(\delta + 1) = W_{\text{sub}} \, \mathbf{a}(\delta), \qquad \mathbf{b}(\delta + 1) = W_{\text{sub}} \, \mathbf{b}(\delta)$" with $W_{\text{sub}} = W_{hh}[\{1,6,7,8\}, \{1,6,7,8,2\}]$. His verdict: "In some cases (especially N7), the predicted next-coefficients are a good match for the values predicted from the weights. Other times the match is far worse. The most likely possibility is that other neurons are important for understanding this circuit!"
- Case 2 ($x_t > M$, reset): "N2 | Max register | $h_{2,t} \approx \kappa_2 \cdot M_{\text{old}}^+$", "N4 | New-max detector | $h_{4,t} \approx \kappa_4 \cdot (M_{\text{new}} - M_{\text{old}}^+)$", giving "$a_n(1) = W_{hh}[n,4]\,\kappa_4, \qquad b_n(1) = W_{hh}[n,2]\,\kappa_2 - W_{hh}[n,4]\,\kappa_4$".
- Case 3 ($S < x_t < M$): "TODO. I've looked into this a bit, but wasn't as easily able to deduce a clear mechanistic explanation for the activation transition."

From the repository README:
- All 16 neurons: "h_t[n] ≈ a(dm,ds)·M^+ + b(dm,ds)·S^+ + c(dm,ds)·x_t + d(dm,ds)" per cell (steps since max, steps since 2nd max); "**R² > 0.95 for all neurons** using this model with appropriate fitting method."
- Neurons 1, 6, 7, 8 use statistics that include the current input ($M_t$, $S_t$); all others use $M_{t-1}$, $S_{t-1}$.
- The Case 1 transition "holds with ~10-20% relative error for most neurons."
- "Replacing neuron activations with linear model predictions maintains >90% accuracy for most neurons".
- Notes in `analysis/output.py`: "N0 is active except when delta_m is precisely 2"; N5 "almost always active" late in the sequence when "delta_m == 1 OR delta_s == 1"; N15 "seems like it can be well explained by $h_{11} - h_{10}$".
- `analysis/position_fits/summary.py`: "N5 is an outlier" in the transition checks; N3 and N15 have fit $R^2 \approx 0.92$–$0.93$; Cases 2 and 3 are listed as remaining.

The coefficient values exist only as a figure in the comment. I re-ran his fit (own check, `thirdparty_check_comparator_fit.py`, 400k Unif$(-1,1)$ lists); the numbers match the figure:

| $\delta$ | $a_1$ | $a_6$ | $a_7$ | $a_8$ | $b_1$ | $b_6$ | $b_7$ | $b_8$ |
|---|---|---|---|---|---|---|---|---|
| 1 | 8.82 | −0.47 | 13.40 | −0.35 | 1.76 | 10.95 | −0.11 | 12.90 |
| 2 | −0.15 | 7.53 | 13.35 | 3.76 | 10.02 | 2.84 | −0.06 | 8.85 |
| 3 | 3.14 | 11.51 | 13.23 | 0.55 | 6.58 | −0.56 | 0.01 | 11.72 |
| 4 | 8.84 | 12.88 | 11.50 | 0.75 | 1.49 | −1.65 | 1.60 | 11.46 |
| 5 | 13.40 | 11.79 | 9.33 | 5.31 | −2.44 | −0.70 | 3.46 | 7.06 |
| 6 | 14.50 | 9.08 | 7.93 | 11.71 | −3.44 | 1.53 | 4.62 | 1.11 |
| 7 | 11.87 | 6.22 | 7.91 | 16.65 | −1.34 | 3.90 | 4.67 | −3.54 |
| 8 | 6.98 | 4.65 | 9.22 | 17.63 | 2.82 | 5.31 | 3.56 | −4.78 |
| 9 | 2.30 | 5.05 | 11.21 | 14.33 | 7.06 | 5.16 | 1.85 | −2.20 |

(Inference from this table, not stated by him.) $(a_n + b_n)/|W_{ih}[n]|$ lies in $[0.89, 1.07]$ for every $n$ and $\delta$, so

$$c_{n,t} \approx |W_{ih}[n]|\,\big[\,S^+ + \alpha_n(\delta)\,(M - S^+)\,\big], \qquad \alpha_n(\delta) = \frac{a_n}{a_n + b_n}.$$

Each comparator compares $x_t$ with a threshold between $S^+$ and $M$ (sometimes outside that interval: $\alpha$ ranges from $-0.05$ to $1.37$). The values are $\alpha_7 = 1.01, 1.00, 1.00, 0.88, 0.73, 0.63, 0.63, 0.72, 0.86$; $\alpha_6 = -0.05, 0.73, 1.05, 1.15, 1.06, 0.86, 0.61, 0.47, 0.49$; $\alpha_1 = 0.83, -0.01, 0.32, 0.86, 1.22, 1.31, 1.13, 0.71, 0.25$; $\alpha_8 = -0.03, 0.30, 0.05, 0.06, 0.43, 0.91, 1.27, 1.37, 1.18$ for $\delta = 1..9$. A constant plus one sinusoid at 42.56° per step fits $a_7(\delta)/|W_{ih}[7]|$ well ($R^2 = 0.97$) but fits neurons 1, 6, 8 less well ($R^2$ 0.7–0.9): at early ages their threshold sits at $S^+$ ($\alpha \approx 0$ for neuron 6 at $\delta = 1$, neuron 1 at $\delta = 2$, neuron 8 at $\delta = 1, 3, 4$).

### 3.2 Martin Keane: phase wheel, reset-and-rebuild, n4→n2 latch

- Comments, 2026-01-31 to 02-05: https://www.lesswrong.com/posts/x8BbjZqooS4LFXS8Z/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/comment/TJ7k2h2nDqD3GBFmq and replies `7dv5LwLsroHiiQC23`, `XWmK7zrPQ3jQZrcjc`, `XcwuoNR4w8EJbc9w9`
- Gist `mechanistic_RNN.py`, 2026-02-05: https://gist.github.com/mrsirrisrm/d6850ff8647d1ed2f67cc92d5bce3ed0
- Fork with `docs/01`–`49` (2026-01-27 to 03-17) and `estimate/` (2026-10-17 to 10-27), head `48917b7`: https://github.com/mrsirrisrm/alg-zoo

Later docs retract earlier ones (doc 20 retracts the "anti-phase interference" of docs 08, 11, 15; doc 16 retracts a "ReLU shifts the frequency" claim). I read docs 11, 16, 18, 20, 22, 27, 32, 37, 48, 49 and `estimate/ARC_REPORT.md`, `EXPLAINER.md` and parts of `FINDINGS.md` in full; the rest only through his own summaries.

Comments:
- "Zeroing out all the inputs except the largest two gets the network to 100%". "There appear to be two different ways in which the output is encoded, depending on whether the 2nd largest input comes before or after the largest." "n7 does a fairly pure comparison with the running max val."
- "The hidden state follows a spiral trajectory through time, implemented by a rotating phase in the hidden state. The W_out projection converts phase angle to position logits." Forward ($M$ first) and reverse ($S$ first) final states differ by "offset(m, s) = f(m) + g(s)  where  g(s) = -f(s)"; "The offset has effective rank ~2"; "The discrimination offset is smaller, ~1/4 magnitude."
- Method of the gist: "deriving a sequence D[t] of which ReLU are active during the sequence - this turns out to be quite stable after an impulse. Then we can mask W_hh via D[t] and combine all the (now-linear) steps to get an effective linear operator for the whole sequence". Result: "In the forward direction, spectral radius is 0.76 – 1.05 with a small spectral gap, while in the reverse direction it is 1.37 – 2.91".

Doc 27, "Mechanistic Reference" (two-impulse inputs only):
- Groups (columns: group, neurons, $W_{ih}$, role): "Comps | n1, n6, n7, n8 | -10 to -13 | Clip to 0 on input; rebuild trajectory encodes position"; "n4 | n4 | +10.16 | One-shot broadcast relay (self-recur = -0.99)"; "n2 | n2 | +0.015 | Memory latch (self-recur = 0.97); context-dependent"; "n9 | n9 | mixed | Secondary relay; critical for small-gap reversed".
- "**The mechanism is reset-and-rebuild-with-memory**: both impulses clip comps to 0; the rebuild trajectory after the last clip depends on residues left by the first impulse (via n2, waves, bridges)."
- "**n2→n4 feedback (-0.49) is a gate, not gain control**."
- "**The encoding is holographic — all 16 neurons are required for readout**."
- A third impulse $T$ after $S$ with $T$ close to $S$ is a failure mode: accuracy 65.8% at $T = 0.79$ ($S = 0.8$), and "Every failure predicts Tt instead of St".

Doc 20: after a clip, n7 follows "0.0  13.4  13.3  13.3  11.6   9.7   8.6   8.8  10.2  11.9" at 0..9 steps (own check: identical). "n7 primarily encodes max_pos"; "n8 encodes both positions".

Doc 22 (n4→n2 latch): when the max (1.0) arrives after an earlier 0.8, n4's pre-activation is input $+10.16$ plus recurrent $-7.16$, total $+3.00$, and the recurrent term is "n2 × W_hh[4,2] = 14.67 × (-0.492) = -7.22". His reading: "**n2's latched value feeds back negatively into n4**". After a clip, n4 cancels n2 in n7: "n7 | +9.43 | -9.42 (via W_hh[7,4]=+1.317) | **≈ 0**".

Doc 18 (single impulse): zero-crossing delays "n4 at +1, {n0,n9,n14} at +2, {n5,n15} at +3, {n11} at +5, {n10} at +6" (own check: n0 and n14 are 0 at age 2, n15 at ages 3–4, n11 at 5, n10 at 6). "Superposition fails catastrophically"; the double-impulse response is always below the sum. "Even A=0.1 switches the prediction from pos 9 to pos 3."

Doc 32 (phase wheel): the readout taken at step $t$ after both impulses counts down, "pred(t) = S_pos + (9 - t)", for example "M@0, S@1: predictions = [4, 9, 8, 7, 6, 5, 4, 3, 2, 1]" (own check: identical). "Single impulse at t=0: predictions = [4, 6, 9, 8, 0, 9, 9, 9, 9, 9]" (own check: identical). W_out row correlations: "0 ↔ 1 | +0.51", "0 ↔ 4 | -0.73".

**Caution on his eigenvalues.** Docs 32, 48, 49 use the full 16×16 $W_{hh}$: "0.43 ± 1.03j | 1.115 | 5.4 steps | Primary oscillator (growing)", spectral radius 1.27. Units 4, 5, 9 are off on most steps, so this is not the operator the state normally sees. An independent review inside `panaali/beyond-sampling` (`m16/approaches/mechanism_independent_review/README.md`) concludes: "The apparent5.4/8.4 discrepancy is raw Wh versus the common gated matrix: independent periods5.37064 and8.42149 on the same public checkpoint." The same review finds that the antisymmetry $g = -f$ is produced by the fitting procedure: "Balanced offset fitting forces G=-F algebraically even for unlearned synthetic data."

`estimate/FINDINGS.md` (October 2026), section "Mechanism (derived from weights, then checked)":
- "n2 = memory: n2' = 0.97 n2 + 1.73 n4; on a record of value a with n4 = 10.16a − 0.49 n2, n2' = 17.58 a + 0.12 n2. Measured median n2/runmax = 17.6 (IQR 16.6–18.4)." In symbols: $h_2' = 0.97\,h_2 + 1.73\,h_4$ and, on a record $a$, $h_4 = 10.16\,a - 0.49\,h_2$, so $h_2' = 17.58\,a + 0.12\,h_2$.
- "after M's record, n2 = 17.58 vM + 0.122·n2_old, where 0.122 = 0.97 − 1.73·0.49 — n2 keeps a ~12% trace of the previous record." (Own check: $0.9688 + 1.7307 \cdot (-0.4923) = 0.117$.)
- "n4 = soft comparator vs n2. One-term threshold 0.49·17.58/10.16 = 0.848·runmax is WRONG ... mean threshold ≈ 0.94, but context-dependent: P(n4 on) = 0.25 at x = 0.90·max, 0.90 at 0.98, 0.994 at 1.00. Main threshold movers after n2: n4 self (−1.0), n6 (+0.93), n8 (−0.76)."
- "n4 active ⇔ x_t is a new running max: 98.5% agreement for t ≥ 3. No neuron tracks "new 2nd max" directly."
- "n7 compares the fresh input to it: 0.674 n2 - 13.17 x_t carries 88% of n7's pre-activation variance and its sign 94% of the time; n7 is off once x_t exceeds ~0.90x the running max."
- "The 5x5 block also has a complex pair 0.69 ± 0.63i (|0.93|, ~42°/step), linear-regime only." (block $\{1, 2, 6, 7, 8\}$)
- Decomposition of the task: "S = argmax(A, B) with A = max before M (the penultimate record), B = max after M", with the error budget 0.0238 (wrong element within $S$'s side), 0.0118 (A/B swap), 0.0095 (predicts $M$), 0.0017 (other), total 0.0468.
- Ground truth: "acc = 0.9531186 ± 1.1e-6 (N = 2^35, RTX 3090, 6.5 min)"; "Failures: 63% predict the 3rd largest, 20% predict the max, 16% other."
- Stiffness: the Gauss-Newton Hessian is "63% in the 25 W_hh weights among {n1, n2, n6, n7, n8}"; top direction "n2->n7 (0.24 of its energy), n7->n8, n2 self, n7->n1, in->n7, n6->n1".
- Settling: "a near M/S tie is 59-82% right immediately and 84-98% one benign step later"; comparisons that arrive at the last steps "are read out unsettled".
- A fit of all 16 neurons on $M$, $S$, $x_t$ and one-hot lags: with the joint (lag $M$, lag $S$) table "the surrogate is better than the net on every h16 model (0.960-0.989 against 0.877-0.953)", so "the lag table carries the algorithm, and the network's errors, its tie strips included, live in the 6-17% residual from it."
- `ARC_REPORT.md`: "No deductive estimator of the accuracy beats Monte Carlo on M(16,10) above about 0.4 MFLOP"; and "plain continued training takes it from 0.9531 to 0.9614".

### 3.3 DraggonZ: "store, detector, keep gate, late gate, hiding neurons, memory"

- https://github.com/DraggonZ/algzoo-interp-public, head `bc04f72` (2026-10-22). README: "This is a personal project, not affiliated with ARC" and "The research, code and text were done with AI assistance."
- Read in full: `research/20261017_handbuilt/NOTE.md`, `FINDINGS.md`, `WRITEUP_ladder.md`, `memory/README.md`, `hiding/README.md`, `lit_review/README.md`, `research/20261016_m16_10/REPORT.md`. Read in part: `gates/`, `switching/`, `neuron3/`, `twin_tests/`, `parameters/memory_register/`, `parameters/spd/`, `robust_twin/DESIGN.md`, `faithful_twin/LEDGER.md`, `superposition/norule/`. The repository has 3,496 files; this is not exhaustive.

Algorithm (`WRITEUP_ladder.md`): "Five groups do the work. The **store** holds the largest number so far, and a **detector** fires when a new largest arrives. A **keep gate** lets a number vote only if it is at least about three-quarters of the second-largest so far. The **hiding neurons** cancel the vote of the current largest. The **late gate** keeps a new largest hidden for the step after it arrives. The **memory** holds the votes and rotates them each step, so each lands on its own position at the readout." In one sentence: "Every number big enough to matter casts a vote for its own position. Four neurons hide the vote of the current largest, and a bigger number switches them off, so the old largest's vote shows just as it becomes the second-largest."

Interventions: "silencing the hiding neurons once, right after the largest arrives, makes the network name the largest on about 70% of lists; silencing any other four neurons gives at most 24%. Switching off the keep gate drops accuracy to 21%. Switching off the late gate drops lists whose answer is an overtaken largest from 95% to 49%."

Store and detector (2, 4), from `FINDINGS.md`:
- "The store holds about 17.6 × the largest, one step behind, and fades about 3% per step. Removing the fade costs 10 points".
- "The store also carries when the largest arrived (a slow wave, up to about a sixth of its value). Hiding and memory neurons use it."
- "The store is an exact weighted sum of the largest and the previous largest (weights by age)" (`NOTE.md`).
- "The detector's pulse at the overtaking is about 2.2 times the true jump, because the faded store sets its threshold at 0.97 of the old record." (`hiding/README.md`)

Hiding neurons (1, 6, 7, 8), from `hiding/README.md`:
- Exact definition: "ref = (W_hh h(t−1))_n / c_n, so its pre-activation is exactly c_n (ref − x_t), with c = 10.56, 11.00, 13.17, 12.31 for neurons 1, 6, 7, 8 (minus their input weights)." That is $\mathrm{pre}_n(t) = c_n\,(\mathrm{ref}_n(t) - x_t)$.
- Rule: "A reference is a fixed mix of the previous largest and the largest, set by the steps since the largest arrived. Neuron 7's is the largest. Neurons 6 and 8 hold the previous largest one step after a record, neuron 1 two steps after, neuron 8 for about four steps."
- Base table ($M$ largest, $P$ previous largest, $K$ steps since $M$ arrived, for $P$ arriving 1 step before $M$):

| $K$ | neuron 1 | neuron 6 | neuron 7 | neuron 8 |
|---|---|---|---|---|
| 1 | $0.18P + 0.84M$ | $1.01P$ | $1.02M$ | $1.09P$ |
| 2 | $0.96P$ | $0.24P + 0.69M$ | $1.01M$ | $0.75P + 0.31M$ |
| 3 | $0.57P + 0.30M$ | $1.05M$ | $1.01M$ | $0.96P + 0.06M$ |
| 4 | $0.06P + 0.85M$ | $-0.15P + 1.17M$ | $0.13P + 0.88M$ | $0.87P + 0.09M$ |
| 5 | $-0.29P + 1.26M$ | $1.07M$ | $0.27P + 0.72M$ | $0.46P + 0.47M$ |

- Full written rule: "ref_n(t) = α_n(K, d) P + β_n(K) M + Σ_j v_n(t − j) x_j [x_j kept] + Σ_j Σ_i T_ni(t − j) e_i(j)", where $e_i(j)$ is the excess of the number at step $j$ over neuron $i$'s reference when neuron $i$ was clipped. 648 fitted numbers; substituted for the four neurons it gives 94.3% (network 95.2%).
- Clipping computes a running maximum: "A ReLU that clips c (ref − x) passes on c (x − ref) for x above ref; through the four's weights that amount lands on another neuron's reference. For neuron 6 into neuron 1 the ratio is 1.04, so the reference becomes ref + (x − ref) = x: a maximum of the old reference and the new number. This is how the group holds the running second-largest without a dedicated neuron."
- Own block: "The four's own block (their weights among themselves) keeps one rotating pair, 43° per step shrinking 6% per step; the other two directions die within a step (eigenvalue sizes 0.937 and 0.043...). The arriving number's push (size 23.6) and the store's push (1.18) lie almost entirely in the dying directions: one step of the block leaves 0.19 and 0.06 of them. The detector's push (1.58) survives (1.43). So a number's direct term cancels across the group one step later, the level is supplied afresh every step from the store, and only the record's jump starts a lasting rotation."
- "Blindness" (`FINDINGS.md`): "every neuron except the gates reads the hiding neurons at right angles to their input weights, so the arriving number cancels." (Own check: the cosine between row $W_{hh}[m, \{1,6,7,8\}]$ and $(c_1, c_6, c_7, c_8)$ is within $\pm 0.04$ of 0 for all $m$ except $m = 5$ ($-0.74$) and $m = 9$ ($-0.69$).)
- "ARC's store weights into the four are −0.048 to −0.051 times their input weights, within 3%." (Own check: $-0.0477, -0.0487, -0.0512, -0.0508$.)
- "They are rebuilt every step: a 10% nudge leaves 2% one step later."
- Partial switching (some of the four off): "Undoing that partial clipping costs 24 points." "One step after a record it supplies the vote of a number between the two largest, as a hinge on how far it exceeds the previous largest".
- Erasure: "The number right after a record keeps its vote in the hiding neurons (40 per unit), so an overtaking wipes it."

Gates (5, 9), from `gates/README.md`. With $r_j$ the switch point of neuron $j$:

$$\mathrm{gate}(t+1) = \mathrm{ReLU}\Big(D + \sum_j s_j\,\mathrm{hinge}_j\Big), \quad \mathrm{hinge}_4 = (x_t - r_4)^+, \quad \mathrm{hinge}_j = (r_j - x_t)^+ \text{ for } j \in \{1, 6, 7, 8\},$$

with slopes "keep gate (5) | +12.06 | −0.01 | +0.17 | −7.81 | −10.46" and "late gate (9) | +1.48 | −3.77 | −0.98 | −66.26 | −13.44" for $j = 4, 1, 6, 7, 8$.
- Keep gate: "neuron 5 is on one step after 99.8% of records. For other numbers its share on rises from 39% at 0.70-0.75 of the second-largest to 84% at 0.80-0.85 and 99.7% at 0.9-1.0". "So the threshold is about 13.0/18.0 = 0.72 of the second-largest". Its job after a kept number: "the gate's write gives the number its vote and also re-hides the part of the largest's vote that the number uncovered by switching hiding neurons off."
- Late gate, short rule: "n9(t+1) = ReLU( 1.77 m1 − 0.48 m2 − 0.08 x_t − 0.23 x_{t+1} + 0.08 n9(t) + 1.48 (x_t − 0.933 m1 − 0.012 m2)⁺ − 66.3 (m1 − x_t)⁺ )". "One step after the overtaking, neuron 7 is off, which releases the late gate (its weight from neuron 7 is −5.03...). The late gate reads the old record's copy from the memory and fires in proportion to it. Its push on neurons 1 and 6 one step later (weights −0.69, −0.64) shifts the four's pattern so that the old record's position gains and the new largest's loses."
- "Neuron 8 holds the second-largest only while the largest is young. Up to four steps after a record its switch point is about the second-largest; from six steps on, about the largest."

Memory (0, 3, 10–15), from `memory/README.md`:
- "Eigenvalues of the 8x8 memory block (all on): 0° at 1.025, ±38.3° at 0.815, ±83.5° at 0.996, ±126.6° at 0.858, 180° at 0.918." (Own check: identical.)
- "Removing the ReLU from all eight memory neurons gives 44.4%". "Neuron 0 is off on 100% of age-2 steps"; "Neuron 10 is off at age 6 (94-96%), neuron 11 at age 5 (85-97%), neuron 12 at age 4 (14-43%)." Interpretation: "Each memory neuron switches off where the current largest's turning wave crosses zero in it."
- Linear machine: "m_t = D_t (A m_{t-1} + W_hh[memory, others] o_{t-1} + w x_t)", with $D_t$ the on/off pattern.
- Where writes land: "For most writers, a unit of activity written at step t is read at the end mainly at position t−2".
- Readout shift relation: "The cosine of W_oh[k+1, memory] with W_oh[k, memory] times the block is 0.97, 0.99, 0.99, 0.99, 0.88, 0.99 and 0.96 for k = 0-6 ... It fails into positions 8 and 9 (0.41 and 0.31)." (Own check with centred rows: 0.97, 0.99, 0.99, 0.99, 0.88, 0.99, 0.96, 0.40, 0.29.) "The level mode is read mainly at position 9".
- Neuron 3: "Neuron 3's two largest input terms come from hiding neuron 1 (weight +1.20...) and hiding neuron 8 (−1.07...). They correlate at −0.91". Later (`FINDINGS.md`): "Neuron 3 is a memory coordinate (about two-thirds of the 127° rotation) that also carries the gap between the largest and the second-largest (hiding 1 minus hiding 8 ≈ age factor × gap)".
- Neuron 15: reads "memory neuron 10 (−1.09...) and 11 (+1.28...)"; "Neuron 15 is off when the summed waves of the numbers so far are below zero."
- Compact switching rule (`switching/README.md`): "Neuron 0 is off when r = 2, neuron 11 when r = 5, neuron 10 when r = 6. Neuron 12 is always on. Neurons 13 and 14 are off only at the first positive number." About 50 fitted numbers give 94.0% against 95.4%.
- "What the memory represents: mainly a clock of the largest's age, which sets the on/off pattern, with kept votes in the same rotating code on top." (`parameters/memory_register/README.md`)
- Mode ablation on the real network (`twin_tests/README.md`): "The 127-degree mode is the only one whose removal hurts mainly lists where the second-largest comes after the largest"; "The 38-degree mode serves mainly lists where the second-largest comes first".

Whole network: "With no biases and ReLUs, each step is exactly linear once the on/off pattern is fixed, so the network is a switching linear system". "In the most common on/off pattern at steps 4–8, ARC's dynamics have rotating pairs at ±43°, ±83°, ±125° and 180° ... shrinking to 0.84–0.95 per step." "The output weights for position k+1 equal those for position k pushed one step through the dynamics (cosine 0.97; plain neighbouring rows 0.56)." "The output layer sees it [the 43° rotation] less than a tenth as well as a random direction."

Cancellation: "ARC's decisions are small leftovers of large parts: per neuron the parts sum to about 30 times the decision on typical lists and 100-300 times at near ties". Formulas for all 16 neurons at once leave 51% accuracy, "because each must be right to a few percent".

Mistakes: "61% name the third, 15% the largest, 18% are lists with at most one positive number, 6% the fourth or lower."

Hand-built networks: v5 (16 neurons, 98.97%, a different algorithm); v18 "the twin" (16 neurons, 96.0%); v23 (16 neurons, 95.8%). "Patching v18's groups into ARC leaves 85% (store and detector), 75% (late gate), 49% (keep gate), 26% (memory) and 16-19% (hiding)."

Third-party SPD run (`parameters/spd/README.md`, a ~200-line reimplementation of Stochastic Parameter Decomposition): "SPD from a random start uses about 30 of 48 recurrent pieces per step at KL 0.022. Its runs match at a median 0.4, and it needs about five times as many numbers as the weights." "SPD started from the neurons keeps them and gets lower loss on every term (KL 0.007, 86 uses against 101). It finds the late gate's timing and no multi-neuron mechanism."

Open questions they list (`FINDINGS.md` §3) include: "How does ARC's memory recirculate without the overtaken largest's wave coming back?", "What fully decides neuron 15's switching?", "Why does the keep gate cut at about 0.75 of the second-largest rather than nearer 1?", "Do missing biases force some memory neurons to hover near zero?", "Is any description of near-tie decisions shorter than the 432 weights?"

### 3.4 Josh Snider: two equal-speed clocks and a write pointer

- Post, 2026-10-26: https://www.lesswrong.com/posts/sAhFp6gYoyw3xeCaF/brute-force-progress-on-algzoo-s-432-param-interpretability
- Repository, head `464f907` (2026-10-26), `REPORT.md` written by Claude: https://github.com/jsnider3/algzoo-432-interp
- Issue on ARC's repo, 2026-10-19: https://github.com/alignment-research-center/alg-zoo/issues/1

`REPORT.md` §1: "The network runs an **online, time-invariant 2nd-max tracker**. It never stores *where* the max and 2nd max are. It stores *how long ago* they appeared, as the **phases of two damped rotating clocks that turn at the same speed (about 43° per step)**. Each clock's amplitude scales with the value it tracks." Update rule: "if the input is a new max, it restarts the max clock and hands the old max clock's phase to the 2nd-max clock; if it is a new 2nd max, it restarts the 2nd-max clock; otherwise both clocks just keep rotating."

§2.3, eigenvalues of "the usually-active neurons (13 of 16)":

| mode | size, angle | role he assigns |
|---|---|---|
| $0.70 \pm 0.65i$ | 0.951, 42.7° | "**max clock**: phase advances 41.5° ± 4.1° per step of max age, concentration 0.94" |
| $0.54 \pm 0.50i$ | 0.735, 43.0° | "**2nd-max clock**: phase advances 45.3° ± 6.0° per step of 2nd-max age, concentration 0.80" |
| $0.11 \pm 0.93i$ | 83.3° | "about the 2nd harmonic" |
| $-0.48 \pm 0.69i$ | 124.8° | "about the 3rd harmonic" |
| $-0.93$ | 180° | "about the 4th harmonic" |
| 1.01, 0.97 | real | "value registers (M and S scale)" |

§2.2, comparator thresholds: "almost exactly a·M + b·S with a + b ≈ 1 (fit quality R² 0.94–0.999; fitted at one step, t = 8...)". n6: threshold $\approx S$ when "the max arrived on the previous step"; n1: when "the max is 1 step old (partly at 2 steps)"; n8: when "the max is 0–3 steps old", "≈M from 5 steps old"; n7 and n4 test against $\approx M$, "drifting to about (0.6–0.75)·M + (0.2–0.3)·S when the max is 4–7 steps old".

§2.4, causal tests: multiplying the 2nd-max clock and its harmonics by $\lambda^k$ moves the answer by $-2$ positions on 78% of inputs ($k = 2$) and by $-1$ on 65% ($k = 1$). "**Rotating the max clock alone changes only 5% of outputs.** So the max clock drives the updates, and the readout relies on the 2nd-max clock." Mean-ablating modes: 2nd-max clock 47%, 83° harmonic 64%, 125° harmonic 65%, max clock 91%.

§7.9, write pointer: "A sparse **write-pointer neuron (n5)** fires one step after each new max or new 2nd max." It fires "the step after a new max: 94–100% of the time; the step after a new 2nd max: 70–97%; otherwise: 5–8%." "From the weights alone, Wo·Whh^a·e₅ decodes to [8, 7, …, 0]: every age correct" (own check: $\arg\max_i (W_{out} W_{hh}^a e_5)_i = 8, 7, \dots, 0$ for $a = 0..8$). "Forcing an n5 pulse moves the answer to the marked position in **88–96%** of runs with no later updates". "So the "clocks" of section 2 are the eigenmode breakdown of n5's impulse response."

Why n5 fires: pre-activation per unit $M$ is $+9.6$ after a new max, $+2.9$ after a new 2nd max, $-21.9$ otherwise; "n5 = ReLU(steady clock drive − comparator activity + new-max bonus). It fires exactly when the comparators were silent on the previous step."

Hand-off from max to 2nd max (new max at step 4 against a matched new-2nd event): $\Delta$logit at the event slot and at the old max's slot are $-37.1$ and $+24.4$ from the comparators, $-10.5$ and $+5.3$ from n4 and n9, $+30.4$ and $-18.0$ from the clock neurons; total $-17.9$ and $+13.1$. "The handoff is carried by the **comparators**, which are all silent when x > M, and the **max markers**."

Readout: "it **decodes the pointer's path**, and it **cancels the large M-scaled components**"; refitting $W_{out}$ by softmax regression on the trained recurrence gives cosine 0.998 to the trained readout (accuracy 0.9560 against 0.9530).

Max suppression: "raising the max's value **lowers its own logit**" (gradient $-20$ for this model). "Ordinary values are written through the comparators as ReLU(threshold − x), which carries −x. A new max switches those comparators off and is written instead through the new-max detector, as ReLU(x − old max), which has the opposite sign."

Errors (§3): "S < 0 (2nd max is negative) | 1.1% | **77%** | 18%" of errors; $M - S < 0.05M$: 20%; $S - R3 < 0.05M$: 53%; clean inputs err 0.23%.

His stated caveat (§8): "**The clock analysis is linear.** It uses the eigenmodes of the usually-active neurons, but those neurons do switch off at certain ages."

Other seeds: "A clean equal-speed 2nd-max clock with phase handoff appears only in the mystery model." Zoo seeds 0, 1, 4 use a "memory" strategy, seed 2 "online selection".

Training recipe (post): "The recipe for $M_{16, 10}$ can be found in Zihao Chen's notebook. It's Unif(−1, 1) inputs, Adam at 0.005, and 1,000 epochs of 100,000 sequences. It doesn't list the batch size, so I tried to reverse engineer that. With batch size of 64 or 128, the mystery model's accuracy is at the 97th percentile of networks trained this way." Issue #1 adds that the checkpoint in ARC's bucket is dated 2024-11-28 and that the model's median $|\text{margin}|$ (8.2) is larger than in all 1,024 networks he trained with the zoo recipe.

### 3.5 Skyway1111: "The runner-up"

- https://github.com/Skyway1111/RNN432, head `166b504` (2026-10-19); site https://skyway1111.github.io/RNN432/ (live; the site does not number neurons, the `research/` notes do).
- README: "This account does not meet that bar. Every neuron has a role that predicts what silencing or nudging it does, but near a tie the decision is a cascade of ReLU gates retuning one another, and predicting it costs about as much as running the network."

`research/findings/circuit.md`:
- "A six-neuron core (running max, increment, four phase-wheel comparators) drives ten satellites; the satellites carry the decision." Ablations: zeroing the satellite→core weights gives 0.508; zeroing the satellite readout 0.112; core only 0.103.
- "The core block of W_hh alone (n2, n7, n6, n1, n8): eigenvalues | 0.96 at 0°; 0.93 at ±42.1°" (own check: $0.9585$ at 0°, $0.9330$ at $\pm 42.09°$).
- "So the clock is the comparators. Their block of W_hh turns the state 42° per step, and the satellites carry the turning state as rectified place cells without changing its angle. Eight steps of 45° close the circle, so the wheel holds positions 0–7, equally spaced. Position 8 would land on position 0's phase, and position 9 arrives before the core's gates settle, so each has its own neurons: n5 and n9 for position 8, n4 for position 9".
- Readout plane (top two singular directions of $W_{oh}$): share of $\lVert W_{oh} \rVert^2$ 0.78; "readout angle step per position, 0 → 7; then 7 → 9 | −46, −40, −52, −42, −41, −45, −53; +17, +9" (own check with centred rows: $-45, -39, -51, -42, -41, -45, -54; +16, +8$).
- On Gaussian inputs the plane is not the decision: "Agreement with the full model when W_oh is truncated to its top 1 / 2 / 3 / 4 singular directions, example | 0.20 / 0.37 / 0.45 / 0.46".
- "The cross-entropy-optimal readout of h_10, fitted without W_oh, against W_oh with rows centered: per-row cosine; R² | ≥ 0.994; 0.985".
- Open: "Why training chose a 45° clock with two dedicated positions, where s0, s3 and s4 turn about 165° per step".

`findings/votes.md`:
- "Per route the logits are exact sums of votes: each input votes for its own slot by its value, except the max, which votes against itself." Mean votes per unit: the max $-22$ for its own position, the 2nd max $+35$, the 3rd $+41$. "So l_q − l_r ≈ 53 S − 57 T + 8 M on average ... and l_q − l_p ≈ 31 M − 10 S − 16 T".
- "Running max n2: coefficients on M, S, T | 12.6, 4.1, 1.2".
- "So the satellites and comparators are place cells for when the max and the 2nd max arrived, each tuned to an age, read by a linear decoder."
- "The bumps are impulse responses": $R^2$ between a neuron's impulse response and its measured coefficient on $M$ has median 0.96 (n3 0.38, n13 0.76).
- "The two kinds of second max, a demoted max (q before p) and a runner-up (q after p), use the same neurons with opposite signs".

`findings/neurons.md`: per-neuron table. "n8 | below the runner-up so far". Timers "tile the runner-up's positions 0–8, a relay n12 → n11 → n10 → n15 among them, and the comparator n7 covers 9". Each timer "points to" positions: n14→0, n15→1, 2, n10→3, 2, n11→4, n12→5, 6, n0→7, n5→8. Open: "Why each timer holds the record best at the ages it does: the reading angle of its weights on the comparators' rotating mode does not predict it (the comparators themselves sit about 80° apart on that mode, two steps of the clock)."

`findings/linearity.md`: "Distinct 160-bit activation patterns in 2^20 inputs | 554,894"; "Margin l_q − l_r, mean coefficient on M, S, T | +8, +53, −57 (sd 36, 65, 55)".

### 3.6 David Khachaturov: "Inside a second-argmax network"

- Article, 2026-10-26: https://egoistmachines.com/research/inside-second-argmax/ (with `technical.html` and `pseudocode.html`); repository https://github.com/Egoist-Machines/inside-second-argmax, head `0d1e584`.

Summary: he rewrites the network exactly as 16 stored coordinates "c, H, L, tau, w16, w87, a, b, z0, z3, z10, z11, z12, z13, z14, z15" and checks that the program reproduces all 432 coefficients (maximum score error $5.6 \times 10^{-12}$ on 16,384 sequences). Two local claims are proved in rational arithmetic. He says: "Several groups of learned weights still appear in the program as filters whose effects depend on input history. We still need to explain what these filters accomplish, why other rank errors occur, and how the circuit determines overall accuracy."

Mechanistic claims:
- Groups: carriers $P = (0, 3, 10, 11, 12, 13, 14, 15)$; comparisons 1, 6, 7, 8; carry/record 2, 4; candidate/relay 5, 9.
- "Neurons 1, 6, 7 and 8 compare each arriving input with learned thresholds... They draw on neuron 2, which carries an approximation to the maximum, and neuron 4, which responds by varying amounts when an input crosses a learned record threshold."
- "Neuron 5 forms a candidate for the preceding input. Its amplitude is proportional to max(0, τ − x), where τ is the stored cutoff and x the arriving input." "The candidate writes one update later by inhibiting eight carriers. Their learned readout can turn this reduction in activity into a relative score advantage for the tagged position."
- Carrier update, in units of the cost of a candidate write: "available = relu(background); spent = min(available, old_candidate); carrier_next = available - spent".
- Thresholds as mixed fields: "T4 = -W4 · h / u4", "R7 = W7 · h / abs(u7)", "H = (T4+R7)/2; L = R7-T4", "tau = W5 · h / abs(u5)"; then "d_next=relu(x-T4) and q_next=relu(R7-x). Those thresholds can differ from the true ranked input values."
- Record switch example: after $[1,0,0,0,0,0,0,0.5,0]$, a final input 0.25 selects position 7 and a final input 2 selects position 0. The comparators' contribution to (score 0 − score 7) is $-18.266$ for 0.25 and 0 for 2, "which closes all four comparisons".
- Supplied-write law: keeping a positive candidate (neuron 5) against erasing it, if carriers 0, 14, 15 have non-negative pre-activations, "the tagged position gains against all nine rivals, the admission base c rises, and the cutoff τ falls." "A write can therefore strengthen the wrong candidate."
- False record: "After inputs `[1,0]`, the learned record threshold is about 0.979. An offer near `511/512` activates record neuron 4 even though it is below the existing maximum, 1." For $[1,0,z,0,\dots,0]$ with $z$ within $1/16384$ of $511/512$ "the network selects maximum position 0 instead of runner-up position 2." (Own check: threshold 0.9789 after $[1, 0]$, 0.9880 after $[1]$; at $z = 511/512$, $h_4 = 0.195$ and the output is 0.) Removing the old record signal's "contrast" part into the comparators, $\text{contrast}_i = W_{i4} - |u_i|\,W_{74}/|u_7|$, repairs the family.

This matches the `false_new_max.png` result already in this workspace (threshold sag after a record).

### 3.7 Ali Panahi: "damped discrete Fourier memory", separable position code

- https://github.com/panaali/beyond-sampling, head `dc6a53e` (2026-10-21, branch `ccr-6a61f25c-wn80ii`); read `m16/FINDINGS.md`, `README.md`, `m16/approaches/mechanism_independent_review/README.md`.

`m16/FINDINGS.md`:
- "**The algorithm is right; the errors come from magnitudes.** Replace every input by the expected Gaussian order statistic for its rank (same order, typical spacing) and the model scores **99.96%**."
- "With the gates that stay on after a record (13 of 16 neurons), the recurrence's eigenvalues form **an approximate harmonic series**: a damped discrete Fourier memory." Table: integrator 1.010 ("n2 (running max), plus a share of every clock"); slow memory 0.973 (n0, n10–n12, n14); fundamental 0.951, period 8.42 (n0, n10, n11, n12, n14, n15); 2nd harmonic 0.937, period 4.32 (n3, n13, n15); 3rd harmonic 0.837, period 2.89 (n3); Nyquist 0.928; "current input only | 0.00 | **n1, n6, n7, n8** (x9 comparators)".
- His own caveat: "this gate set is only the *most common* one, covering 16% of states at steps 6-10... only 2 of 5 independently trained seeds show a similar series."
- "**The clock: PASSED.** The phase of the fundamental mode steps by **-42.7 degrees per position of the max**... One complex coordinate of the hidden state is a clock that says how long ago the max arrived."
- "**Position code: mostly PASSED.** The hidden state is close to a *sum of two position vectors*: h ~ m2 U[p2] + (m1 - m2) V[p1]." That is $h \approx m_2\,U[p_2] + (m_1 - m_2)\,V[p_1]$. "This separable code alone scores 90-91%, against the model's 95-96%, when x9 is below both."
- Group ablation at the final state: clocks (0, 10, 11, 12, 14, 15) give 0.127 / 0.437 / 0.229 for the cases "$x_9$ is max / is 2nd / below both"; comparators give 0.840 / 0.000 / 0.085.
- "**Gating is set by record history.** At the last step, each neuron's on/off state is 92-100% predictable from the sequence of record / new-2nd-place events alone."
- Failed hypotheses he keeps: "*p2 is a second clean phasor in the fundamental mode*: residual phases are irregular. p2 is spread across harmonics and gating." "*Clock neurons are linear filters of per-step features*: R^2 0.72-0.93, but the replacement scores only **32%**. The gating carries essential information."
- Near-tie anatomy: "The m2 vs m3 comparator is almost balanced: gain ~78 on each. The gain ratio is 0.982 on correct cases and 0.999 on flips."
- "One-sentence labels for all 16 neurons rebuild only 54% accuracy".

### 3.8 Kishore Kumar Mariappan: "Causal Geometry of an Alg-Zoo RNN"

- Preprint dated 2026-03-08, `main.pdf` in https://github.com/datasci3nce/causal-geometry-algzoo (head `ebf562b`). Not on arXiv.
- Claims: "the network's computation is governed by a continuous, distributed **invariant manifold** (the "Logic Ring")"; ablation finds "13 out of 16 neurons" necessary; hidden states "form a continuous 1D $H_1$ manifold tracking `current_max`" ("Pearson ρ = 0.86 for Neuron 2"); "We derive a mechanistic error bound of **3.44%** from the manifold's geometric resolution limit", with "a worst-case threshold of δ ≈ 0.0227" found by binary search; in the most frequent linear region "15 have magnitude strictly less than 1" and one eigenvalue has "|λ| ≈ 1.023".
- Assessment (inference): little weight-level content. His PCA figure shows a wedge opening from one point, which is what positive homogeneity gives (states scale with the running max), not evidence of a ring. The 3.44% figure uses a threshold measured on samples and Monte Carlo integration.

### 3.9 mogar: neuron dictionary for $M_{16,10}$

- https://github.com/mogar/algzoo_interp, head `5566526` (2026-10-22). The main target is the 32×10 zoo model (§4).
- `results/example_16x10/dictionary/dictionary.md`, best single-feature fits on $\mathcal N(0,1)$ inputs ($\mathrm{max0} = \max(0, x_0..x_t)$): n2 `max0_prev` ($R^2$ 0.974, slope 17.19); n4 `max0-max0_prev` (0.994, slope 10.31; event agreement 0.990); n7 `max0-x_t` (0.986, slope 12.76); n6 `max0_excl(t-1)-x_t` (0.972, slope 10.57); n1 `max0_excl(t-2)-x_t` (0.939, slope 9.98); n8 `relu(2nd0_prev-x_t)` (0.886, slope 13.03); n5 `relu(x_{t-1})` (0.855, slope 9.94); n9 `relu(x_{t-1})` (0.683, slope 1.20). Neurons 0, 3, 10–15 are labelled "unexplained".
- "The blog's 16x10 model looks like the handcrafted model: no detector veto, no clock, a low reset share (0.17), and a near-exact `max0 − x_t` neuron" and "The blog's 16x10 model is close to a compressed version of the whole handcrafted scheme". Here "clock" is their strict test: "neurons always on at some lag ≥ 2 of the max and always off at another".
- A fit of each neuron on (step, lag of max, lag of 2nd max) cells, linear in the two values and the current input, reaches median $R^2$ "0.97, 15 of 16 neurons at 0.9" for this model.

### 3.10 kas.eth (krandder): noisy fine-tunes

- https://github.com/krandder/arc-mean-estimation, `lanes/algzoo-noisy/report/algzoo-noisy-2026-10-29.md`, head `be67c6f`.
- Activation noise at fraction $\sigma$ of each neuron's scale: only a fine-tune at $\sigma = 0.001$ keeps the loss. "The sigma = 0.001 model ... is ARC's network plus negligible noise: same fragility, same structure." "the max-tracking neurons {1, 2, 4, 6, 7}, plus 8, are the algorithmic part (R^2 >= 0.99)"; "neuron 7 alone carries 39-52% of the noise that reaches the decision"; "the question needs up to 13 of 16 directions per step (9-13 from step 4 on)".

### 3.11 Minor or unverifiable items

- **Marcos Brito**, https://github.com/dinamicoplus/algzoo-legibility (one commit, 2026-10-16): scripts only. README: "A quick, informal experiment on ARC's 432-parameter AlgZoo model". No written results.
- **WFJKK**, https://github.com/WFJKK/Mech-Interp-Agent-level1 (2026-11-19): an agent writes a cheaper program that predicts the network's output. For "ARC's challenge network" the README reports fidelity 99.52% at 0.134 of a forward pass. The program itself and any mechanism description are not in the repository.
- **"phoenix"** (LessWrong): a private tool "ATLAS". Comment of 2026-08-05 (https://www.lesswrong.com/posts/vLFh8HP3hyNy9MCwe/returning-to-arc, comment `LGpDvgKEXFoXB8xxo`), in an LLM-written summary: "read from the weights it is a running-2nd-argmax by elimination, and its accuracy is an exact sum over rank-tie boundaries — it errs only where two consecutively-ranked inputs are too close to separate. The certificate closes at 0.953019 == MC 0.953019 with residual 0e+00". And: "The repo is currently private". Cannot be checked. The statement that the model errs "only" at rank ties conflicts with the measured 18% of errors on lists with a non-positive 2nd max (§3.3, §3.4).
- **emanuelr**, comment on Snider's post, 2026-11-02: "For this model it seems like the representations vary a lot depending on the seeds, so there's probably no "human interpretation" as for the smaller ones."
- **`aneeshchopada-lab/AlgZoo-M16-10-analysis`** (2026-10-20): a copy of ARC's `alg_zoo` package files; no analysis.
- Forks `poonai/alg-zoo`, `ChamodKalupahana/alg-zoo` (February 2026): small explorations of other models; nothing on $M_{16,10}$.
- Mentions without analysis: Leo Gao's shortform (2026-01-27, 2026-07-22: "fully interpreting the algzoo models, possibly using circuit sparsity techniques"); Logan Riggs, "Mech Interp is a Verifiable Task If..." (2026-09-21).

## 4. Other AlgZoo models (context)

- **mogar, 32×10 zoo model** (`results/2nd_argmax_32x10/README.md`): event detectors (a neuron equal to the increase of the positive running max; a "new top-2" neuron), stored values, "Each new max zeroes about ten neurons, which then run a fixed on/off sequence that encodes how long ago the max arrived", a shift register, and readout vetoes. "The two 16x10 models use two different solutions". Also a complete exact account of the 4×3 model.
- **kiankyars/alg-zoo** (`research/README.md`, 2026-10-31): exact accuracies by region enumeration for all AlgZoo RNN classifiers with at most 5 inputs; deterministic estimators that beat sampling for 3-input models. It states that $M_{16,10}$ "is still open".
- **jc1122/mechanistic-interpretability-circuit-atlas** (`entries/algzoo-2nd-argmax.md`, 2026-11-15): 280 zoo networks (widths 2–32, lengths 2–10, 5 seeds); after its own audit "16 proven / 257 read / 7 unexplained". It does not analyse the $M_{16,10}$ example.
- **Xabier Ariznabarreta**, "Taking apart a 32-number network", 2026-11-08, https://xabier.me/posts/2026-11-08-taking-apart-a-32-number-network (EA Forum: https://forum.effectivealtruism.org/posts/boaXwxTaB2pYuyGmj/taking-apart-a-32-number-network-1), code https://github.com/Xabilimon1/leeuwenhoek. Subject: the 32-parameter **argmedian** RNN (4 neurons, 3 inputs). "The four neurons are two mirror pairs: one pair fires when the new number is higher than the last, the other when it is lower." Each pair carries "a large **comparison** channel whose sign says up or down, and a small **memory** channel that rebuilds the previous number for the next comparison." Relevant as another bias-free ReLU RNN that stores a value as the difference of two rectified units.
- **mi-for-the-rust-of-us/candle-mi** (`src/stoicheia/`): Rust tooling for AlgZoo models; no analysis of $M_{16,10}$.

## 5. Related literature (not about AlgZoo)

Verified only at abstract level.
- Karuvally, Sejnowski, Siegelmann, "Hidden Traveling Waves bind Working Memory Variables in Recurrent Neural Networks", ICML 2024, arXiv:2402.10163. Working-memory variables are stored as travelling waves, not in fixed registers; the linear case gives a shift matrix plus a low-rank matrix. DraggonZ cites Karuvally et al. for the shift-register test.
- Karuvally, Delmastro, Siegelmann, "Episodic Memory Theory for the Mechanistic Interpretation of Recurrent Neural Networks", arXiv:2310.02430. A variable-binding circuit that trained RNNs converge to, and a privileged basis for reading it.
- Brenner, Hemmer, Monfared, Durstewitz, "Almost-Linear RNNs Yield Highly Interpretable Symbolic Codes in Dynamical Systems Reconstruction", NeurIPS 2024, arXiv:2410.14240. RNNs with few ReLU units as piecewise-linear systems with a symbolic code of linear regions. This is the "switching linear system" view that DraggonZ, Keane and Skyway1111 arrive at.
- Eisenmann, Brändle, Monfared, Durstewitz, "Detecting Invariant Manifolds in ReLU-Based RNNs", ICLR 2026, arXiv:2510.03814. Exact stable and unstable manifolds of piecewise-linear RNNs. Cited by Mariappan.
- From DraggonZ's literature review (not re-read by me): Sussillo and Barak 2013 (fixed-point analysis; not applicable since the network is input-driven), Gross et al. 2024 (arXiv:2406.11779, compact proofs for a max-of-$K$ transformer; "compounding structureless errors"), Kaufman et al. 2014 (output-null activity, the closest analogue of a clock that the readout cannot see), Nanda et al. 2023 and Zhong et al. 2023 (rotation codes in modular addition; different seeds learn different algorithms).
- I found no paper on a tiny ReLU RNN that tracks a running maximum with a rotating age code (§9).

## 6. Own checks of third-party numbers

Scripts `thirdparty_check_claims.py`, `thirdparty_check_comparator_fit.py`, `thirdparty_check_modes13.py` in this folder. All claims checked reproduce.

| Claim (source) | Own result |
|---|---|
| Full $W_{hh}$ eigenvalues $-1.274$, $1.167$, $0.435 \pm 1.027i$ ($\lvert\lambda\rvert = 1.115$), ... (Keane doc 49) | identical; the complex pair is at 67.0°, period 5.37 |
| 13 units without 4, 5, 9: $1.01$, $0.97$, $0.951 \angle 42.7°$, $0.937 \angle 83.3°$, $-0.93$, $0.837 \angle 124.8°$, $0.735 \angle 43.0°$ (Snider, Panahi, DraggonZ) | $1.0099$, $0.9735$, $0.9509 \angle 42.75°$, $0.9370 \angle 83.28°$, $-0.9283$, $0.8373 \angle 124.78°$, $0.7350 \angle 42.99°$, $0.2310$, $0.0010$ |
| Comparator block $\{1,6,7,8\}$: one pair at 43°, sizes 0.937 and 0.043 (DraggonZ) | $0.9375 \angle 42.56°$; $0.0432$ (pair $0.041 \pm 0.013i$) |
| Core block $\{2,7,6,1,8\}$: 0.96 at 0°, 0.93 at ±42.1° (Skyway1111; Keane $0.69 \pm 0.63i$) | $0.9585$; $0.9330 \angle 42.09°$ ($0.692 \pm 0.625i$); $0.084$; $0.005$ |
| Memory block $\{0,3,10..15\}$: $1.025$; $0.815 \angle 38.3°$; $0.996 \angle 83.5°$; $0.858 \angle 126.6°$; $-0.918$ (DraggonZ) | identical to 3 digits |
| Memory block with n0 off: largest real mode 0.22, 36° at 0.74, 88° at 0.91; with n15 off: 61° and 115° at about 0.70 (DraggonZ) | $0.224$; $0.739 \angle 35.6°$; $0.913 \angle 87.6°$; and $0.701 \angle 61.3°$, $0.696 \angle 115.2°$ |
| Block $\{10,11,12\}$: $-0.248$, $0.529 \pm 0.297i$ (Keane doc 16) | identical |
| $h_2' = 0.97 h_2 + 1.73 h_4$; $h_4 = 10.16 a - 0.49 h_2$; $17.58$; trace 0.12 (Keane) | $0.9688$, $1.7307$, $10.157$, $-0.4923$; $17.579$; $0.1168$ |
| Median $h_2 / $ running max $= 17.6$ (Keane, DraggonZ) | 17.5–18.0 for $t = 1..9$ (200k $\mathcal N(0,1)$ lists) |
| $\sum_{j \in \{1,2,6,7,8\}} W_{hh}[n,j]\,W_{ih}[j] \approx 0$ (Goldowsky-Dill) | $0.016$, $0.017$, $0.141$, $-0.005$, $0.146$ for $n = 1, 2, 6, 7, 8$ (single terms are of size 5–14) |
| Blindness: all rows except the gates are orthogonal to the comparators' input weights (DraggonZ) | cosines within $\pm 0.04$ except n5 $-0.74$, n9 $-0.69$ |
| Store→comparator weights $= -0.048$ to $-0.051 \times$ input weights (DraggonZ) | $-0.0477$, $-0.0487$, $-0.0512$, $-0.0508$ |
| Impulse response of n7 (Keane doc 20); zero-crossing ages (Keane doc 18, DraggonZ) | identical; n0, n14 at age 2; n15 at 3–4; n11 at 5; n10 at 6 |
| After an impulse: n4 = 10.2, then n5 = 12.2 and n9 = 1.5 one step later (DraggonZ) | $10.16$; $12.21$; $1.49$ |
| Countdown readouts $[4,9,8,7,6,5,4,3,2,1]$ and $[4,6,9,8,0,9,9,9,9,9]$ (Keane doc 32) | identical |
| Record threshold 0.979 after $[1,0]$; false record at $z = 511/512$ (Khachaturov) | $0.9789$; $h_4 = 0.195$, output 0 |
| $\arg\max W_{out} W_{hh}^a e_5 = 8, 7, \dots, 0$ (Snider) | identical for $a = 0..8$ |
| Readout shift cosines 0.97, 0.99, 0.99, 0.99, 0.88, 0.99, 0.96, then 0.41, 0.31 (DraggonZ) | 0.97, 0.99, 0.99, 0.99, 0.88, 0.99, 0.96, 0.40, 0.29 |
| Readout plane share 0.78; angle steps $-46, -40, -52, -42, -41, -45, -53; +17, +9$ (Skyway1111) | 0.78; $-45, -39, -51, -42, -41, -45, -54; +16, +8$ |
| Goldowsky-Dill's coefficient figure | reproduced (table in §3.1) |
| Comparator 4-vector: 89.8%, 5.0%, 3.2%, 2.0% of variance (DraggonZ) | 92%, 4%, 3%, 2% (steps 1–9, 200k lists; sample differs) |

One further own check, bearing on a disagreement in §8 (`thirdparty_check_modes13.py`). For the 13-unit matrix, the squared weight of each eigenvector on (comparators / unit 2 / memory units):

| mode | right eigenvector | left eigenvector | readout visibility |
|---|---|---|---|
| $1.0099$ | 0.15 / 0.08 / 0.77 | 0.04 / 0.85 / 0.11 | 0.03 |
| $0.9735$ | 0.08 / 0.07 / 0.85 | 0.51 / 0.43 / 0.07 | 0.13 |
| $0.9509 \angle 42.75°$ | 0.11 / 0.01 / 0.87 | 0.97 / 0.00 / 0.03 | 0.03 |
| $0.9370 \angle 83.28°$ | 0.01 / 0.00 / 0.99 | 0.61 / 0.00 / 0.39 | 0.18 |
| $0.7350 \angle 42.99°$ | 0.06 / 0.00 / 0.94 | 0.92 / 0.00 / 0.08 | 0.35 |

"Readout visibility" is $\lVert W_c r \rVert / (\lVert W_c \rVert_2 \lVert r \rVert)$ with $W_c$ the row-centred readout. The shares depend on the per-neuron scale gauge, so read them qualitatively: both 43° modes are driven almost only by the comparators (left eigenvectors) and appear mostly in the memory units (right eigenvectors). The main clock is nearly invisible to the readout; the 0.735 mode is the most visible one.

## 7. What third parties say that bears on the "off-centre circle with an ersatz bias" hypothesis

The hypothesis: the age of the current max is stored on a circle whose directions are fixed but whose centre and radius depend on the max and 2nd-max values, because (a) the circle must be off-centre to survive the ReLUs, (b) rotating about an off-centre point needs a bias, (c) with no bias the network builds one from stored values.

Statements that support it, verbatim:
1. DraggonZ, `faithful_twin/LEDGER.md` (a design argument written while building a hand-made twin, not a measurement on ARC's network): "**What holds the wave must itself be a comparator.** The wave has to be wiped when a record arrives. A ReLU network without biases can only wipe a signed 2-D state by clipping it, so the wave must ride on a positive offset that drops to zero at a record: the comparator level gc (m - x). That is why the clocks exist." And: "a bias-free ReLU network can only wipe a signed 2-D state by clipping it on a comparator level. So both sets need three neurons (level + two wave directions)."
2. DraggonZ, `robust_twin/DESIGN.md`: "**A small wave on a large level.** In the hiding group, |wave| / |level| has median 0.24 in the trained network ... A larger ratio means the level no longer keeps the neurons positive, so they clip: that is the trained network's partial switching."
3. DraggonZ, `gao/README.md`: "Cancellers' 4-vector: principal directions carry 89.8%, 5.0%, 3.2%, 2.0% of variance (one shared level plus a small rotating wave)." "Picture: a positive number lowers the shared level; while the wave is still large (soon after a new largest), the neurons at its low point dip below zero."
4. DraggonZ, `hiding/README.md` (quoted in §3.3): "the level is supplied afresh every step from the store, and only the record's jump starts a lasting rotation." And `robust_twin/DESIGN.md`: "Their wave comes 81% from themselves and 19% from the store and detector, and the store's push points along the wave... So the trained hiding wave is a damped rotation (shrinking 6% per step) continually re-driven by the store, not a free-running rotation started once by a record".
5. DraggonZ, `superposition/norule/README.md`: "With no biases, a neuron that reads a rotating direction swings below zero unless it also carries the largest so far, and some neurons are left hovering near zero." Their test over 47 networks: "Across all memory neurons, the more a neuron reads the rotation and the less it reads the largest so far, the closer it hovers to zero (rank correlations 0.38 and 0.42)." And: "Not tested. Whether the missing biases force hovering. That needs networks trained with biases".
6. DraggonZ, `superposition/v23/README.md`: "ARC stores its memory spread across neurons on a low shared level (its rotating part about a quarter of the level) and its hiding neurons have gains of about 10, so its neurons switch off often (partial switching)." And `NOTE.md` item 10: "the memory holds each earlier vote next to a larger negative imprint of the largest, and the store holds the largest and discounts later arrivals by it".
7. Panahi: "h ~ m2 U[p2] + (m1 - m2) V[p1]". The max-position pattern scales with the gap $m_1 - m_2$, the 2nd-max pattern with $m_2$.
8. Snider: "Each clock's amplitude scales with the value it tracks." Comparator thresholds are "a·M + b·S with a + b ≈ 1".
9. Goldowsky-Dill's fit and my reproduction of it (§3.1): $c_{n,t} \approx |W_{ih}[n]|\,[S^+ + \alpha_n(\delta)(M - S^+)]$.

What this adds up to (inference). For the comparators, the third-party fits agree on a level-plus-wave form in which both parts are set by stored values:

$$h_n \approx c_n\big[\underbrace{S^+ + \bar\alpha_n (M - S^+)}_{\text{level}} + \underbrace{(\alpha_n(\delta) - \bar\alpha_n)(M - S^+)}_{\text{wave}} - x_t\big]^+ .$$

The centre depends on $M$ and $S$; the radius is proportional to the gap $M - S^+$ (DraggonZ: "only the record's jump starts a lasting rotation"; the jump $M - P$ equals $M - S$ right after a record). The wave directions $\alpha_n(\delta) - \bar\alpha_n$ are fixed functions of age. So parts (a) and (c) are stated by DraggonZ in almost the same words, as a design argument and with a partial population test. For part (b): nobody derives the level as a substitute for a bias from the weights of the trained network; the nearest weight-level facts are that unit 2 feeds each comparator at $-0.05 \times$ its input weight (so the store alone sets the threshold $\approx 0.05 \cdot 17.6\,M \approx 0.88\,M$ before the other terms) and that this push lies in the block's fast-decaying directions, so it must be re-supplied at every step.

Statements that complicate it:
- The level is not a clean function of $M$ and $S$. DraggonZ's references use the previous largest $P$, move to the running second-largest through clipping "echoes", and need 648 fitted numbers to get within 1 point; a rule on $P$ and $M$ only gives 74.6%.
- The circle is not one plane for the whole state. The comparator block has one rotating pair (43°); the memory block has pairs at 38°, 84°, 127° and a 180° mode; Snider and Panahi see the 2nd max in a second 43° mode and the harmonics. Panahi: "*p2 is a second clean phasor in the fundamental mode*" failed.
- The rotation is gated: memory units switch off at fixed ages (0 at age 2, 11 at 5, 10 at 6), and without that switching accuracy is 44%. A circle in a fixed linear subspace describes only the most common gate pattern ("16% of states at steps 6-10", Panahi; "12% of steps", DraggonZ).
- DraggonZ, `parameters/memory_register`: "Snapping the angles to multiples of 42° costs KL 0.97"; angles must be right to about a quarter of a degree unless other weights are refitted.

## 8. Where the sources disagree

| Question | Positions | Resolution, if any |
|---|---|---|
| Rotation speed | Keane docs 32, 48: a "Primary oscillator" of period "5.4 steps", and "Rotates ~36° per timestep"; others: 42–43° per step, period 8.4 | Keane used the full $W_{hh}$ including units 4, 5, 9, which are normally off. With those removed the pair is $0.951 \angle 42.75°$ (own check; Panahi's review says the same). His two-impulse PCA angle of about 36° is a different measurement |
| Where the clock lives | Skyway1111: "the clock is the comparators"; Panahi: the fundamental is carried by n0, n10, n11, n12, n14, n15 and the comparators are a "current input only" mode; DraggonZ: the comparator block has its own 43° pair and the memory has a 38° pair | Own check (§6): in the 13-unit matrix both 43° modes are driven by the comparators (left eigenvectors) and expressed in the memory units (right eigenvectors) |
| Is the 2nd max's age a second clock? | Snider: yes, the 0.735 mode ("concentration 0.80"); Panahi: "residual phases are irregular"; DraggonZ: kept votes are "in the same rotating code on top" of the max's clock, phase spread 54–79° on one list | Open. This workspace's probe result (main-clock $R^2 = 0.44$ for the 2nd max's age) sides with "not a clean phasor" |
| Is there a clock at all? | mogar: "no clock" for this model, by a strict always-on/always-off criterion; everyone else: yes | Different definitions. Neuron 0 is off at age 2 on 100% of steps (DraggonZ; own check on the impulse) |
| What unit 5 is | Keane: handler for "2nd after argmax", promotes class 8; DraggonZ: keep gate with threshold ≈ 0.72–0.75 of the 2nd largest; Snider: write pointer; Khachaturov: candidate $\propto \max(0, \tau - x)$; mogar: $\mathrm{relu}(x_{t-1})$ | Compatible descriptions of one unit: it fires one step after a number that clears a cutoff, and its pulse is what later reads out as that number's position |
| What unit 9 is | Keane: "Comparator Integrator" / secondary relay for small-gap reversed pairs; DraggonZ: late gate, fires one step after a record, pushes down comparators 1 and 6 two steps after; Snider: max marker | Compatible; DraggonZ's is the most specific and has interventions |
| Antisymmetric offset $f(m) - f(s)$ (Keane) | Panahi's review: the antisymmetry is forced by the fit | Treat only the separability ($R^2 \approx 0.91$) as a finding |
| Does the readout read a 2-D phase? | Keane: phase wheel; Skyway1111: only on two-impulse inputs; on Gaussian inputs truncating $W_{oh}$ to 2 directions gives 0.37 agreement | Agreed by the later sources: the top plane is a clock for positions 0–7, but the decision uses all ten directions |
| How compact can the account be? | Everyone who tried reports that formulas reproducing each neuron to $R^2 > 0.95$ still lose many points when substituted (Goldowsky-Dill > 90% per neuron; DraggonZ 51% for all 16; Panahi 54% from one-sentence labels; Keane: swapping n7's $R^2 = 0.99$ fit alone costs 4 points) | Consistent with the colleague's result that a faithful account is not shorter than the weights |

## 9. Searches that returned nothing relevant, and what could not be fetched

Nothing relevant:
- arXiv listing search for "AlgZoo" (titles, abstracts and metadata, not full text): "Sorry, your query for all: AlgZoo produced no results." The arXiv API returned HTTP 429, so the HTML search was used.
- arXiv listing search for `"second argmax" OR "2nd argmax" RNN`: first 38 of 50 hits inspected, none about this task.
- Goodfire research index (https://www.goodfire.com/research, 47 entries on 2026-11-19): no entry on AlgZoo, ARC, second argmax, tiny RNNs or "reverse-engineered networks". "Neural Networks Are Already Clean" (2026-10-15, revised 2026-10-30): no mention of AlgZoo or the 432-parameter RNN in the two windows read (the last part of the page was truncated). "Neurons as an Interpretable Basis" (2026-11-10) and "Interpreting Language Model Parameters" (2026-05-05): none in the first 100k characters of each page; the rest was not read.
- Web searches for the two working-paper titles in `reverse-engineered-networks/.../original-papers/` ("How ARC's 432-parameter RNN finds the second-largest input...", "comparators, a recorder and a rotating tape"): no public copy found. Disclosure: these two queries sent the title phrases of the internal working papers to the search service. No names, e-mail addresses or paths were sent in any request.
- "parameter decomposition" / "VPD" / "SPD" with AlgZoo: no Goodfire or MATS publication. The only hit is DraggonZ's own SPD reimplementation (§3.3).
- LessWrong site search (posts and comments) for "AlgZoo", "2nd argmax", "second argmax", "432-parameter", "alg-zoo": only ARC's post, Snider's post and the comments listed above. The Alignment Forum shares the same database; the AlgZoo post has 10 comments in total, all read.
- GitHub repository search for `algzoo`, `alg-zoo`, `"2nd argmax"`, `"second argmax"`, `432-parameter`, `RNN432`, `second-argmax`, `2nd-argmax`, `M16,10`, `example_2nd_argmax`, `algzoo in:readme`: every relevant hit is listed above. ARC's repo has 7 forks (4 with their own commits, all inspected) and one issue (Snider's).
- Web searches for tiny ReLU RNNs with a running maximum and a rotating age code, and for "bias-free ReLU RNN emulating an offset with a stored magnitude": no paper found.
- EA Forum and Substack: only Ariznabarreta's argmedian post.

Could not be fetched or only partly read:
- X/Twitter: single posts by id only (Hilton's posts of 2026-01-26 and 2026-10-16). Replies and quote-posts could not be listed, so third-party threads on X may be missing.
- Google Scholar: not queried directly (no API access); web search surfaced no paper citing AlgZoo.
- "phoenix"/ATLAS repository: private.
- Keane's fork: docs 01–10, 12–15, 17, 19, 21, 23–26, 28–31, 33–36, 38–47 were not read in full; `estimate/FINDINGS.md` (139 KB) was read in part.
- DraggonZ's repository: about a dozen of several hundred notes read in full; interactive HTML explainers not rendered.
- Snider's `REPORT.md` (226 KB): §§1–7.3, 7.9–7.12, 8, 10 read; §§7.4–7.8 and 7.13–7.21 (training dynamics, estimators) skipped.
- Panahi's repository (18,686 files): only the files named in §3.7.
- ARC's own material (Chen's 2024 notes and Colab, Hilton's and Robinson's Drive notes, the IPAM talk) is out of scope here and was not fetched. Third parties quote Chen's notes as: "projecting out the negative input neurons (N1, N6, N7, N8) gives an argmax/max sub-circuit that accurately finds the max index" and "For each iteration t, if x_t > max_t (running max value), the old max logit gets a boost" (quoted in DraggonZ's `lit_review/README.md`; not verified by me).
