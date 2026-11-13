# AI interpred models

Interpretability work on small reverse-engineered networks. General preferences are in `~/.claude/CLAUDE.md`. Keep this file updated with project state, decisions, findings and gotchas.

## Hard rule: `reverse-engineered-networks/` is read-only
It holds a colleague's material: papers, code, weights and explainers for three case studies (`1-algzoo-second-argmax-432`, `2-tiny-algzoo-rnns`, `3-transformers-trained-on-python-generators`). Reading it is fine. Never edit, delete or add anything there, including `__pycache__`: use `sys.dont_write_bytecode = True` or `python -B` when importing its code. All new work goes in sibling folders at the workspace root, named after the case study.

## Layout
- `1-algzoo-second-argmax-432/`: our work on ARC's 432-parameter second-argmax RNN, $M_{16,10}$.
  - Top level: only folders, no loose figures.
  - `plots/`: weight, mode, synthesis and pre-activation figures (user moved them here, 2026-11-13). Their scripts are in `hide/` (`plot_weights*.py`, `plot_synthesis.py`, `preact_*.py`) and write to `plot_weights.PLOTS` = `<case study>/plots/`. In `plot_weights.py`, `HERE` is still the case-study folder (used for the weights path).
  - `plots/more_plots/`: extra figures that won't be used (`eigenvalues.png`, `unit_profiles.png`, `weights_original_order.png`). `plot_weights.py`'s `fig_eigen`, `fig_unit_profiles` and non-grouped `fig_overview` write here; `main()` doesn't call them.
  - **All code and data for this case study live in `1-algzoo-second-argmax-432/hide/`** (user decision, 2026-11-13). The analysis folders hold only figures and `report.md`. Each linear-probe analysis has its own subfolder, `hide/<analysis name>/`, because the four analyses share script names. In those scripts, `HIDE` is the script folder (CSVs, `.npz`), `ROOT = HIDE.parent.parent` is the case-study folder, `OUT = ROOT / HIDE.name` is where figures go, and `plot_weights.py` is imported from `ROOT / "hide"`.
  - `linear-probes-for-every-t/`: archived first analysis, with **separate probes for every t**: regression for the three largest values, classifiers for "steps since" each. It has `report.md`, R², accuracy, a scatter plot and weights at t = 9. The user asked to keep it; don't overwrite it.
  - `linear-probes/`: current analysis, **one probe per target shared by all t** (what the user wanted). Same targets and data. Scripts, CSVs and the probe weights (`value_probes.npz`, `age_probes.npz`) are in `hide/linear-probes/`.
  - `linear-probes-positive/` and `linear-probes-for-every-t-positive/`: the same two analyses, but states whose $k$-th max is $\le 0$ are left out of training and scoring for the $k$-th value and age. The filter is per state, so a sequence counts again once its $k$-th max turns positive.
  - `replica/`: retraining of the original from its notebook recipe, plus two bias variants. **Read `replica/README.md`** for the recipe, the sweep results and the file list.

## The second-argmax network
- $h_t = \mathrm{ReLU}(W_{ih} x_t + W_{hh} h_{t-1})$, $h_{-1} = 0$, $\ell = W_{out} h_9$. There are no biases, and the input has length 10. Parameters: $16 + 256 + 160 = 432$.
- Weights: `reverse-engineered-networks/1-algzoo-second-argmax-432/code/m16_10.pth` (also `m16_10_weights.npz`). `code/full_account.py` defines the colleague's 13-mode basis (`ANALYSIS`, `SYNTHESIS`, `ORDINARY`, `PULSE_UNITS`).
- Unit groups (from the colleague's paper): comparators 1, 6, 7, 8 | pulses 4, 5, 9 | wave units 0, 2, 3, 10–15. The shared unit order and colours are in `1-algzoo-second-argmax-432/hide/plot_weights.py` (`GROUPS`, `ORDER`, `UNIT_COLOR`).
- Input distributions: the colleague's analysis uses iid $\mathcal{N}(0,1)$. The original was trained on iid $\mathrm{Unif}(-1,1)$. Being bias-free, the network is scale-equivariant.
- Original accuracy: 93.78% on uniform inputs and 95.31% on Gaussian inputs ($2^{20}$-sample test sets).

## Findings so far
- Replica training (see `replica/README.md`): with batch size 64 the original is roughly a top-10% seed. Best of 32 seeds gives `replica_nobias` at 95.6% uniform. Adding biases trains worse (median about 75% vs 90.5%), probably because biases break positive homogeneity.
- Linear probes on $h_t$, separate per $t$ (`linear-probes-for-every-t/`; Gaussian inputs, 200k train / 50k test, one probe per target and $t$ plus one pooled over $t$). Targets use the window $x_0..x_t$ ("incl") or $x_0..x_{t-1}$ ("excl"). Steps since the $k$-th max is $t - p_k$, counted from the current step.
  - Values (affine least squares, R²): max 0.97–1.00; 2nd max 0.81–0.95; 3rd max 0.45–0.88.
  - Steps since (multinomial logistic regression, C = 10, accuracy): max 0.97–1.00; 2nd max about 0.95 (incl) and 0.78 (excl) at late $t$; 3rd max falls to about 0.24, against a 0.10 majority baseline at $t = 9$.
  - The user asked for ages to be probed only with classifiers, and the regression results for ages were removed. When they were regressed, R² was only 0.06–0.74, because the age is held as a clock phase rather than linearly.
  - Independent agents checked the targets (exact match) and spot-checked the R² and accuracy values (within 0.02). Weaker regularisation adds 0.01–0.03 to a few classifier cells.
  - Probe weights at $t=9$ (`probe_weights_t9.png`) are stable across seeds: value probes within 0.02, classifier median relative change 0.4%. The max rests on unit 2, plus unit 4 (record pulse) only when $x_9$ is included. The 2nd and 3rd max rest on unit 7 (negative) and unit 2. Unit 2 is large in all three value probes, so it acts as a general level. Exception: nearly separable cells such as "steps since 3rd, incl", age 0 on unit 4 (−19 to −32) have regularisation-set magnitudes; read only their sign.
- Linear probes on $h_t$, **one probe per target shared by all $t$** (`linear-probes/`; same data; the probe is not told $t$; scored per $t$ and over all $t$).
  - Max: $R^2$ of 0.97–0.98 from $t \ge 3$, so it has one readout. 2nd max: peaks at 0.86 ($t = 4$) and falls to 0.62–0.67 at $t = 9$ (a separate probe at $t = 9$ gets 0.85–0.89), so its direction drifts with $t$. 3rd max: no better than a shared regression on the true max and 2nd max (0.63 vs 0.60 over all $t$).
  - Ages: $a_1$ reaches 0.97–0.99 for $t \ge 4$ (shared clock code). $a_2$ reaches 0.84–0.85 at late $t$ (incl), against 0.96 for a separate probe at $t = 9$. $a_3$ is near its baseline.
  - `steps_classifier.py` takes about 4 min (6 pooled fits on about 2M states each). The probe weights are saved in `hide/linear-probes/*.npz`, and `scatter_values.py` and `probe_weights.py` read them.
- Positive-only variants (states with $k$-th max $\le 0$ excluded):
  - Shared probes improve a lot: max $R^2 \ge 0.99$ at every $t$; 2nd max 0.89–0.91 overall; 3rd max 0.65–0.68, now clearly above the true-order-statistic baseline (0.48–0.50). $a_2$ (incl) reaches 0.96–1.00 at every $t$ (it was 0.68–0.85). So much of the apparent drift with $t$ was negative values, which the bias-free ReLU net encodes as about 0.
  - Per-step probes at $t = 9$ barely change, since only a few % of those states are excluded.
  - Scored by $u_k$, the steps since the $k$-th value last changed ($u_k = \min(a_1, \dots, a_k)$; `by_update_age.py`, same probes, `*_by_update.png`):
    - The max is flat at 0.99–1.00.
    - The 3rd-max $R^2$ decays with $u_3$, from 0.70 to 0.43–0.53. It leaks over time, which was hidden in the per-$t$ view.
    - In-bin classifier baselines are trivial for $a_1$ (1.00), and about 0.5 and 0.33 for $a_2$ and $a_3$.
  - Update-event classifiers (`linear-probes-positive/`, `hide/linear-probes-positive/update_probes.py`, about 20 s). Probe 1 reads $[h_{t-1}, x_t]$; probe 2 reads $h_{t-1}$ and asks about $x_{t-1}$. The events are $x_s$ lands at rank $k$, the $k$-th max changed (rank $\le k$), and it shifted (rank $< k$).
    - A new max, and the one-sided changed or shifted events, are linear in both probes (0.93–1.00).
    - "New 2nd max" is linear in $h_s$ (0.97–0.99) but not in $[h_{t-1}, x_t]$. That is an interval in $x_t$, which a linear score (monotone in $x_t$) cannot express.
    - "New 3rd max" is not linear in either probe. It is in the state nonlinearly: gradient-boosted trees reach about 0.95 against a 0.85 baseline. Don't write that the network "keeps little" about it.
    - Weights (`update_weights.png`, stable across seeds).
      - New max at $s = t$ is $x_t$ against unit 2 + unit 4: the previous max is about unit 2 (level) plus unit 4 (record increment), with $R^2$ 0.97.
      - New max at $s = t-1$ is $u_4 > 0$ (99%).
      - New 2nd max at $s = t-1$ is unit 4 negative plus comparators 8, 6 and 1 low.
      - The unit-4 weight in "2nd max changed" at $s = t-1$ is set by regularisation, because the event is nearly separable; read only its sign.

## Gotchas
- Start Claude Code from this workspace root. Auto-memory is stored per launch folder, so a session launched from a subfolder starts with an empty memory. This CLAUDE.md is still found from subfolders.
- `hide/linear-probes-for-every-t/steps_classifier.py` takes about 4.5 min locally (66 logistic fits in a process pool). `linear_probes.py` takes about 13 s.
- In the probe scripts, a constant target gets accuracy NaN: steps since max at $t=0$ (incl) and $t=1$ (excl).
