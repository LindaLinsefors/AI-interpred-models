# Digest: the program-route working paper on $M_{16,10}$

**Paper.** "How ARC's 432-parameter RNN finds the second-largest input, and why its mistakes have resisted a compact account so far". Draft, 28 October 2026. 17 pages, no author line, one reference (Alignment Research Center 2025, *AlgZoo: Uninterpreted Models with Fewer Than 1,500 Parameters*).
File: `reverse-engineered-networks/1-algzoo-second-argmax-432/original-papers/agent-program-route_second-argmax-paper-v3.pdf`.

**How to read this digest.**

- "p. N" is the page number printed in the PDF.
- Numbers and equations are copied from the paper. Anything tagged **[mine]** is my own remark, arithmetic or reproduction and is not in the paper. My reproductions come from `check_program_paper.py` in this folder (output: `check_program_paper_output.txt`; it only loads the weights `.npz` and prints).
- Evidence tags: **W** read directly off the weights; **D** derived from the weights under an assumed gate pattern; **M** measured on sampled sequences; **R** regression or fit; **C** causal test (replacement, removal, forcing, redraw); **E** single example; **P** property of the fitted idealised program; **A** asserted, no evidence shown.
- Notation: neurons $n_0,\dots,n_{15}$ are the components of $h_t$; $w = W_{ih}$; the paper's readout $W_{oh}$ is $W_{out}$; position = step $t = 0..9$; $x_{\max}^{<t}$, $x_{2\mathrm{nd}}^{<t}$ are the maximum and runner-up of the inputs before step $t$.
- Where things are: mechanistic claims in sections 2 and 3, tables in 4, figures in 5, negative results in 6, the paper's own uncertainty in 7, my assessment in 8, my reproduction table in 9.

## 0. The paper in brief

- Claimed algorithm (abstract, p. 1): "It records the inputs that come near the top two, carries the records on a shared set of rotating modes whose phase encodes age, and makes the maximum count against its own position, so that the runner-up's record wins."
- It offers three descriptions and tests each by putting it in place of the network:

| description | size | agreement with the network | network's mistakes kept | task accuracy |
|---|---|---|---|---|
| rule "answer the true second largest" (reference) | 0 | 95.3% | 0% | 100% |
| idealised algorithm (Sec. 3) | 170 numbers + the network's 160 readout weights | 93.2% | 6% | 96.6% |
| explicit formulas (Sec. 7) | 3,888 gains + 16 input weights | 95.7% | 24% | 97.6% |
| full account (Sec. 5) | 285 numbers (network: 272 input + recurrent weights) | 99.6% | 95.6% | 95.23% (on 19,776 development inputs, where the network has 95.29%) |

- Its verdict on "how much of the network's behaviour, including its mistakes, do explanations of this kind keep?" is "largely negative" (p. 2): "of the accounts we tried, only near copies of the weights keep 80%" of the mistakes (p. 1).
- Relation to ARC's account (p. 15): "Where ARC's hand-built network computes every leave-one-out maximum, the trained network records the inputs that come close to the top two and carries them on a tape of rotating clocks."

## 1. Setup, vocabulary and test protocol

### 1.1 Network and task (p. 1)

$$h_t = \mathrm{ReLU}(w\,x_t + W_{hh}\,h_{t-1}),\qquad h_{-1} = 0,\qquad \text{logits} = W_{oh}\,h_9 ,$$

432 parameters: $w$ (16), $W_{hh}$ ($16\times16$), $W_{oh}$ ($10\times16$). No biases. "On standard Gaussian inputs the network answers correctly 95.3% of the time."

### 1.2 What ARC had said, as the paper reports it (p. 1)

- $n_2$ "approximately holds the maximum of the inputs before the latest one", $n_4$ "the increase in the running maximum", and $n_1$, $n_6$, $n_7$ "the maximum of some earlier inputs minus the latest one".
- A hand-built 22-neuron network that computes all ten leave-one-out maxima reaches 99%, "but ARC could not match most of the trained network's neurons to hand-built ones".
- ARC's standard for an explanation: "a mechanistic estimate of the model's accuracy, deduced from its structure, whose error matches that of sampling for a given compute".

### 1.3 Vocabulary (p. 2)

- **gate**: whether a neuron is on (positive pre-activation) or off at a step.
- **new maximum**: an input above every earlier input. **new runner-up**: one between the two largest so far.
- **age** of an input: number of steps since it arrived.
- **near tie**: two of the three largest inputs within 0.05 of each other.
- **own effect** of an input: "its effect on the logit at its own position, per unit of its value".
- **agreement**: fraction of inputs on which a changed network, or an account put in its place, gives the unchanged network's answer ("keeps that share of the network's answers"). **mistakes kept**: that fraction among the inputs the network answers wrongly.
- **tape**: the eight memory neurons $n_2, n_3, n_{10}..n_{15}$ plus $n_0$. **record**: what the recorder $n_5$ writes one step after an input arrives. **trace**: what a past input has left in the state, through its record or through the max detector.
- **pulses**: $n_4$, $n_5$, $n_9$, the three neurons "that fire briefly on events" (p. 4).
- **resting pattern**: $n_4, n_5, n_9$ off and the other thirteen on; **resting map** $A = G\,W_{hh}$ with that $G$ (p. 5). **post-maximum pattern**: one step after a new maximum, only $n_4$ is off (p. 5).
- **displaced contenders**: "inputs that were once among the top two and have been pushed out" (p. 10).

### 1.4 How descriptions are tested (p. 3; "Replacement", p. 15)

- Replacement: "Start from $h_{-1} = 0$; at each step compute the network's update from the current, possibly changed, state, then overwrite the replaced neurons with the account's values, which depend only on the inputs and the account's own state; after step 9, read the logits with the network's readout."
- Floor: the neuron zeroed. Reference: the rule "answer the true second largest" (agreement 95.3%, mistakes kept 0).
- Mistakes are fragile: the network's logit margin has median 1.0 on mistakes and 8.5 on correct answers. So "mistakes kept" is compared with agreement on correct answers reweighted to the mistakes' margin distribution ("margin-matched correct answers"). Calibration: weights rounded to fp16 keep 97.7% of mistakes and 98.1% of margin-matched correct answers. "An account that keeps far fewer mistakes than margin-matched correct answers is closer to the task than to the network."

### 1.5 Data (p. 15)

- Checkpoint: ARC's `example_2nd_argmax()`, "run in true fp32 (float64 for the trace analyses of Sections 2 and 4)". Inputs: independent standard Gaussian rows of ten.
- Mechanism measurements: $2^{17}$–$2^{22}$ fresh sequences (stated per caption).
- Idealised algorithm, full account and explicit formulas are scored on 131,072 held-out rows (6,172 network mistakes, Table 3 caption).
- Full-account removals (Fig. 7) and the "levels at 1" refits: a separate development set of 19,776 rows (931 network mistakes; network accuracy 95.29%), refits trained on 100,000 other rows. Tied-fundamental refits: 20,000 further held-out rows.

## 2. Mechanistic claims, by mechanism

### 2.1 What happens to one input (Fig. 1, p. 3)

"An input passes a threshold when it exceeds it, which turns $n_4$ on and the other comparators off." The comparators "sort each input by the thresholds it passes":

| the input $x_t$ ... | what the paper says happens | evidence |
|---|---|---|
| passes $n_4$'s threshold (new maxima and misfires) | "$n_4$ makes the input's own effect negative, and $n_5$ records a new maximum as about $12x_t$" | D, M |
| passes $n_8$'s threshold, not $n_7$'s ("about half of new runners-up") | "$n_5$ records about $7.7x_t - 2x_{2\mathrm{nd}}^{<t}$; at the next step the input also reaches $n_3$, $n_0$ and $n_{15}$" | D (Fig. 4a), M |
| passes none, above $0.7\,x_{2\mathrm{nd}}^{<t}$ ("inputs near the runner-up so far, below or just above it") | "$n_5$ records about $18x_t - 12.8x_{2\mathrm{nd}}^{<t}$" | D, M (Fig. 4b) |
| passes none, below $0.7\,x_{2\mathrm{nd}}^{<t}$ | "$n_5$ stays off, and $x_t$ drops out of the state" | D, M |

Then: "memory neurons and $n_0$ carry each record and the maximum forward, rotating them at every step"; the thresholds "are read from the tape"; in the readout "each record raises the logit mainly at its input's position; the last input is read directly from the comparators".

### 2.2 The sixteen units

"The sixteen neurons fall into four roles: five comparators, a recorder, two timers and eight memory neurons" (p. 2). The paper says it can name "the roles of seven neurons individually and of the nine tape neurons as a group" (p. 2). Last column: Table 3 (explicit formula alone: agreement / mistakes kept; neuron zeroed: agreement).

| unit | role | input weight (paper; **[mine]** from the weight file in brackets) | what the paper says about it | Table 3 |
|---|---|---|---|---|
| $n_0$ | timer, also tape | [0.069] | "driven by $n_4$ one step after a new maximum and is off the step after that; on other steps it carries records at age 2" (p. 2). It "keeps the maximum's trace from raising the logit one position earlier" (p. 8). Forcing it off at the steps after a new maximum keeps 43% of answers (p. 8). Reached at the next step by inputs that pass $n_8$'s threshold (Fig. 1). | 92.4% / 50% / 19% |
| $n_1$ | comparator | $-10.56$ | on while $x_t < \theta_1$, activation $\lvert w_1\rvert(\theta_1 - x_t)$. Threshold rises above the maximum at max-age 6 (peak 1.42 of the gap). | 96.7% / 61% / 15% |
| $n_2$ | memory | [0.015] | "integrates the maximum: its row of $W_{hh}$ is 0.97 on itself and 1.73 on $n_4$" (p. 2); "the tape's integrator of the maximum" (p. 15). | 98.2% / 85% / 18% |
| $n_3$ | memory | [$-0.123$] | no individual role. Reached at the next step by inputs that pass $n_8$'s threshold. | 93.8% / 56% / 55% |
| $n_4$ | comparator = **max detector** | $+10.16$ | the only comparator with a positive input weight; fires when $x_t > \theta_4$, which "lies just below the maximum so far while the maximum is young" and sags toward the runner-up at ages 5–7. Firing on a non-maximum is a "misfire". The newest position reads it with weight $-4.99$. A pulse in the idealised and full accounts. | 99.5% / 95% / 18% |
| $n_5$ | **recorder** | [$-0.281$]; the paper's formula has $-0.28x$ | "reads the previous step's comparators and writes, one step after an input arrives, a record of that input: a value that grows with it" (p. 2). See 2.6. | 94.2% / 57% / 21% |
| $n_6$ | comparator | $-11.00$ | threshold at the runner-up one step after a new maximum, above the maximum at max-age 4. | 97.3% / 70% / 14% |
| $n_7$ | comparator | $-13.17$ | threshold at the maximum while it is young, sags toward the runner-up at ages 5–7. The recorder reads it. | 96.9% / 66% / 10% |
| $n_8$ | comparator | $-12.31$ | threshold near the runner-up while the maximum is young, above the maximum at ages 7–8. "About half of new runners-up" pass it and not $n_7$'s. The recorder reads it. | 96.2% / 53% / 18% |
| $n_9$ | timer | [$-0.228$] | "fires one step after a new maximum (its gate equals $n_4$'s gate one step earlier on 99.0% of steps after the first)" (p. 2). An arriving input sends it 84.3 per unit, but it "is off unless $n_4$ fired" (p. 6). Forcing it off after new maxima keeps 67%. | 96.8% / 80% / 67% |
| $n_{10}$ | memory | [0.060] | no individual role | 90.6% / 52% / 13% |
| $n_{11}$ | memory | [0.148] | no individual role | 95.3% / 57% / 11% |
| $n_{12}$ | memory | [0.298] | no individual role | 90.6% / 46% / 13% |
| $n_{13}$ | memory | [$-0.047$] | no individual role | 93.3% / 57% / 30% |
| $n_{14}$ | memory | idealised $-1.25$ [$-1.321$] | the one tape neuron whose input weight the idealised algorithm keeps; the removal "tape input weights → 0" exempts it. | 93.3% / 62% / 20% |
| $n_{15}$ | memory | [$-0.119$] | no individual role. Reached at the next step by inputs that pass $n_8$'s threshold. | 93.7% / 53% / 64% |

The memory neurons "have small input weights and carry the maximum and the records forward" (p. 2). In the full account, thirteen neurons (0, 1, 2, 3, 6, 7, 8, 10–15) are "ordinary" and $n_4, n_5, n_9$ are pulses (p. 9).

### 2.3 How the running maximum is stored

| # | claim | numbers / equations | evidence | page |
|---|---|---|---|---|
| M1 | $n_2$ integrates the maximum. | row 2 of $W_{hh}$: 0.97 on itself, 1.73 on $n_4$ | W | 2 |
| M2 | $n_4$ fires when the input exceeds a threshold that the previous state sets and that lies just below the maximum so far while the maximum is young. | $\theta_4 = -W_{hh}[4]\cdot h_{t-1}/w_4$, $w_4 = +10.16$. Threshold position (0 = runner-up, 1 = maximum) about 1 at ages 1–3, about 0.62 at ages 6–7 (Fig. 3). | W, M | 2, 6 |
| M3 | In the eigenbasis of the resting map the maximum is a "level", a real eigenvalue slightly above 1. | eigenvalue 1.0098: "a 1% growth per step, 10% over ten steps". $R^2 = 0.999$ of that mode's coordinate on a linear function of the top three earlier values $M, S, T$ within each pair of ages. | D, R | 9 |
| M4 | The maximum also leaves a rotating trace. "Right after a new maximum its trace lies along $n_4$; the post-maximum pattern and then the resting map carry it forward and rotate it." | "maximum's clock" = main clock: rotation 42.75° per step, radius 0.951; $R^2 = 0.989$ | D, R | 6, 9 |
| M5 | The exact level eigenvalues matter. | setting both level eigenvalues to 1, no refit, keeps 89% of answers | C | 10 |
| M6 | Idealised counterpart: the exact positive maximum $M' = \max(M, x)$, a pulse $h_4 = 10[x - M]_+$, and tape features $M$, $M\cos\phi_M$, $M\sin\phi_M$, $M r$. | max detector's 10 (network 10.16); clock 0.74 rad | P | 4, 16–17 |

### 2.4 How the runner-up is stored

| # | claim | numbers / equations | evidence | page |
|---|---|---|---|---|
| S1 | The runner-up is the second "level" of the resting map. | real eigenvalue 0.9736; $R^2 = 0.986$ on $(M, S, T)$ within age cells | D, R | 9 |
| S2 | There is no separate clock for the runner-up: "the runner-up's record rotates in the same modes as every record, and its phase encodes its age". | Table 2, row "runner-up's clock" | D | 9 |
| S3 | A new runner-up is written by the recorder one step after it arrives (see 2.6); "a final runner-up that arrives before the last step is recorded on 99% of sequences". | 99% | M | 6 |
| S4 | A displaced maximum becomes the runner-up by "relabelling". "A new maximum switches $n_1$, $n_6$, $n_7$ and $n_8$ all off at once (on 86% of arrivals), which exposes the memory neurons' positive reading of the old maximum: its own effect jumps to +30 per unit, against −16 had no new maximum arrived." | 86%; +30 against −16 (Fig. 6c) | M, D | 8 |
| S5 | A runner-up at the last position, "about a tenth of sequences, is never recorded; the readout reads it from the comparators' gates at that step". | about a tenth | M for the frequency, A for the readout | 6 |
| S6 | Idealised counterpart: $S' = \max(S, \min(M, x))$ with its own clock, reset to 1 when the value is replaced; "on a new maximum, the old maximum's clock passes to the runner-up". Tape features $S$, $S z_S$, $S z_S^2$, $S z_S^3$ (real and imaginary parts) and $S\cdot[\text{new runner-up}]$. | 42.4° per step, radius 1; harmonics 84.8° and 127.2° | P | 4, 9, 16–17 |

### 2.5 Comparators and their thresholds

| # | claim | numbers / equations | evidence | page |
|---|---|---|---|---|
| C1 | The five neurons with large input weights switch at a threshold that the previous state sets. | $\theta_j = -W_{hh}[j]\cdot h_{t-1}/w_j$; $n_4$ (+10.16), $n_1$ (−10.56), $n_6$ (−11.00), $n_8$ (−12.31), $n_7$ (−13.17) | W | 2 |
| C2 | $n_1, n_6, n_7, n_8$ are on while the input is below their thresholds and their activations hold how far below. | $h_{t,j} = \lvert w_j\rvert(\theta_j - x_t)$ | W | 2 |
| C3 | Each threshold is nearly linear in the top two so far, within groups of the same step and the same ages of maximum and runner-up. | $\theta_j \approx g_{\max}\,x_{\max}^{<t} + g_{2\mathrm{nd}}\,x_{2\mathrm{nd}}^{<t} + b$ "explains 96–99% of each threshold's variance"; groups of at least 30 sequences | R | 5, 15 |
| C4 | "The gains on the maximum follow from the weights." | derived gain "reads the trace $A^{a-2}\,G_{\mathrm{post}}\,W_{hh}\,e_4$ through comparator $j$'s row"; derived and fitted gains agree "to within 0.1 at ages 2–9" | D against R | 6, 15 |
| C5 | Each threshold's position between runner-up and maximum swings with the maximum's age, following one clock. | position $= (\theta_j - x_{2\mathrm{nd}}^{<t})/(x_{\max}^{<t} - x_{2\mathrm{nd}}^{<t})$; a fit of the thresholds computed from the weights on $\{1, \cos 0.74a, \sin 0.74a, r\}$, $a$ = age − 1, $r = [a = 0]$, has $R^2 \ge 0.993$ | D, R, M (Fig. 3) | 5–6 |
| C6 | "The staggered peaks are one rotating vector read along five directions. $n_4$ and $n_7$ sit at the maximum while it is young and sag toward the runner-up at ages 5–7; $n_6$, $n_1$ and $n_8$ rise above the maximum in turn, at ages 4, 6 and 7–8." | $n_1$ peaks at 1.42 of the gap | M | 6 |
| C7 | Every threshold stays at or above the runner-up "(to within 0.05 of the gap)", so an input below the runner-up so far rarely passes any, and a new maximum usually passes each. | an input below the runner-up so far passes each comparator's threshold on "at most 0.5%"; "a new maximum passes each of them on 93–95% of arrivals" | M | 6 |
| C8 | The sag of $n_4$'s threshold causes misfires. | "on new runners-up within 0.05 of the maximum it misfires on 27% at age 1 and 96–98% at ages 5–7"; forcing $n_4$ off whenever the input is not a new maximum raises accuracy by 0.35 points | M, C | 6, 11 |
| C9 | Idealised counterpart: threshold = runner-up plus a clock-dependent share of the gap. | $\theta_j = S + (M - S)\,\alpha_j$, $\alpha_j = \alpha_j^0 + \alpha_j^c\cos\phi + \alpha_j^s\sin\phi + \alpha_j^r r$ (Table 1); gains 10.5, 11, 13, 12 (network 10.56, 11, 13.17, 12.31). Same clock, "smaller swings": idealised $n_1$ peaks at 0.85 of the gap against 1.42. | P | 4–6 |
| C10 | The network's behaviour is very sensitive to the thresholds. | idealised comparators in place of the network's together keep 28% (worst one alone 33%); the network's own clock-fit thresholds, one comparator at a time, keep 78–88% (max detector's 99.2%); comparators reading only the two levels and the main clock keep 36%; rounding the gains to 10.5, 11, 13, 12 keeps 89% | C | 5, 10, 14 |

### 2.6 The recorder $n_5$

| # | claim | numbers / equations | evidence | page |
|---|---|---|---|---|
| R1 | An input that passes no threshold reaches the rest of the network only through the recorder. | "with all four comparators $n_1, n_6, n_7, n_8$ on, $W_{hh}$ sends it 18.0 per unit to $n_5$, 84.3 to $n_9$ (which is off unless $n_4$ fired) and at most 0.4 to any other neuron" | W / D (Fig. 4a: $W_{hh} G_t w$) | 6 |
| R2 | It records nearly every new maximum, "as about 12 times its value". | idealised pulse $h_5 = 12e$, $e = [x - S]_+$ of the previous step; explicit formula after a new maximum $[12.14M - 3.19S - 3.58T - 0.28x]_+$ | M, R | 4, 6, 12 |
| R3 | It records "most new runners-up". One that passes $n_8$'s threshold "is recorded as about $7.7x_t - 2x_{2\mathrm{nd}}^{<t}$". | explicit formula after a new runner-up $[0.89M + 7.81S - 3.54T - 0.28x]_+$, "where $S$ is the new runner-up itself" | M, R | 6, 12 |
| R4 | For an input that passes no threshold, it "compares such an input with the runner-up so far, through $n_7$ and $n_8$, and writes about $18x_t - 12.8x_{2\mathrm{nd}}^{<t}$". | "it turns on when the input exceeds 0.72–0.78 of the runner-up, slightly above the 0.71 this formula gives" | D, M (Fig. 4b) | 6 |
| R5 | An input it does not record "reaches the state only through the tape neurons' small input weights". | — | D | 6 |
| R6 | Runner-up against third: "whichever of the two arrives later is recorded relative to the earlier one: the later value's record is about 18 times its value minus 8–13 times the earlier one, so their part of the margin grows with their difference." | 18; 8–13 | A (no figure) | 8 |
| R7 | The formulas give the gate but not the value. | setting only $n_5$'s gate from them, inside the network at every step, keeps 98% of answers; setting its value too keeps 78% | C | 6 |
| R8 | Recording inputs below the runner-up is a cost: "Recording the third at all only narrows the runner-up's margin". | an oracle that forces the recorder off after every input below the runner-up so far raises accuracy from 95.3% to 95.9% | C | 8, 11 |
| R9 | Example: inputs 1.58, 0.57, 0.42 at steps 2, 3, 4 give records 19.5, 5.8, 0.3 at steps 3, 4, 5. | Fig. 2 | E | 3–4 |

### 2.7 Timers and the reset

| # | claim | numbers / equations | evidence | page |
|---|---|---|---|---|
| T1 | $n_9$ fires one step after a new maximum. | "its gate equals $n_4$'s gate one step earlier on 99.0% of steps after the first"; idealised $h_9 = 1.5\,M r$ | M, P | 2, 4 |
| T2 | $n_0$ is driven by $n_4$ one step after a new maximum, is off the step after that, and otherwise "carries records at age 2". | — | A (Fig. 2b shows the on/off pattern once) | 2 |
| T3 | The timers make the maximum count against its own position during its first two steps: "before the other comparators' negative reading has built up, the timers keep the memory neurons' positive reading from making the maximum's own effect positive ... one step after the arrival $n_9$ fires and $n_0$, driven by $n_4$, keeps the maximum's trace from raising the logit one position earlier." | forcing $n_0$ off at the steps after a new maximum keeps 43% of answers, $n_9$ 67% | C, D | 8 |
| T4 | The reset has an eigenmode of its own. | "reset transient: real eigenvalue 0.231"; $R^2 = 0.98$; removing it keeps 89% | D, R, C | 9–10 |
| T5 | Idealised counterpart: a reset flag $r$, "1 for one step after a new maximum", entering the thresholds ($\alpha_j^r$), $h_9$ and the tape feature $M r$. | Table 1; Appendix B | P | 4 |
| T6 | $n_5$ and $n_9$ are on at step 0 when $x_0 < 0$, "because the state starts at zero, their input weights are negative". | Fig. 2 caption | E, W | 4 |

### 2.8 The tape: how ages and positions are stored

| # | claim | numbers / equations | evidence | page |
|---|---|---|---|---|
| A1 | "The tape rotates each record at every step, so that its direction encodes its age and the readout raises the logit at its input's position." | — | D (Fig. 5a), M (Fig. 5b) | 7 |
| A2 | A fixed gate pattern makes the update linear; the reference is the resting map. | $h_t = G(w x_t + W_{hh} h_{t-1})$; $A = G\,W_{hh}$ with $n_4, n_5, n_9$ off | D | 5 |
| A3 | Eigenvalues of $A$: two levels, a main clock, two harmonics, a reset transient, two "history modes" and an "echo". | 1.0098; 0.9736; rotation 42.75° with radius 0.951; 83.27° with 0.937; 124.78° with 0.837; 0.231; alternating −0.928; short-lived rotation 43.0° with radius 0.735; 0.001 | D | 7, 9 |
| A4 | The clocks make a linear readout of position possible: "rotation, with its harmonics, is a Fourier code for age, from which a linear readout can raise the logit at the position where a value arrived". | — | A (interpretation) | 5 |
| A5 | A record carried by $A$ and read through the readout has a large positive own effect. | "+62 to +76 per unit at its input's position, about +20 one position later, and less or negative elsewhere" (Fig. 5a: 72, 76, 71, 70, 67, 62, 62 for positions 2..8) | D | 7 |
| A6 | Measured records agree for recent positions and fade for old ones. | recorded third-largest inputs: +61 to +67 at positions 4–8; "+55 and +39 measured at positions 3 and 2, against +76 and +72" | M (Fig. 5b, $2^{20}$ sequences) | 7 |
| A7 | The fading is "consistent with some tape neurons being clipped at zero on most steps, dropping part of each record". | resting pattern exact on 17% of steps whose input is below the runner-up so far; all thirteen on 34% | M (hedged) | 5, 7 |
| A8 | The harmonics are multiples of 41.6°, not of the main clock's 42.75°. | 83.27° and 124.78°; idealised ties all three to 42.4° | D | 10 |
| A9 | Two modes with no idealised counterpart carry "older inputs outside the top three, among them displaced contenders". | alternating mode and short-lived rotation "are only 81–91% explained" by the top three within age cells; "on resting steps adding the last two inputs explains almost none of the rest" | R | 10 |
| A10 | "The harmonics are partly history too (87%), and the echo holds the current input for one step." | $R^2$ 0.87, 0.87; echo 0.54 | R | 10 |
| A11 | Removing the history modes changes many answers. | short-lived rotation → 0 keeps 51%; alternating mode → 0 keeps 91% | C | 10 |
| A12 | Clipping is part of the mechanism. "The tape neurons' rectification is written back into the modes: a clipped neuron removes part of each mode it reads." | writing back the unclipped pre-activations keeps 60% | C | 10 |
| A13 | Calibration of clocks and inputs. | main clock radius → 1 keeps 92%; harmonic damping → powers of the main clock's keeps 98% of answers and 82% of mistakes; tape input weights → 0 except $n_{14}$'s keeps 97% of answers and 78% of mistakes; harmonics tied at 2× and 3× 41.6°, or main clock → 42.4°, keep at least 99.4%; all three angles → ideal keeps 98.3% of answers and 84% of mistakes | C | 10 |
| A14 | The resting map is "a reference, not the typical step". | 34% and 17% as in A7 | M | 5 |

### 2.9 How the readout forms the answer

| # | claim | numbers / equations | evidence | page |
|---|---|---|---|---|
| O1 | Each record raises the logit mainly at its input's position. | see A5, A6 | D, M | 7 |
| O2 | A new maximum at the last position counts against itself through $n_4$. | "The newest position reads $n_4$ with weight −4.99, so a new maximum at the last position has an own effect of $10.16 \times (-4.99) = -50.7$ per unit." | W | 8 |
| O3 | An older maximum counts against itself through the other comparators. | "From age 2 on, the four other comparators give the maximum an own effect of −72 per unit and the memory neurons +54." Fig. 6c (means): −72, 0, 0, +54, total −18. Fig. 6a (medians): about −4 to −8 at ages 2–9. | M, D | 8 |
| O4 | "Either way, the runner-up's trace raises the maximum's logit at least as much as its own, so the margin between them comes from the maximum's negative own effect." | "The median margin of the runner-up's logit over the maximum's is +12." | M for +12, A for the rest | 8 |
| O5 | The runner-up beats the third through the relative record (R6). | — | A | 8 |
| O6 | "Most other inputs are never recorded and contribute almost nothing." | — | A | 8 |
| O7 | The last input is read directly from the comparators. | Fig. 1; S5 | A | 3, 6 |

### 2.10 How errors arise

| # | claim | numbers | evidence | page |
|---|---|---|---|---|
| E1 | Kinds of mistake. | "The network errs on 4.7% of inputs: it answers the third largest on 2.97%, the maximum on 0.95% and a smaller input on 0.76%." | M | 11 |
| E2 | Mistakes are mostly near ties. | errors whose answer lies within 0.05 of the runner-up are 65% of the errors ($2^{22}$ sequences): runner-up/third near ties 48%, maximum/runner-up near ties 15%, a smaller input within 0.05 of the runner-up 2% | M | 11 |
| E3 | On a near tie "the comparison of the two values is small, and the rest of the margin decides, including the maximum's term and traces of other inputs". | — | A, supported by E7, E8 | 11 |
| E4 | The idealised algorithm is better on near ties and worse elsewhere. | right on 91% of runner-up/third near ties against the network's 78%; on inputs without near ties 97.3% against 98.4% | M | 11 |
| E5 | Cost of unnecessary records. | recorder forced off after every input below the runner-up so far: 95.3% → 95.9% | C | 11 |
| E6 | Cost of misfires. | $n_4$ forced off whenever the input is not a new maximum: +0.35 points | C | 11 |
| E7 | A displaced contender can flip the answer. | $[1.2, 0.1, 0, 1.23, 1.26, 0, 0, 0, 0, 0]$ is answered 3 (correct); with the second entry changed to 1.1 it is answered 0. The deciding margin differs by 5.5 logits: +6.5 from the short-lived rotation, +0.9 and +0.6 from the second and third harmonics, −1.1 from the main clock, −1.3 from the alternating mode, "almost nothing from the levels". The idealised algorithm answers 3 on both. | E, D (full account) | 11–12 |
| E8 | Inputs outside the top three decide many mistakes. | keeping the top three of the first nine inputs and the last input, and redrawing every other input from the standard normal truncated below the third largest, changes the answer on 36% of wrong inputs and 1.4% of right ones (131,072 inputs); only displaced contenders (on average two inputs): 23%; only inputs never in the top two (on average four): 25%; margin-matched correct answers: 14.5% | C | 12 |
| E9 | So "inputs that were never contenders leave traces too". | from E8 | C | 12 |
| E10 | Mistakes sit at small margins. | median margin 1.0 on mistakes, 8.5 on correct answers | M | 3 |
| E11 | The network is sensitive to weight precision. | 2 significant decimal digits keep 67.9% of answers; 3 digits 98.4% of answers and 82.3% of mistakes; fp16 99.8% and 97.7% | C | 13–14 |
| E12 | Summary (Discussion): mistakes "are near ties decided by small calibrations of the levels, clocks and gains, by traces of inputs outside the top three, by clipping of the tape, and by how the comparators read those traces". | — | summary | 15 |

### 2.11 The updates as a timeline

This only reorders the claims above (the ordering is **[mine]**; pages as given there).

A new maximum arrives at step $t$:

- Step $t$: $x_t > \theta_4$, so $n_4 = 10.16\,(x_t - \theta_4)$ is on (M2), and $n_1, n_6, n_7, n_8$ all go off on 86% of arrivals (S4). If $t = 9$, the readout reads $n_4$ with weight −4.99 (O2).
- Step $t+1$, the post-maximum pattern (only $n_4$ off): $n_9$ fires (T1); $n_5$ writes the record, about 12 times the value (R2); $n_0$ is driven by $n_4$ (T2); $n_2$ adds 1.73 per unit of $n_4$ to 0.97 of itself (M1); the old maximum is relabelled as runner-up (S4); the thresholds take their reset values (C5, C9). The new maximum's own effect is about −29 per unit (read by eye from Fig. 6a; −28.2 in my reproduction).
- Step $t+2$: $n_0$ is off (T2).
- From then on, with the resting map as reference: the levels persist (eigenvalues 1.0098 and 0.9736), the trace rotates 42.75° per step with radius 0.951 (M4, A3), the thresholds swing with the maximum's age (C5, C6), and the four comparators give the maximum a negative own effect that outweighs the memory neurons' positive one (O3).

Any other input arrives at step $t$:

- Step $t$: each comparator that is on holds $\lvert w_j\rvert(\theta_j - x_t)$ (C2).
- Step $t+1$: $n_5$ writes a record if the input was a new runner-up or exceeded about 0.72–0.78 of the runner-up so far (R3, R4); otherwise the input "drops out of the state" (R5).
- Later steps: the tape carries the record, so that at the end it raises the logit at its own position by +62 to +76 per unit and the next position by about +20 (A5); old records fade (A6).

## 3. The three accounts

### 3.1 The idealised algorithm (Sec. 3, pp. 4–5; Appendix A, p. 16; Appendix B, pp. 16–17)

State carried from step to step, "six named quantities" (p. 4):

- the exact positive maximum $M$ and runner-up $S$ so far, $M' = \max(M, x)$, $S' = \max(S, \min(M, x))$;
- a clock for each, "a unit complex number multiplied by $e^{0.74i}$ every step (42.4°) and reset to 1 when its value is replaced; on a new maximum, the old maximum's clock passes to the runner-up";
- a reset flag $r$, 1 for one step after a new maximum;
- the previous step's increment $e = [x - S]_+$, "how far the input exceeded the runner-up".

Emission of all sixteen neurons at each step (p. 4). Comparators and pulses read the state before the update:

$$h_j = g_j\,[\theta_j - x]_+,\quad \theta_j = S + (M - S)\,\alpha_j,\quad \alpha_j = \alpha_j^0 + \alpha_j^c\cos\phi + \alpha_j^s\sin\phi + \alpha_j^r\,r \qquad (j = 1, 6, 7, 8),$$

$$h_4 = 10\,[x - M]_+,\qquad h_5 = 12\,e,\qquad h_9 = 1.5\,M r ,$$

with $\phi$ the phase of the maximum's clock. The nine tape neurons read the updated state: "each is a rectified sum, with half-integer weights, of twelve named features: $M$, $S$, $M$ times the maximum's clock (two features) and reset flag, $S$ times the runner-up's clock, its square and its cube (six features), and $S$ times a flag for a new runner-up. $n_{14}$ also has an input weight of −1.25." The network's own readout is applied to the emitted state.

The complete program (Appendix B), copied:

```python
import math
import numpy as np
ROT = complex(math.cos(.74), math.sin(.74))
COMPARATORS = [1, 6, 7, 8]
TAPE = [0, 2, 3, 10, 11, 12, 13, 14, 15]
GAIN = np.array([10.5, 11, 13, 12])
ALPHA = np.array([
    [0.57, -0.22, -0.18, 0.27],  # neuron 1
    [0.75, -0.04, 0.03, -0.74],  # neuron 6
    [0.76, 0.07, -0.01, -0.04],  # neuron 7
    [0.68, 0.04, -0.32, -0.47],  # neuron 8
])
EMIT = np.array([
    [10.5, -5.5, 0.5, -3.5, -2.0, -2.0, -2.0, 3.0, 0.0, 1.5, 3.0, 1.0],  # neuron 0
    [12.5, 7.5, -0.5, 1.0, -6.5, 0.5, 1.5, -0.5, 1.5, -2.0, 1.5, -1.0],  # neuron 2
    [3.0, 1.5, 1.0, 0.0, -3.0, 0.5, -0.5, 1.5, -3.0, 5.5, -0.5, -8.0],  # neuron 3
    [10.5, -5.5, 0.5, 4.5, -4.0, -0.5, 6.5, 2.5, -0.5, -0.5, -2.5, 3.0],  # neuron 10
    [10.0, -6.0, 3.5, 3.0, -6.5, 4.0, 3.0, -2.0, -3.0, 1.5, 1.5, 1.0],  # neuron 11
    [13.0, -4.0, 5.0, -1.5, -7.5, 6.0, -1.5, -1.0, 2.0, 2.5, -2.0, -3.5],  # neuron 12
    [5.5, 1.5, -0.5, 0.0, -2.0, -0.5, -1.0, 3.5, 0.5, 0.5, 0.0, -3.0],  # neuron 13
    [10.5, -1.5, 0.0, -3.5, -2.0, -2.5, -3.0, -2.5, 0.0, -0.5, -2.5, 8.5],  # neuron 14
    [8.0, -1.5, 3.0, -3.0, -6.0, 3.0, -3.5, 4.0, -4.5, -0.5, 0.0, -5.5],  # neuron 15
])

def step(carry, x):
    if carry is None:
        carry = (0., 0., 1., 1., 1., 0.)
    M, S, zM, zS, reset, increment = carry
    phase = np.array([1., zM.real, zM.imag, reset])
    threshold = S + (M-S)*(ALPHA @ phase)
    h = np.zeros(16)
    h[COMPARATORS] = GAIN * np.maximum(threshold-x, 0.)
    h[4] = 10*max(x-M, 0.)
    h[5] = 12*increment
    h[9] = 1.5*M*reset
    increment = max(x-S, 0.)
    zM, zS = zM*ROT, zS*ROT
    reset = runner_reset = 0.
    if x > M:
        M, S, zM, zS, reset = x, M, 1., zM, 1.
    elif x > S:
        S, zS, runner_reset = x, 1., 1.
    features = [M, S, M*zM.real, M*zM.imag, M*reset]
    for z in (zS, zS**2, zS**3):
        features.extend((S*z.real, S*z.imag))
    features.append(S*runner_reset)
    h[TAPE] = EMIT @ features
    h[14] -= 1.25*x
    return (M, S, zM, zS, reset, increment), np.maximum(h, 0.)
```

Columns of `EMIT`: $M$, $S$, $M\,\mathrm{Re}\,z_M$, $M\,\mathrm{Im}\,z_M$, $M r$, $S\,\mathrm{Re}\,z_S$, $S\,\mathrm{Im}\,z_S$, $S\,\mathrm{Re}\,z_S^2$, $S\,\mathrm{Im}\,z_S^2$, $S\,\mathrm{Re}\,z_S^3$, $S\,\mathrm{Im}\,z_S^3$, $S\cdot[\text{new runner-up}]$.

How it was obtained (pp. 5, 16):

- Hand-set constants "rounded from the network: the clock angle (0.74 rad against the network's 0.746), the comparator gains, the max detector's 10, the recorder's 12, $n_9$'s 1.5 and $n_{14}$'s −1.25".
- Emission weights and threshold coefficients "initialised by least squares, then trained by Adam (4,001 steps, batch 2,048) on final-step task cross-entropy through the fixed readout plus 0.02 times the squared error to the network's hidden states, on 32,768 training rows, and rounded" (to halves and hundredths). "They are calibrated, partly to the network, not derived from its weights."

Performance (pp. 5, 13, Table 4):

- 96.6% correct on 131,072 held-out inputs (network 95.3%). 170 numbers ("every numeric literal in its program, indices and zeros included"); with the 160 borrowed readout weights 330, against 432.
- Agreement with the network 93.2%; mistakes kept 6% (6.0% in Table 4); margin-matched correct answers kept 90.0%. Both right on 92.9% of inputs, only the idealised algorithm on 3.7%, only the network on 2.4%.
- Its emitted comparators, pulses and $n_2$ correlate 0.96–1.00 with the network's, its other tape neurons 0.67–0.88, "partly by construction".
- Its four comparators put in place of the network's together keep 28% of the answers; the worst single comparator 33%.

Its counterparts in the network, as the paper lists them (p. 5): "the clock-swung thresholds in the comparators, the records in the recorder, the clocks in the tape, and the reset in the max detector and timers".

### 3.2 The full account (Sec. 5, pp. 9–10; Appendix A, p. 15)

With ordinary neurons (0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15) and pulses $p = (h_4, h_5, h_9)$, the 13-dimensional state $m$ obeys

$$f_t = \Lambda\,m_{t-1} + K\,p_{t-1},\qquad u_t = [\,V f_t + d\,x_t\,]_+,\qquad p_t = [\,Q\,m_{t-1} + F\,p_{t-1} + e\,x_t\,]_+,\qquad m_t = V^{-1} u_t .$$

- "$V$ is the eigenvector basis of the ordinary neurons' block of the resting map, with phases aligned to the recorder's input and columns scaled to a largest entry of one; $\Lambda$ holds the eigenvalues." Rounding grids: $V, Q$: 1/1024; $K, F$: 1/256; $d, e$: 1/64; eigenvalues: 1/2048.
- In words (p. 9): "each coordinate decays or rotates by its eigenvalue; the pulses add fixed vectors to the coordinates; each neuron reads the coordinates through a fixed row, adds its input weight times the input and is rectified; and the result is written back."
- "It is a reparametrisation of the weights, not a compression or an independent explanation": 285 numbers against the network's 272 input and recurrent weights. **[mine]** $285 = 13 + 39 + 169 + 13 + 39 + 9 + 3$ for $\Lambda, K, V, d, Q, F, e$.
- Agreement 99.6% on 131,072 held-out inputs, 95.6% of mistakes kept; worst neuron alone 99.5% (Table 4). On the 19,776 development inputs: accuracy 95.23% (network 95.29%), agreement 99.65%, 96.7% of mistakes kept (Fig. 7 caption).
- Its use: "each idealised quantity becomes a separate constant (Table 2) that can be removed or idealised on its own". The corrections the network adds to the idealised constants are named as "calibration, a history of inputs outside the top three, clipping, and how the comparators read the history" (p. 9). The removals are in Fig. 7 (section 5 below) and in A11–A13, M5, C10, T4.
- "None of these removals raises the account's accuracy above the network's 95.29% on these inputs, or toward the idealised algorithm's 96.6%." (p. 10)

### 3.3 The explicit formulas (Sec. 7, pp. 12–13; Appendix A, p. 16)

$$\hat h_{t,i} = \big[\,d_i\,x_t + g_i^M(a,b)\,M + g_i^S(a,b)\,S + g_i^T(a,b)\,T\,\big]_+ ,$$

"where $M \ge S \ge T$ are the three largest positive earlier inputs, $a$ and $b$ the ages of the largest two, $d_i$ the neuron's input weight", and the gains are "fitted for each pair of ages to the neuron's pre-activation on 40,000 inputs, 3,888 gains in all (16 neurons, three gains, 81 age cells including those for an absent value)". Ridge regressions (ridge 0.01) for each age pair; "ages are counted to the previous step and set to 10 when a value is absent" (p. 16).

- One neuron at a time: every formula keeps at least 90.6% of answers and 46–95% of mistakes; the neuron zeroed keeps 10–67% (Table 3). But "only seven of the sixteen keep more answers than the rule's 95.3%".
- All sixteen together: 95.7% of answers, 24% of mistakes (23.9% in Table 4), against 92.4% of margin-matched correct answers; "about evenly across the network's three kinds of mistake (23% of those that answer the third largest, 22% the maximum, 28% a smaller input)". Accuracy 97.6%, above both the network and the idealised algorithm.
- Why only 24%: "Redrawing every input outside the top three, which keeps their positions, still leaves about 64% of the network's mistakes unchanged (Section 6), so a description by the top three and their positions could keep at most about that many; the formulas keep 24% because, besides ignoring history, they omit the third value's age and are linear within each age cell."
- Their age tables "follow the clock of Figure 3", but compressing them to damped clock curves ("336 numbers for the 72 cells with both values present, keeping the other 9 cells' 432 gains and the 16 input weights") "explains 89% of the gains' variance and keeps only 50% of the network's answers together".
- "The formulas recover the roles": the recorder's is $[12.14M - 3.19S - 3.58T - 0.28x]_+$ after a new maximum and $[0.89M + 7.81S - 3.54T - 0.28x]_+$ after a new runner-up, "and the comparators' gains reproduce Figure 3".
- Integration (p. 13): at the last step the formulas' logits depend only on $(x_9, M, S, T, a, b)$ and scale with $M$. "Writing $S = \sigma M$, $T = \sigma\tau M$ and $x_9 = zM$, the sixteen ReLU breakpoints and the truth boundaries $z = \sigma$ and $z = 1$ cut the $z$-line into intervals on which every logit is affine; on each interval the correctly answered inputs form an interval whose probability is a difference of normal CDFs." The rest is integrated by quadrature over the exact density of the top three, $504\,\varphi(M)\varphi(S)\varphi(T)\,\Phi(T)^6$, "for each of the 72 equally likely age pairs, with the cases of fewer than three positive earlier inputs handled separately". Gauss–Legendre over $\sigma$, $\tau$ and $M \in [0, 8]$; "the cases with zero or one positive earlier input condition on the largest negative earlier input" (p. 16). Result 97.57%; "three grids agree to 0.005 points, and the formulas' sampled accuracy is 97.59%".
- Verdict: "Judged by ARC's criterion, a comparison with sampling at equal cost, it fails: the 40,000 fitting samples alone would estimate the network's accuracy to about ±0.1 points, while the integral is 2.3 points too high, the standard error of a sample of about 85 inputs. It is a mechanistic estimate of the formulas' accuracy, not of the network's."
- Extension: "Adding the previous input, a leaky input sum and the third value's age together raises the mistakes they keep to 41%; we did not separate the history terms from the third value's age."
- Why the formulas and not the idealised algorithm were integrated: "an estimate of the network's accuracy has to start from a description that agrees with the network" (p. 13).

## 4. The paper's tables

### Table 1 (p. 5)

"In the idealised algorithm, each comparator's threshold is the runner-up plus a clock-dependent share of the gap to the maximum. $\theta_j = S + (M - S)(\alpha_j^0 + \alpha_j^c\cos\phi + \alpha_j^s\sin\phi + \alpha_j^r r)$, where $\phi$ is the phase of the maximum's clock and $r$ the reset flag."

| comparator | gain | $\alpha^0$ | $\alpha^c$ (cos) | $\alpha^s$ (sin) | $\alpha^r$ (reset) |
|---|---|---|---|---|---|
| $n_1$ | 10.5 | 0.57 | −0.22 | −0.18 | 0.27 |
| $n_6$ | 11 | 0.75 | −0.04 | 0.03 | −0.74 |
| $n_7$ | 13 | 0.76 | 0.07 | −0.01 | −0.04 |
| $n_8$ | 12 | 0.68 | 0.04 | −0.32 | −0.47 |

### Table 2 (p. 9)

"Each idealised quantity is a separate constant of the full account; the top values explain the levels, main clock and reset transient but not the history modes."

| idealised quantity | in the full account | in the idealised algorithm | $R^2$ by $M, S, T$ |
|---|---|---|---|
| maximum $M$ | level: real eigenvalue 1.0098 | exact, 1 | 0.999 |
| runner-up $S$ | level: real eigenvalue 0.9736 | exact, 1 | 0.986 |
| maximum's clock | main clock: rotation 42.75° per step, radius 0.951 | 42.4°, radius 1 | 0.989 |
| runner-up's clock | no separate mode: the runner-up's record rotates in the same modes as every record, and its phase encodes its age (Figure 5) | 42.4°, radius 1 | (blank) |
| runner-up's clock squared and cubed | harmonics: 83.27° and 124.78° per step, radii 0.937 and 0.837 | 84.8° and 127.2°, radius 1 | 0.87, 0.87 |
| reset flag | reset transient: real eigenvalue 0.231 | one-step flag | 0.98 |
| (inputs outside the top three) | history modes: alternating (−0.928) and short-lived rotation (43.0°, radius 0.735) | absent | 0.81, 0.91 |
| (current input) | echo: real eigenvalue 0.001 | absent | 0.54 |
| comparator gains | 10.56, 11, 13.17, 12.31 | 10.5, 11, 13, 12 | (blank) |
| max detector's input weight | 10.16 | 10 | (blank) |

Caption notes: the idealised gains and angle were rounded from these values, "so their closeness is by construction"; the eigenvalues are those of the resting map $A$; last column is the "share of the variance of each mode's coordinate explained, within each pair of ages of the maximum and runner-up, by a linear function of the top three earlier values $M, S, T$ (756,592 steps from step 3 on, of 131,072 held-out inputs)".

### Table 3 (p. 13)

"Each neuron's explicit formula, alone at every step, keeps 90.6–99.5% of the network's answers but only 46–95% of its mistakes." 131,072 held-out inputs, 6,172 network mistakes. "Replacing a neuron by its mean at each step keeps 13–70%."

| neuron | agreement, formula | mistakes kept, formula | agreement, zeroed |
|---|---|---|---|
| $n_4$ | 99.5% | 95% | 18% |
| $n_2$ | 98.2% | 85% | 18% |
| $n_6$ | 97.3% | 70% | 14% |
| $n_7$ | 96.9% | 66% | 10% |
| $n_9$ | 96.8% | 80% | 67% |
| $n_1$ | 96.7% | 61% | 15% |
| $n_8$ | 96.2% | 53% | 18% |
| $n_{11}$ | 95.3% | 57% | 11% |
| $n_5$ | 94.2% | 57% | 21% |
| $n_3$ | 93.8% | 56% | 55% |
| $n_{15}$ | 93.7% | 53% | 64% |
| $n_{13}$ | 93.3% | 57% | 30% |
| $n_{14}$ | 93.3% | 62% | 20% |
| $n_0$ | 92.4% | 50% | 19% |
| $n_{12}$ | 90.6% | 46% | 13% |
| $n_{10}$ | 90.6% | 52% | 13% |

### Table 4 (p. 14)

"Only near copies of the weights keep 98% of the network's answers and 80% of its mistakes."

| account | numbers | worst neuron alone | all together | mistakes kept | margin-matched correct kept |
|---|---|---|---|---|---|
| rule: answer the true second largest | 0 | — | 95.3% | 0% | 100% |
| network weights, fp16 (10-bit binary significand) | 272 | — | 99.8% | 97.7% | 98.1% |
| network weights, 3 significant decimal digits | 272 (4,350 bits) | 95.7% | 98.4% | 82.3% | — |
| network weights, 2 significant decimal digits | 272 | — | 67.9% | 37.1% | 48.7% |
| explicit formulas (Table 3) | 3,904 | 90.6% | 95.7% | 23.9% | 92.4% |
| explicit formulas, clock-compressed | 784 | 63.8% | 50.4% | 26.6% | — |
| explicit formulas with history terms | > 3,904 | 93.5% | 95.6% | 41.4% | — |
| full account | 285 | 99.5% | 99.6% | 95.6% | — |
| full account, sparse (8 of 9 modes per neuron), refit | < 285 | 96.8% | 97.4% | 76.9% | — |
| full account, levels 1, refit (float) | 283 | 97.5% | 98.5% | 86.1% | — |
| full account, one fundamental, refit (float) | ≈ 280 | 97.1% | 98.1% | 82.7% | — |
| the same, coarse grids | ≈ 280 | 95.9% | 96.7% | 72.2% | — |
| full account, levels, damping, angles, echo idealised, refit (float) | ≈ 280 | 94.8% | 96.5% | 72.9% | — |
| idealised algorithm | 170 | 33% (worst comparator) | 93.2% | 6.0% | 90.0% |
| idealised algorithm, network's clock-fit thresholds | 170 | 78% (worst comparator) | 92.3% | 4.9% | — |
| idealised algorithm, refit to the network | 170 | 26% (worst comparator) | 87.8% | 8.9% | — |

Caption notes: network weights at 3 and 2 significant digits on 200,000 inputs; history terms, sparse and tied-fundamental accounts on 20,000 inputs; levels set to 1 and idealised-and-refit on 19,776 development inputs; other rows on 131,072 held-out inputs. "For the idealised rows, only the comparators are replaced one at a time." "On 20,000 inputs (about 940 mistakes), mistakes kept near 80% has a binomial standard error of about 1.3 points." "Numbers exclude the 160 readout weights, which every account borrows." "When agreement collapses, mistakes kept means little: the clock-compressed formulas keep more mistakes than the full formulas at half their agreement."

## 5. The paper's figures

Values marked "by eye" are my readings of the plots.

**Figure 1 (p. 3).** Title: "The comparators sort each input by the thresholds it passes, the recorder records the inputs near the top two, the tape carries each record by age, and the readout reads it by position." A flow diagram. Input $x_t$ enters a box "Comparators: $n_1\ n_4\ n_6\ n_7\ n_8$ compare $x_t$ with thresholds near the top two so far, read from the tape". Four outcome boxes (the four rows of the table in 2.1). The first three feed a box "Tape: memory neurons and $n_0$ carry each record and the maximum forward, rotating them at every step"; the fourth ("$n_5$ stays off, and $x_t$ drops out of the state") feeds nothing. The tape feeds "Readout: each record raises the logit mainly at its input's position; the last input is read directly from the comparators". A dashed arrow from the tape back to the comparators is labelled "thresholds are read from the tape". A legend lists the roles: $n_4$ max detector; $n_1\ n_6\ n_7\ n_8$ other comparators; $n_5$ recorder; $n_9$, $n_0$ timers; $n_2\ n_3\ n_{10}$–$n_{15}$ memory.

**Figure 2 (p. 4).** Title: "In one sequence, the max detector fires on the maximum and the recorder records the top three one step after each arrives."
(a) Bar chart of the ten inputs: maximum 1.58 at step 2, runner-up 0.57 at step 3, third largest 0.42 at step 4; the others by eye −1.23, −0.65 (steps 0, 1), 0.24, −0.40, −0.38, −0.21, −0.29 (steps 5–9).
(b) Grid of every neuron's gate at every step (filled = on), grouped by role, with $n_5$'s records printed: 19.5, 5.8, 0.3 at steps 3, 4, 5. By eye:

| neuron | on at steps |
|---|---|
| $n_4$ | 2 |
| $n_7$ | 0, 1, 3–9 |
| $n_6$ | 0, 1, 4–9 |
| $n_1$ | 0, 1, 3–9 |
| $n_8$ | 0, 1, 4–9 |
| $n_5$ | 0, 3, 4, 5 |
| $n_9$ | 0, 3 |
| $n_0$ | 2, 3, 5–9 |
| $n_2$ | 2–9 |
| $n_{12}$ | 1–9 |
| $n_{13}$ | 0, 2–9 |
| $n_{14}$ | 0, 1, 3–9 |
| $n_3$ | 0–3, 5, 6, 8, 9 |
| $n_{10}$ | 1–7 |
| $n_{11}$ | 1–6, 8, 9 |
| $n_{15}$ | 0–4, 6–9 |

So at step 2 (the maximum) all four other comparators are off; at step 3 (the runner-up) $n_6$ and $n_8$ are off and $n_7$, $n_1$ on; at step 4 (the third) all four are on. Caption: "The max detector fires at step 2; $n_9$ fires at step 3 and $n_0$ is off at step 4; the recorder $n_5$ writes records of 19.5, 5.8 and 0.3 at steps 3–5." Text (p. 3): the third largest "is recorded, barely, at step 5. Other inputs are not recorded, and the readout of the final state puts the highest logit at position 3."

**Figure 3 (p. 6).** Title: "Each comparator's threshold swings around the top two as the maximum ages, following one clock." Five panels ($n_4$, $n_7$, $n_6$, $n_1$, $n_8$); x: age of the maximum, 1–9; y: threshold position $(\theta_j - x_{2\mathrm{nd}}^{<t})/(x_{\max}^{<t} - x_{2\mathrm{nd}}^{<t})$, from about −0.2 to 1.6. Circles: measured medians "over arrivals at steps 2–9 with the maximum more than 0.2 above the runner-up ($2^{20}$ sequences)". Solid blue: the four-number clock fit of the thresholds computed from the weights ($R^2 \ge 0.993$); it passes through the circles. Dashed orange: the idealised algorithm (Table 1). The circles, from my reproduction (they match the plot; the caption prints only $n_1$'s 1.42), and the dashed curves, computed by me from Table 1:

| age of the maximum | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| $n_4$ network | 0.99 | 0.98 | 0.95 | 0.86 | 0.73 | 0.63 | 0.62 | 0.70 | 0.82 |
| $n_4$ idealised | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| $n_7$ network | 1.02 | 1.02 | 1.01 | 0.87 | 0.68 | 0.56 | 0.55 | 0.66 | 0.84 |
| $n_7$ idealised | 0.79 | 0.80 | 0.76 | 0.71 | 0.69 | 0.71 | 0.75 | 0.80 | 0.83 |
| $n_6$ network | −0.04 | 0.70 | 1.04 | 1.19 | 1.08 | 0.81 | 0.50 | 0.32 | 0.36 |
| $n_6$ idealised | −0.03 | 0.74 | 0.78 | 0.80 | 0.79 | 0.77 | 0.73 | 0.71 | 0.70 |
| $n_1$ network | 0.85 | −0.01 | 0.29 | 0.82 | 1.30 | 1.42 | 1.13 | 0.59 | 0.07 |
| $n_1$ idealised | 0.62 | 0.29 | 0.37 | 0.56 | 0.75 | 0.85 | 0.80 | 0.63 | 0.43 |
| $n_8$ network | 0.06 | 0.38 | 0.07 | 0.06 | 0.42 | 0.97 | 1.42 | 1.49 | 1.16 |
| $n_8$ idealised | 0.25 | 0.49 | 0.36 | 0.40 | 0.58 | 0.82 | 0.98 | 0.98 | 0.83 |

**Figure 4 (p. 7).** Title: "An input that passes no threshold reaches only the recorder, which records it when it exceeds about 0.7 of the runner-up so far."
(a) Bars of $W_{hh} G_t w$, "what an arriving input sends to each neuron at the next step, per unit", for two gate patterns at arrival. Blue ($n_1, n_6, n_7, n_8$ all on): +18 on $n_5$, 84 on $n_9$ (drawn gray and clipped), about 0 elsewhere. Light purple ($n_8$ off, "as for about half of new runners-up"): bars on every neuron. My computation of $W_{hh} G_t w$ for that pattern matches the plot: $n_0$ −10.2, $n_1$ −1.6, $n_2$ +1.9, $n_3$ −13.5, $n_4$ −1.0, $n_5$ +7.5, $n_6$ −1.0, $n_7$ +2.1, $n_8$ +7.7, $n_9$ 70.9 (gray), $n_{10}$ −2.6, $n_{11}$ −2.0, $n_{12}$ −3.4, $n_{13}$ −3.6, $n_{14}$ −1.2, $n_{15}$ −9.1. Annotation: "$n_9$ (gray): 84 and 71; off unless $n_4$ fired".
(b) Fraction of inputs after which $n_5$ is on, against input / runner-up so far (0.4 to 1), "for inputs arriving at steps 2–8 below the runner-up so far", in three bands of maximum − runner-up so far (below 0.25, 0.25–0.75, above 0.75; $2^{21}$ sequences). Three sigmoids: about 0.02–0.04 at ratio 0.4, half-way points by eye at about 0.78, 0.75 and 0.72 (wider gap, earlier turn-on), about 1 from ratio 0.9. A dashed vertical line at 0.71 is labelled $18x_t = 12.8x_{2\mathrm{nd}}^{<t}$.

**Figure 5 (p. 7).** Title: "The tape carries each record to its input's position; older records fade faster than the linear map predicts." Two heat maps of final logits per unit of a recorded input; rows = position of the recorded input (2–8), columns = logit position (0–9); colour bins at ±10, ±25, ±40, ±55; own position outlined and printed.
(a) "Derived from the weights, carried by the resting map $A$": diagonal 72, 76, 71, 70, 67, 62, 62. Only the diagonal is printed. My reconstruction (formula in 8.1) reproduces the diagonal and every colour bin, so the full panel is, to rounding:

| recorded input at position | logit 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | −44 | −31 | **72** | 19 | −14 | 3 | −5 | 10 | 13 | −23 |
| 3 | −34 | −36 | −27 | **76** | 18 | 1 | 8 | 0 | 14 | −20 |
| 4 | −26 | −30 | −31 | −23 | **71** | 20 | 9 | 15 | 9 | −12 |
| 5 | −8 | −26 | −29 | −26 | −15 | **70** | 20 | 9 | 15 | −10 |
| 6 | 3 | −8 | −24 | −24 | −19 | −12 | **67** | 21 | 9 | −9 |
| 7 | −1 | 1 | −6 | −17 | −15 | −15 | −9 | **62** | 20 | −9 |
| 8 | −4 | −2 | −2 | −6 | −9 | −11 | −12 | −6 | **62** | −8 |

(b) "Measured on recorded third-largest inputs that arrived after the top two ($2^{20}$ sequences)": diagonal 39, 55, 62, 67, 65, 61, 62; the off-diagonal pattern is like (a) but weaker. "Positions 0, 1 and 9 are omitted because a third largest that arrives after the top two cannot sit there or, at position 9, is never recorded before the end."

**Figure 6 (p. 8).** Title: "The max detector makes a new maximum's own effect negative, and a new maximum switches the other comparators off."
(a) The final maximum's own effect per unit against its age 0–9: measured median (black), quartiles (gray band), and "derived from the weights with the commonest gate pattern at each step" (blue). By eye: −51 at age 0, −29 at age 1, then between −4 and −8 for ages 2–9 (black); the blue curve is within about 3 of the black one and reaches about 0 at age 9; the quartile band widens with age, to roughly −32..0 at age 9.
(b) Scatter of the measured median final own effect against the derived one, "per group of sequences with the same positions of maximum and runner-up", blue = runner-up arrived first, gray = runner-up arrived later; both axes −40..0. Most points lie on the diagonal between −12 and +2; two blue points near (−28, −27). "The few gray outliers, where the derivation predicts −17 to −38 and the network gives −6 to −22, are not explained by the derivation, which assumes the commonest gate pattern at each step."
(c) Bars: a maximum's mean own effect split by neuron group, for three cases.

| group | a maximum at the end, ages 2–9 | old maximum, had the step been resting | old maximum, after a new maximum |
|---|---|---|---|
| $n_1\ n_6\ n_7\ n_8$ | −72 | −46 | −3 |
| $n_4$ | 0 | 0 | −3 |
| timers, recorder | 0 | +5 | +10 |
| memory | +54 | +26 | +26 |
| total | −18 | −16 | +30 |

The caption notes these are "means, which a few large negative values pull below the medians of (a)".

**Figure 7 (p. 11).** Title: "Clock angles and the echo cost few answers; calibration, the history modes, clipping and how the comparators read the modes carry the network's behaviour." A dot plot on 19,776 development inputs (931 network mistakes): agreement with the network (circle), mistakes kept (diamond) and task accuracy (gray tick) "when one group of the full account's constants is idealised or removed, with no refit". Text values where the paper gives them, otherwise by eye (≈):

| change | agreement | mistakes kept | task accuracy |
|---|---|---|---|
| none (full account) | 99.65% | 96.7% | 95.23% |
| harmonics tied at 2×, 3× of 41.6° | at least 99.4% (≈ 99.6) | ≈ 96.5 | ≈ 95.2 |
| echo → 0 | ≈ 99.6 | ≈ 96 | ≈ 95.2 |
| main clock angle 42.75° → 42.4° | at least 99.4% | ≈ 93 | ≈ 95.2 |
| all clock angles → ideal | 98.3% | 84% | ≈ 94.8 |
| harmonic damping $\rho_2 \to \rho_1^2$, $\rho_3 \to \rho_1^3$ | 98% | 82% | ≈ 95.0 |
| tape input weights → 0 (except $n_{14}$) | 97% | 78% | ≈ 94.4 |
| main clock radius → 1 (no damping) | 92% | ≈ 64 | ≈ 90.4 |
| alternating mode → 0 | 91% | ≈ 69 | ≈ 89.2 |
| reset transient → 0 | 89% | ≈ 59 | ≈ 88 |
| comparator gains → 10.5, 11, 13, 12 | 89% | ≈ 65 | ≈ 87.4 |
| both level eigenvalues → 1 | 89% | ≈ 59 | ≈ 87.4 |
| tape clipping not written back | 60% | ≈ 40 | ≈ 59 |
| short-lived rotation → 0 | 51% | ≈ 37 | ≈ 50.5 |
| comparators read only levels + main clock | 36% | ≈ 23 | ≈ 36 |

$\rho_1, \rho_2, \rho_3$ are the radii of the main clock and the two harmonics. How the changes are made (p. 15): they "set a mode's eigenvalue to zero or a constant to its ideal value, with no refit; 'comparators read only the levels and main clock' masks their rows of $V$ for reading only; 'tape clipping not written back' writes the tape neurons' pre-activations, not their rectified values, into $m_t$".

## 6. Negative results and failed searches

1. **No compact account of the mistakes.** "we cannot yet explain its mistakes compactly" (p. 2); the third question's answer is "largely negative" (p. 2).
2. **The bar** (p. 13), set before searching: at least 95% of answers with any one neuron replaced, 98% with all replaced, 80% of the network's mistakes. It is the level "that the network's own weights rounded to three significant decimal digits just pass" (95.7% / 98.4% / 82.3%). Only near copies pass: fp16 weights (99.8%, 97.7%), the full account (99.5% / 99.6% / 95.6%), "full account, levels 1, refit (float)" (97.5% / 98.5% / 86.1%) and "full account, one fundamental, refit (float)", narrowly (97.1% / 98.1% / 82.7%, standard error 1.2 points).
3. **Idealised algorithm.** Agreement 93.2%, below the 95.3% of the rule "answer the true second largest"; 6.0% of mistakes against 90.0% of margin-matched correct answers; comparators in place together 28%; worst comparator alone 33%. It "is a caricature, not a replacement" (p. 1).
4. **Idealised algorithm with the network's clock-fit thresholds**: 92.3% together, 4.9% of mistakes, worst comparator 78%. **Refit to the network's logits**: 87.8%, 8.9%, worst comparator 26% ("that refit may simply have found a worse optimum"). "Neither of two variants makes it more faithful."
5. **The network's own clock-fit thresholds** (Fig. 3's four numbers per comparator, $R^2 \ge 0.993$), one comparator at a time, keep only 78–88% of answers (max detector's 99.2%).
6. **Recorder formulas** (2.6): gate only 98%, value too 78%.
7. **Explicit formulas**: together 95.7% of answers and 23.9% of mistakes (margin-matched correct 92.4%); only seven of sixteen beat the rule's 95.3% alone; worst neuron alone 90.6%; integrated accuracy 97.57% (sampled 97.59%), 2.3 points above the network, so the estimate "fails" ARC's criterion.
8. **Clock-compressed explicit formulas** (784 numbers): explain 89% of the gains' variance but keep 50.4% of answers together, worst neuron 63.8%, 26.6% of mistakes.
9. **Explicit formulas with history terms** (previous input, leaky input sum, third value's age; more than 3,904 numbers): 95.6% of answers, worst neuron 93.5%, 41.4% of mistakes.
10. **Full-account compressions.** Sparse (8 of 9 modes per neuron, refit): 96.8% / 97.4% / 76.9%, "the closest" of those that fall short. (The sparse accounts "keep $k$ of the nine mode groups in each ordinary neuron's row of $V$, refit for 3,000 Adam steps on joint and single-neuron KL, and keep a dense writeback", p. 16.) One fundamental with coarse grids: 95.9% / 96.7% / 72.2% ("Rounding that variant to coarser grids, with refits after each stage, breaks it"). Levels, damping, angles and echo idealised, refit: 94.8% / 96.5% / 72.9%.
11. **Weights at 2 significant decimal digits**: 67.9% of answers, 37.1% of mistakes, 48.7% of margin-matched correct answers.
12. **Removals in the full account** (Fig. 7): none raises accuracy above the network's 95.29% or toward the idealised 96.6%; the largest losses are comparators reading only levels and main clock (36%), short-lived rotation removed (51%), clipping not written back (60%).
13. **A description by the top three and their positions cannot keep more than about 64% of the mistakes** (redraw experiment, E8), and the formulas reach 24%.
14. **The resting map is not the typical step**: exact resting pattern on 17% of steps whose input is below the runner-up so far, all thirteen on 34%.
15. **Old records are not predicted by the linear map**: +55 and +39 measured at positions 3 and 2 against +76 and +72.
16. **Fig. 6b outliers**: derivation −17 to −38, network −6 to −22, "not explained by the derivation".
17. **History terms and the third value's age were not separated** (p. 13).
18. **Most of the full account is uninterpreted**: "most of its 285 numbers remain uninterpreted" (p. 2); "its read and write matrices and the formulas' age tables remain uninterpreted numbers" (p. 15).
19. Background: ARC "could not match most of the trained network's neurons to hand-built ones" (p. 1).

## 7. The paper's own statements of uncertainty and open questions

- Title: the mistakes "have resisted a compact account so far".
- p. 2: roles are named for "seven neurons individually and of the nine tape neurons as a group".
- p. 5: the idealised coefficients "are calibrated, partly to the network, not derived from its weights"; the correlations of its emitted neurons hold "partly by construction".
- p. 5: the resting map "is a reference, not the typical step".
- p. 6: the recorder formulas "capture when the recorder turns on but not exactly what it writes".
- p. 7: faster fading of old records is "consistent with some tape neurons being clipped at zero on most steps" (not tested directly).
- p. 8: the gray outliers of Fig. 6b "are not explained by the derivation".
- p. 9: the full account "is a reparametrisation of the weights, not a compression or an independent explanation"; Table 2's closeness of idealised gains and angle "is by construction".
- p. 10: that the clock-fit thresholds keep only 78–88% is "consistent with the comparators' reads of the other modes mattering" (not shown).
- p. 12: of the redraw effect, "Part of this is fragility, but not most".
- p. 13: the 28% for the idealised comparators is "possibly because its thresholds and emission were fitted together and compensate for each other"; the refit "may simply have found a worse optimum"; "we did not separate the history terms from the third value's age"; the integral "is a mechanistic estimate of the formulas' accuracy, not of the network's".
- p. 14: sampling errors of 1.2–1.3 points on mistakes kept near 80% with 20,000 inputs; "When agreement collapses, mistakes kept means little".
- p. 15: "What we do not have is a compact account of the network's mistakes." "An explanation in ARC's sense would need that history and its reads in a few rules, and an integrated estimate that lands on 95.3%."

## 8. My assessment (all of this section is **[mine]**)

### 8.1 What I could reproduce

Section 9 lists the quoted quantities that I recomputed from the weights (38 rows); nearly all match to the printed precision. In particular the program printed in Appendix B does reproduce the headline numbers of the idealised algorithm (96.57% accuracy, 93.24% agreement, 27.9% for the four comparators together), and two figures can be reconstructed from simple formulas:

- Fig. 5a is $E_{q,p} = W_{out}[q]\;A^{8-p}\;G_{\mathrm{rec}}\,W_{hh}\,(G_{\mathrm{rest}}\,w)$, with $G_{\mathrm{rec}}$ the resting pattern plus $n_5$ on. It gives 72.0, 75.7, 71.0, 69.5, 66.6, 61.5, 62.2 on the diagonal and 17.9–20.7 one position later.
- The "own effect" of Fig. 6 is the Jacobian entry $\partial\,\mathrm{logit}_p/\partial x_p = W_{out}[p]\,\big(\prod_{t=p+1}^{9} G_t W_{hh}\big)\,G_p\,w$ along the actual gates. With that definition I get Fig. 6c's green bars (−72.2, −0.1, −0.1, +54.2, total −18.1) and Fig. 6a's medians (−51.8, −28.2, −5.1, −6.4, −6.4, −8.4, −9.0, −6.8, −4.7, −3.7 for ages 0–9). The paper never states this definition.
- Because the network is positively homogeneous, $\mathrm{logits} = J x$ exactly (checked to $10^{-12}$), so own effects and cross effects add up to the logits.
- Two of the idealised program's hand-set constants are single weight products: the recorder's 12 is $W_{hh}[5,4]\,w_4 = 12.06$ and $n_9$'s 1.5 is $W_{hh}[9,4]\,w_4 = 1.48$, the drive of $n_5$ and $n_9$ per unit of $n_4$'s excess $x - \theta_4$.

I did not check anything about the full account, the explicit formulas, Tables 3–4 (except the idealised row), Fig. 7, the redraw experiments, the rounding experiments or the oracle interventions.

### 8.2 Inconsistencies and loose statements

1. **Three different descriptions of what the recorder writes.** (i) Idealised: $12\,[x - S]_+$, the excess over the runner-up so far. In the labels of Sec. 7 (values after the arrival) that is $12M - 12T$ after a new maximum and $12S - 12T$ after a new runner-up. (ii) Sec. 4.2: "about 12 times its value", $7.7x_t - 2x_{2\mathrm{nd}}^{<t}$, $18x_t - 12.8x_{2\mathrm{nd}}^{<t}$. (iii) Sec. 7: $12.14M - 3.19S - 3.58T$ and $0.89M + 7.81S - 3.54T$. For a new runner-up the coefficient on the displaced runner-up is −2 in (ii) and −3.54 in (iii), and (iii) has an extra $+0.89M$. The paper notes the relabelling of $S$ but not this difference, and does not say which age cell the Sec. 7 coefficients come from. From the weights, the leading coefficients are exact: $W_{hh}[5,4]\,w_4 = 12.06$, $W_{hh}[5,7]\,w_7 = 7.81$, $W_{hh}[5,8]\,w_8 = 10.46$, and the four comparators together give 18.1 (18.0 with the tape neurons). The subtracted terms (−2, −12.8, −3.19, −3.58, −3.54) are not derived anywhere in the paper.
2. **"18 times its value minus 8–13 times the earlier one"** (p. 8) describes only the case where the later of runner-up and third passes no threshold. A later runner-up that passes $n_8$'s threshold is recorded as $7.7x_t - 2x_{2\mathrm{nd}}^{<t}$ by the paper's own p. 6, and that is "about half of new runners-up".
3. **Misfire rates.** The paper: "on new runners-up within 0.05 of the maximum it misfires on 27% at age 1 and 96–98% at ages 5–7". I get 26.7% and 96.2%, 98.0%, 97.9% only when I restrict to the sequence's final maximum and final runner-up. Over every new-runner-up event at steps 2–9 the rates are 16.9% at age 1, 29.9% at age 2, 46.2% at age 3, 80.7% at age 4 and 92.1%, 96.1%, 96.7% at ages 5–7. The sentence does not say which is meant.
4. **Two statistics about new maxima use different step ranges.** "a new maximum passes each of them on 93–95% of arrivals" matches arrivals at steps 2–9 (92.8–95.2%; steps 1–9 give 88.2–89.9%). "switches $n_1$, $n_6$, $n_7$ and $n_8$ all off at once (on 86% of arrivals)" matches steps 1–9 (85.9%; steps 2–9 give 89.8%).
5. **"The network's mistakes are near ties"** (title of Sec. 6). By the paper's 0.05 criterion 65% are; 35% are not.
6. **Means quoted as if typical.** "the four other comparators give the maximum an own effect of −72 per unit and the memory neurons +54" are means (total −18, pulled down by a tail). The median own effect of a maximum aged 2–9 is only −4 to −9 per unit. Only the caption of Fig. 6c says "means". So the maximum's negative own effect at ages 2 and above is a small difference of two large terms.
7. **"the runner-up's trace raises the maximum's logit at least as much as its own"** (p. 8) has no number or figure. With the Jacobian I find medians $\partial L_{\max}/\partial x_{2\mathrm{nd}} = +37.4$ and $\partial L_{2\mathrm{nd}}/\partial x_{2\mathrm{nd}} = +42.7$; "at least as much" holds on 50% of sequences. The conclusion drawn from it does hold when the margin $L_{2\mathrm{nd}} - L_{\max}$ (median 12.7) is split by input: the terms in $x_{\max}$ contribute +16.2, the terms in $x_{2\mathrm{nd}}$ −0.1, all other inputs −2.1 (each a median, so they need not add up).
8. **"nearly every new maximum" and "most new runners-up" are recorded** (p. 6). The split is by sign: $n_5$ is on one step after 99.9% of positive new maxima and 5.0% of negative ones, and after 99.9% of positive new runners-up and 7.2% of negative ones (89.9% and 77.8% overall). The paper never discusses negative values in the network; it only builds positivity into the idealised algorithm ("exact positive maximum"), the formulas ("three largest positive earlier inputs") and the integral (which treats "zero or one positive earlier input" as separate cases).
9. **"small differences in the thresholds matter"** (p. 5). The idealised thresholds differ from the network's by up to 0.57 of the gap ($n_1$ at age 6: 0.85 against 1.42), 0.51 ($n_8$ at age 8), 0.39 ($n_6$ at age 4), which is not small. Replacing one comparator by its idealised version keeps 33.5% ($n_7$), 34.9% ($n_1$), 39.2% ($n_6$), 48.8% ($n_8$) in my run.
10. **"one damped clock"** (p. 5) against the fit actually reported, which is undamped: the basis is $\{1, \cos 0.74a, \sin 0.74a, r\}$. The main clock's radius is 0.951 per step, a factor 0.64 over nine steps, and the measured swings in Fig. 3 do not visibly shrink with age ($n_1$ goes from −0.01 at age 2 to 1.42 at age 6; $n_8$'s largest value is at age 8). Also, $r = [a = 0]$ is nonzero at age 1 only, so that parameter fits age 1 exactly and the "$R^2 \ge 0.993$ over nine ages" is three numbers fitted to eight points.
11. **Role count.** p. 15 says "what we interpret is ... every neuron's role"; p. 2 says seven neurons individually and nine as a group. For $n_3$, $n_{10}$–$n_{13}$ and $n_{15}$ the paper gives nothing individual beyond a row of `EMIT`.
12. **Fig. 1 is not exhaustive.** Among new runners-up at steps 2–9, 52.4% pass $n_8$'s threshold and not $n_7$'s, 32.5% pass none and 5.9% pass $n_4$'s. The remaining 9% or so pass other combinations that the figure does not list.
13. **Mistakes kept by the idealised algorithm**: "6%" in the text, 6.0% in Table 4; the README next to the PDF reports 5.2% on another sample, and I also get 5.2% (32,768 inputs, standard error about 0.6 points).
14. Minor: the last column of Table 2 gives "0.81, 0.91" for the two history modes without saying which is which.

### 8.3 Claims that look under-supported

1. **Which level holds which value.** Table 2 names the 1.0098 mode "maximum" and the 0.9736 mode "runner-up", and says the last column "checks the names". That column regresses each coordinate on all of $M, S, T$ jointly, within age cells. $R^2 = 0.999$ and $0.986$ show that each level is a linear function of the top three; they do not show that one is $M$ and the other $S$. No coefficients are given.
2. **That the history modes carry inputs outside the top three.** The $R^2$ evidence (0.81–0.91) conditions on the ages of the top two only. A mode that carried the third value's record, with a phase set by the third value's age, would also leave unexplained variance, and the paper itself says the formulas "omit the third value's age". The direct evidence is one constructed pair (+6.5 logits from the short-lived rotation) and the removal (51%). The redraw experiment shows that answers depend on history, not which mode carries it.
3. **The correspondence between idealised quantities and modes is not one to one.** The idealised algorithm gives the maximum's clock a fundamental only and the runner-up's clock a fundamental plus two harmonics. In the network all records share one set of modes ("no separate mode"). Table 2 says so, but the heading of Sec. 5 ("the idealised constants are separate parameters") reads stronger than that.
4. **$n_0$ "carries records at age 2"** and **"the readout reads it from the comparators' gates"** (runner-up at the last position) are stated without measurement. The last-position readout is analysed only through $n_4$'s weight −4.99.
5. **The function of the threshold swings is not explained.** The paper reports their cost (misfires) but not why the thresholds swing. Its own numbers suggest a reading it does not state: the four comparators give an aged maximum its negative own effect (−72 per unit on average), and their activations depend on the maximum through the age-dependent gains $\alpha_j(a)$, so the staggered swings are an age code for the maximum that the readout can use to subtract at the maximum's position. I have not tested this.
6. **Replacement inside the network is a very harsh test**, because the recorder and $n_9$ read the comparators with gains of 18 and 84 per unit of input. A threshold error of 0.3 of the gap changes a record by several times the gap. The 28%, 33% and 78–88% figures therefore measure this gain as much as the quality of the description.
7. **Fig. 5b measures only third-largest inputs** that arrived after the top two. The runner-up's own record is not measured that way. In my Jacobian check the runner-up's own effect has median +42.7 per unit, well below the +62 to +76 of Fig. 5a.
8. **The intercept $b$** in $\theta_j \approx g_{\max}x_{\max}^{<t} + g_{2\mathrm{nd}}x_{2\mathrm{nd}}^{<t} + b$ is not reported. The network has no bias, so $b$ can only be the mean effect of the other inputs.
9. **756,592 steps** in the caption of Table 2 is not explained. It matches the number of steps from 3 on with at least two positive earlier inputs: $131{,}072 \times \sum_{t=3}^{9}\big(1 - (1 + t)/2^t\big) = 756{,}736$ expected.
10. **Fig. 6c's "old maximum" bars are not defined.** The paper does not say which step's state is read out for "the old maximum when a new maximum arrives", nor how the counterfactual "had the step been resting" is built. The relabelling numbers (+30 against −16) rest on these bars. Note also that in them the memory neurons read the old maximum at +26 in both cases; the jump comes from the comparators (−46 to −3), which fits "switches all off" but not the wording "exposes the memory neurons' positive reading" as a change in the memory term.
11. **The idealised algorithm says what suffices, not what the network does.** It was fitted to the task loss through the network's readout, with a small pull (0.02) toward the network's states. The paper is clear about this.

### 8.4 What the paper does not contain

- The words "score register", "shift register", "wave", "circle", "orbit", "plane" and "centre" do not occur. Its vocabulary is tape, record, trace, level, main clock, harmonics, history modes, echo, reset transient. "Bias" occurs only in "without biases".
- No analysis of $W_{out}$ beyond own effects (no account of its structure across positions).
- No treatment of sequences whose top two are negative, and no statement of how negative values are held.
- No per-neuron account of the memory neurons other than $n_2$ (and $n_{14}$'s input weight).
- No statement of how "own effect" is measured (see 8.1).

### 8.5 Bearing on the off-centre-circle idea

The paper does not discuss it, but its two accounts both have the structure "centre set by stored values, radius set by stored values":

- **Idealised program.** With the maximum alone ($S = 0$, no reset) a tape neuron is

  $$h_i = M\,\big[\,c_i + a_i\cos\phi_M + b_i\sin\phi_M\,\big]_+ = M\,\big[\,c_i + \rho_i\cos(\phi_M - \phi_i)\,\big]_+ ,$$

  a circle with centre $c_i M$ and radius $\rho_i M$. From `EMIT`, $c_i$ = 10.5, 12.5, 3.0, 10.5, 10.0, 13.0, 5.5, 10.5, 8.0 and $\rho_i$ = 3.54, 1.12, 1.00, 4.53, 4.61, 5.22, 0.50, 3.50, 4.24 for $n_0, n_2, n_3, n_{10}, n_{11}, n_{12}, n_{13}, n_{14}, n_{15}$, so $\rho_i/c_i$ = 0.34, 0.09, 0.33, 0.43, 0.46, 0.40, 0.09, 0.33, 0.53. All are below 1: the maximum's circle alone is never clipped. The runner-up moves the centre by $S$ times the second column of `EMIT` (−5.5, 7.5, 1.5, −5.5, −6.0, −4.0, 1.5, −1.5, −1.5) and adds its own circles with radius proportional to $S$. Scale equivariance forces both centre and radius to be proportional to the stored values.
- **Network (full account).** The centre is carried by two real eigenmodes of the resting map with eigenvalues near 1 (1.0098 and 0.9736), which play the part of the missing bias. The rotation is the main clock with radius 0.951 per step, so the orbit is a slow inward spiral whose centre drifts (+1% and −2.6% per step), plus harmonics at about 2× and 3× 41.6°.
- **Clipping still happens and matters.** Paper: the exact resting pattern holds on only 17% of the steps whose input is below the runner-up so far, and not writing the clipping back keeps only 60% of answers. My count, on steps with two positive inputs so far, of how often each tape neuron is at zero:

  | | $n_0$ | $n_2$ | $n_3$ | $n_{10}$ | $n_{11}$ | $n_{12}$ | $n_{13}$ | $n_{14}$ | $n_{15}$ | all nine on |
  |---|---|---|---|---|---|---|---|---|---|---|
  | network | 20.1% | 0.0% | 23.8% | 8.3% | 8.9% | 3.7% | 0.3% | 5.8% | 27.4% | 37.4% |
  | idealised program | 6.1% | 0.0% | 15.1% | 4.6% | 7.7% | 3.5% | 0.0% | 0.4% | 9.0% | 66.8% |

  So in the network the offset keeps $n_2$, $n_{13}$ and $n_{12}$ on almost always, but not $n_0$, $n_3$ and $n_{15}$. Those three are the ones that an input passing $n_8$'s threshold drives down (Fig. 4a).
- The paper does not test whether the level components are what keeps the tape neurons on, nor which stored value sets the centre for which neuron. That part of the idea is open as far as this paper goes.

## 9. Reproduction table (**[mine]**; `check_program_paper.py`)

| quantity | paper | mine |
|---|---|---|
| input weights of $n_4, n_1, n_6, n_8, n_7$ | +10.16, −10.56, −11.00, −12.31, −13.17 | 10.157, −10.565, −10.996, −12.312, −13.169 |
| $W_{hh}[2,2]$, $W_{hh}[2,4]$ | 0.97, 1.73 | 0.969, 1.731 |
| $W_{out}[9,4]$; times $w_4$ | −4.99; −50.7 | −4.992; −50.70 |
| eigenvalues of $A$ | 1.0098; 0.9736; 0.951 at 42.75°; 0.937 at 83.27°; 0.837 at 124.78°; 0.231; −0.928; 0.735 at 43.0°; 0.001 | 1.0099; 0.9735; 0.9509 at 42.75°; 0.9370 at 83.28°; 0.8373 at 124.78°; 0.2310; −0.9283; 0.7350 at 42.99°; 0.0010 |
| $W_{hh} G w$ to $n_5$, to $n_9$, largest other | 18.0, 84.3, at most 0.4 | 18.009, 84.335, 0.43 (on $n_0$) |
| $n_9$ with $n_8$ off (Fig. 4a) | 71 | 70.9 |
| Fig. 5a diagonal, positions 2–8 | 72, 76, 71, 70, 67, 62, 62 | 72.0, 75.7, 71.0, 69.5, 66.6, 61.5, 62.2 |
| Fig. 5a one position later | about +20 | 17.9–20.7 |
| constructed pair | answers 3 and 0; margin difference 5.5 | 3 and 0; 5.46 |
| accuracy | 95.3% | 95.31% ($2^{20}$ inputs, fp32) |
| answers third / maximum / smaller | 2.97% / 0.95% / 0.76% | 2.97% / 0.94% / 0.78% |
| errors within 0.05 of the runner-up; split | 65%; 48%, 15%, 2% | 64.8%; 48.3%, 14.8%, 1.6% |
| median margin, mistakes / correct | 1.0 / 8.5 | 1.04 / 8.57 |
| network right on runner-up/third near ties | 78% | 77.4% |
| network right without near ties | 98.4% | 98.49% |
| median margin of runner-up's logit over maximum's | +12 | 12.7 |
| runner-up at the last position | about a tenth | 10.0% |
| $n_9$'s gate = $n_4$'s gate one step earlier | 99.0% | 98.97% |
| new maximum switches all four comparators off | 86% | 85.9% (steps 1–9); 89.8% (steps 2–9) |
| new maximum passes each threshold | 93–95% | 92.8–95.2% (steps 2–9); 88.2–89.9% (steps 1–9) |
| input below the runner-up passes a threshold | at most 0.5% each | 0.01%, 0.48%, 0.50%, 0.01%, 0.31% for $n_4, n_1, n_6, n_7, n_8$ |
| new runners-up passing $n_8$'s threshold, not $n_7$'s | about half | 52.4% |
| resting pattern: all thirteen on / exact | 34% / 17% | 34.3% / 16.9% |
| final runner-up before the last step recorded | 99% | 99.2% |
| Fig. 3, $n_1$'s peak | 1.42 | 1.424 at age 6 |
| misfires at age 1; ages 5–7 | 27%; 96–98% | 26.7%; 96.2%, 98.0%, 97.9% (final top two only). 16.9%; 92.1%, 96.1%, 96.7% (all events) |
| maximum's own effect, age 0 and 1 | −50.7; about −29 (Fig. 6a) | −51.8 median (−50.7 through $n_4$ alone); −28.2 |
| Fig. 6c, maximum at the end aged 2–9 | −72, 0, 0, +54, total −18 | −72.2, −0.1, −0.1, +54.2, total −18.1 |
| idealised accuracy | 96.6% | 96.57% (32,768 inputs) |
| idealised agreement | 93.2% | 93.24% |
| idealised: mistakes kept | 6% | 5.2% |
| both right / only idealised / only network | 92.9% / 3.7% / 2.4% | 93.00% / 3.57% / 2.30% |
| runner-up/third near ties, idealised / network | 91% / 78% | 90.2% / 77.7% |
| no near tie, idealised / network | 97.3% / 98.4% | 97.57% / 98.54% |
| correlations of emitted neurons | 0.96–1.00; 0.67–0.88 | 0.955–0.998; 0.672–0.877 |
| idealised comparators together | 28% | 27.9% |
| worst idealised comparator alone | 33% | 33.5% ($n_7$) |
| numbers in the program | 170 | 170 numeric literals |
