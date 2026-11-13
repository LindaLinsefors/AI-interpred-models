#set document(title: "How a 432-parameter RNN finds the second-largest number")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.2cm), numbering: "1")
#set text(font: ("Libertinus Serif", "New Computer Modern"), size: 10.5pt)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.")
#show heading.where(level: 1): set text(size: 12.5pt)
#show heading: set block(above: 1.3em, below: 0.7em)
#show figure.caption: set text(size: 9.2pt)
#show figure.caption: set align(left)
#set figure(gap: 0.8em)
#show raw: set text(size: 9pt)
#show table: set par(justify: false)

#align(center)[
  #text(size: 16pt, weight: "bold")[How a 432-parameter RNN finds the second-largest number:\ an idealised algorithm, a faithful rewrite, and the gap between them]
  #v(0.4em)
  #text(size: 10pt)[Goodfire parameter-decomposition team. A short paper for readers outside the team, November 2026.]
]

#v(0.6em)
#block(inset: (x: 1.2em))[
  *Abstract.* Mechanistic interpretability tries to explain how a trained network computes its outputs, in terms a person can check. The Alignment Research Center (ARC) released AlgZoo, a set of tiny networks trained on algorithmic tasks, as test cases with no explanations attached. However, even these networks have resisted explanations that are both short and faithful to what the network actually does. Here we show that one of them, a 16-unit ReLU RNN that outputs the position of the second-largest of ten numbers, runs a recognisable core algorithm, and we measure how far that algorithm is from the network. The network approximately keeps the largest and second-largest values so far and a rotating "clock" whose angle tracks how many steps ago the largest value arrived. Four comparator units test each new input against thresholds that swing with that clock. A short idealised program built on these parts (171 numeric constants) solves the task more often than the network (96.5% against 95.3%), but it agrees with the network on only 93.2% of inputs and reproduces 5% of the network's mistakes. A rewrite of the network's own recurrence in the same named quantities agrees on 99.6% of inputs and reproduces 96% of the mistakes, but it is not shorter than the network and most of its coefficients have no interpretation. So the core algorithm is clear, while our attempts so far at a complete short explanation of the network's exact behaviour, including its mistakes, are largely negative.
]

= Introduction

A network that finds the second-largest of ten numbers, read one at a time, must keep more than a running maximum. When the sequence ends, it has to know which earlier input was second and where that input arrived. AlgZoo, from ARC, is a collection of small networks trained on such tasks and released without explanations#footnote[ARC, "AlgZoo: uninterpreted models with fewer than 1,500 parameters", https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/]. We study its network for this task, which ARC calls $M_(16,10)$. ARC described units 1, 2, 4, 6 and 7. We propose roles for all 16, although four of them (units 3, 10, 11 and 15) only as parts of a group.

We give two accounts. The *idealised account* is a short program that states the algorithm. The *full account* is the network's exact recurrence, rewritten in named quantities and rounded. Comparing them separates two questions: how the network solves its task, which has a clear answer, and what exactly the network computes, including its mistakes, which so far has no short answer.

= The task, the network and the tests

*Task.* The input is ten independent standard-normal numbers $x_0, dots, x_9$. The correct output is the position (0 to 9) of the second-largest.

*Network.* A one-layer ReLU RNN without biases, starting from $h_(-1) = 0$:
$ h_t = "ReLU"(w x_t + W h_(t-1)), quad "logits" = U h_9 . $
The input weights $w$ (16 numbers), the recurrent weights $W$ ($16 times 16$) and the readout $U$ ($10 times 16$) make 432 parameters, 272 of them before the readout. Only $h_9$ reaches the readout. The network is correct on 95.3% of inputs.

*Tests.* An account is a program that outputs a 16-number hidden state at every step.
- *Agreement:* the program's state replaces the network's at every step, the network's own readout $U$ turns the final state into an answer, and we count how often it equals the network's answer.
- *Mistakes reproduced:* among inputs the network gets wrong, the fraction where the program gives the same wrong answer.
- *Single-unit replacement:* the real network runs, but one unit is overwritten at every step by the program's value for that unit, and we count how often the network's answer survives.
- *KL divergence* from the network's output distribution to the program's, in nats.

The rule "answer the true second-largest" agrees with the network on 95.3% of inputs and reproduces none of its mistakes. So agreement alone is a weak test.

*Data.* Each number in this paper comes from one of four samples of fresh random inputs. Table 1 uses 131,072 inputs (our run of `code/verify.py`). The near-tie and readout statistics use 20,000 inputs, and the clock statistics 4,000 (`code/analysis.py`, `code/readout_stats.py`). The removal experiments in Section 5 and the threshold fits in Section 7 come from the original authors' notes (`code/accounts_working_notes.md`), on 19,776 inputs. Those inputs were held out from fitting the idealised account, but they were used to choose the rounding of the full account.

= The algorithm

#figure(image("figures/algorithm.svg", width: 100%), caption: [*The core algorithm, as roles of the 16 units.* At each step the input is compared with thresholds read from memory; the comparisons update memory; after the last step the readout turns stored ages into positions. Unit numbers are the network's.]) <fig-alg>

@fig-alg summarises the algorithm. We group the units by what they do. The *comparators* are units 1, 6, 7 and 8. The *pulses* are units 4, 5 and 9. The *wave units* are the other nine (0, 2, 3, 10–15).

*Memory.* Memory means a few quantities that the hidden state carries from step to step. They are not stored in dedicated units: they are a fixed linear re-coding of the 13 non-pulse units, comparators included. Memory holds approximately two *levels*: $M$, the largest value so far, and $S$, the runner-up (second-largest so far). It also holds a *clock*, a two-dimensional state that rotates by about 42° per step. A new maximum restarts the clock, so its angle tracks the *age* of the maximum, the number of steps since it arrived. Further oscillating components, which we call *history waves*, carry information about earlier candidates, including the runner-up.

*Comparisons.* Unit 4 fires when the input beats the maximum. Its output is roughly $10 dot max(x - M, 0)$, but its real threshold is not exactly $M$. The comparators output $g dot max(theta - x, 0)$ with gains $g approx 10.5, 11, 13, 12$, where each threshold $theta$ is set by memory, lies near $S$ and $M$, and swings with the clock. Unit 5 fires on the step after a record or a new runner-up, and unit 9 on the step after a record.

== The idealised account

The idealised account (`code/idealised_account.py`) makes this literal. It keeps the exact maximum and runner-up of the positive inputs,
$ M' = max(M, x), quad S' = max(S, min(M, x)), $
and gives each one a clock $z$ on the unit circle. Every step multiplies both clocks by $e^(0.74 i)$ (42.4°). A clock resets to 1 when its value is replaced, and on a record the old maximum's clock passes to the runner-up. The comparators are $h_j = g_j max(theta_j - x, 0)$ with
$ theta_j = S + (M - S)(A_j + B_j "Re" z_M + C_j "Im" z_M + D_j r), $
where $r$ marks a record on the previous step. Unit 4 is $10 max(x - M, 0)$, unit 5 is $12 max(x_(t-1) - S, 0)$, and unit 9 is $1.5 M r$. The nine wave units are a ReLU of a half-integer linear mix of 12 features: the two levels, the maximum's clock, and the first three powers of the runner-up's clock. The threshold coefficients and the mixing weights were fitted by gradient descent on training inputs. The loss was the task cross-entropy through the network's readout, plus 0.02 times the mean squared difference from the network's hidden states. The coefficients were then rounded. Because both levels start at zero, the program cannot locate a negative runner-up. This affects about 1% of inputs, those with at most one positive value.

== A worked example and the readout

#figure(image("figures/example_trace.png", width: 82%), caption: [*One input and the network's response.* (A) The inputs; the answer is position 1. (B) The real network's units at each step. Unit 4 fires at positions 0, 1 and 6, when a new maximum arrives, and the comparators are then silent. On other steps the comparators fire, most strongly at position 5 (input $-1.0$).]) <fig-trace>

In @fig-trace the largest value, 1.8, arrives at position 6. The second-largest, 1.4, arrived at position 1 and was the maximum until position 6. The network answers 1, which is correct.

@fig-vote splits the final logits into what the comparators contribute and what the other 12 units contribute. The other units give the largest input's position +47, and the comparators give it −35. The runner-up's position wins because of the comparators (+39); the other units give it −2. Neither group picks the answer alone. The comparators' largest vote is at position 2 (+57), which the other units cancel (−46). Position 9, the newest input, gets the largest votes of both groups (+51 and −82). The network's own rewrite reads position 9 through a separate route: the current input, the runner-up level and the record pulse (`code/exact_rewrite.py`, docstring).

#figure(image("figures/example_vote.png", width: 82%), caption: [*The final vote in the example.* Contribution of the comparators and of the other 12 units in $h_9$ to each output logit, through the readout $U$.]) <fig-vote>

The same pattern holds across inputs. On 16,076 random inputs whose maximum and runner-up both arrive before position 9, among positions 0–8:
- the other units' largest vote is at the maximum's position on 87% of inputs, with median +44 there;
- the comparators' most negative vote is at the maximum's position on 76% of inputs, with median −28 there;
- at the runner-up's position the medians are +15 from the other units and +12 from the comparators;
- the comparators favour the runner-up over the maximum on 97% of inputs, and the other units on 8%.

So the other units vote mainly for the maximum, the comparators move votes from the maximum to the runner-up, and the answer comes from the sum.

= Evidence

== The clock tracks the age of the maximum

#figure(image("figures/clock_phase.png", width: 58%), caption: [*The clock's angle tracks the age of the maximum.* Median angle (dots) and middle 50% of steps (band), by steps since the current maximum arrived; 4,000 inputs, steps with a positive maximum. The clock is read from the exact rewrite (`code/exact_rewrite.py`). The dashed line is a straight-line fit.]) <fig-clock>

The median angle rises by about 42° per step from age 1 to age 9 (@fig-clock), and at every age the middle half of steps lies within about 12° of the median. Without inputs the clock would turn 42.75° and shrink by about 5% per step. With inputs, its median radius stays between 0.55 and 0.63 from age 1 to age 9, so later inputs keep pushing it back out.

== Both accounts, tested against the network

#figure(
  text(size: 9pt, table(
    columns: (2.3fr, 1fr, 1fr, 1fr, 1fr, 1.4fr),
    align: (left, right, right, right, right, right),
    stroke: (x, y) => if y == 0 { (bottom: 0.6pt) } else { none },
    inset: (x: 5pt, y: 3.5pt),
    [*program*], [*task accuracy*], [*agreement*], [*mistakes reproduced*], [*KL (nats)*], [*worst single unit*],
    [network], [95.29%], [—], [—], [—], [—],
    [rule "answer the true 2nd-largest"], [100%], [95.29%], [0%], [—], [—],
    [idealised account], [96.46%], [93.18%], [5.2%], [1.80], [33.5% (unit 7)],
    [full account], [95.25%], [99.64%], [96.1%], [$6.7 times 10^(-4)$], [99.55% (unit 7)],
    [exact rewrite], [95.29%], [99.98%], [99.7%], [$1.8 times 10^(-6)$], [99.94% (unit 7)],
  )),
  caption: [*The idealised account solves the task; the full account reproduces the network.* 131,072 fresh standard-normal inputs, measured with `code/verify.py`. The exact rewrite is the network's recurrence in named coordinates at five significant digits. It checks that the re-coding is correct and simplifies nothing.],
) <tab-main>

The idealised account solves the task more often than the network, but it agrees with the network less often than the plain rule. When any one of its comparators replaces the real one, the network keeps only 33–49% of its answers. So it states the algorithm without describing the units faithfully. The full account passes every test: each unit replaced alone keeps at least 99.5% of the network's answers.

*Size.* Counting every numeric literal in the code, with the readout excluded, the idealised account has 171 and the full account 334 (285 of them coefficients). The network has 272 weights before the readout. So the full account is not a compression of the network. It changes coordinates so that the parts can be named and removed one at a time.

== How the full account relates to the idealised one

The full account (`code/full_account.py`) writes the recurrence of the 13 non-pulse units in the eigenbasis of their recurrent matrix. Each eigenvalue is a *mode*: a real number is a component that decays or grows by that factor each step, and a complex pair is a component that rotates by an angle and shrinks to a radius each step. The three pulses are kept separate, and every constant is rounded to a fixed grid. @tab-map compares the modes and gains with the idealised account.

#figure(
  text(size: 9pt, table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    stroke: (x, y) => if y == 0 { (bottom: 0.6pt) } else { none },
    inset: (x: 6pt, y: 3pt),
    [*quantity*], [*full account*], [*idealised account*],
    [maximum level], [mode with factor 1.0098], [held exactly (factor 1)],
    [runner-up level], [mode with factor 0.9736; in level coordinates about $0.974 S - 0.016 M$], [held exactly],
    [main clock], [rotation 42.75°, radius 0.951], [42.40°, radius 1],
    [runner-up's age], [no separate clock: two further rotations at 83.27° and 124.78° (about 2× and 3× of 41.6°) and a short-lived clock (42.98°, radius 0.735) carry it, each as an independent component], [a second clock; it and its 2nd and 3rd powers (84.8°, 127.2°) feed the wave units],
    [comparator gains (1, 6, 7, 8)], [10.56, 11, 13.17, 12.31], [10.5, 11, 13, 12],
    [record unit input weight], [10.16], [10],
    [no counterpart], [alternating mode (factor −0.928); short transient (0.231); echo of the current input (0.001)], [—],
  )),
  caption: [*Modes and gains of the two accounts.* The runner-up's level-coordinate form is a regression from the original notes, not a coefficient of `full_account.py`.],
) <tab-map>

The levels, the main clock and the gains have close idealised counterparts. Most of the full account is something else: 256 of its 285 coefficients are dense couplings: how the units and pulses read the modes, and how the pulses write into them. We have no interpretation for most of them.

= What the idealised algorithm leaves out

The original authors removed or idealised one part of the full account at a time, without refitting, and measured agreement and mistakes reproduced (@fig-abl).

#figure(image("figures/ablations.png", width: 100%), caption: [*One change at a time.* Agreement and mistakes reproduced after one change to the full account, with no refitting. 19,776 inputs, the full account's development set; numbers from `code/accounts_working_notes.md`. Dotted lines mark 98% and 80%. Definitions: the *echo* is a mode that holds the current input for one step. *Clipping write-back* means that memory is updated from the wave units' values after the ReLU; the ablation writes back their values before it. *Wave-unit input weights* are the small direct weights from $x$ to the wave units.]) <fig-abl>

We read four things from @fig-abl.
- *Exact angles matter little, but the shared frequency does.* The two extra rotations have their own base frequency, about 41.6°. Tying them to exact multiples of 41.6° costs nothing. Putting all angles on the idealised account's single 42.40° frequency keeps 98.3% of answers but only 84% of mistakes, and tying the two rotations to multiples of the main clock's 42.75° keeps 81% of mistakes.
- *Idealising the levels or the clock's decay is costly without refitting.* Each keeps 89–92% of answers and 59–64% of mistakes. The original authors also refitted all other constants with the levels held exactly. This recovered 98.5% agreement and 86% of mistakes, so the level detuning can be compensated elsewhere.
- *The extra modes matter.* Removing the short-lived clock keeps only 51% of answers. Removing the alternating mode or the transient keeps 89–91%.
- *The calibration matters.* If the comparators read only the levels and the main clock, 36% of answers survive. Without clipping write-back, 60% survive.

Removing unit 5 or unit 9 is a control: the idealised account has both pulses too. Their large cost shows that these pulses are essential to both accounts, not that the idealised account lacks them.

These experiments show what the full account needs in order to reproduce the network. They do not show which part causes which mistake. We have not tested that, for example by adding parts to the idealised account and checking which mistakes appear.

== The network's mistakes

#figure(image("figures/accuracy_by_gap.png", width: 62%), caption: [*The network's mistakes are near-ties.* Network accuracy by the smaller of the two gaps (largest minus second, second minus third); 20,000 inputs.]) <fig-gap>

The network's mistakes are mostly near-ties (@fig-gap). On 20,000 inputs, 77% of its mistakes fall on the 19% of inputs where two of the three largest values are within 0.05 of each other, and 63% of mistakes name the third-largest. The idealised account behaves differently, according to the original notes on 19,776 inputs. It is right more often than the network when the second and third values are within 0.05 (90.8% against 77.8%). It is less often right when both gaps are at least 0.05 (97.3% against 98.4%). And 383 of its 690 mistakes pick a value ranked 4th to 10th, against 161 of the network's 931.

The network also remembers more than the top three. The inputs $(1.2, 0.1, 0, 1.23, 1.26, 0, 0, 0, 0, 0)$ and the same with $x_1 = 1.1$ have the same three largest values at the same positions. The network answers 3 for the first (correct) and 0 for the second. The changed input was briefly the runner-up, then dropped to fourth, and it still changes the answer. The idealised account answers 3 for both. In the full account, most of the difference comes from the short-lived clock (original notes).

= Two efforts with overlapping results

Two efforts in our team, both carried out by AI agents, reached overlapping descriptions.
- *Weight decomposition.* The first split the network's weight matrices into a sum of simple rank-one pieces, each needed on few inputs, and interpreted the pieces.
- *Program writing.* The second gave agents the network, without the decomposition, and asked for a program that reproduces its hidden state. Their first program was the exact rewrite. The idealised and full accounts came in later sessions, and their write-up adopted the decomposition paper's names for the unit roles.

The two efforts agree on what each unit does but group and name some units differently:

#figure(
  text(size: 9pt, table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    stroke: (x, y) => if y == 0 { (bottom: 0.6pt) } else { none },
    inset: (x: 6pt, y: 3pt),
    [*units*], [*decomposition paper*], [*this paper*],
    [1, 6, 7, 8], [comparators], [comparators],
    [4], [comparator ("max detector": fires on a new maximum)], [record pulse],
    [5], [recorder: writes a record of an input one step after it arrives], [pulse after a record or new runner-up],
    [9], [timer: fires one step after a new maximum], [pulse after a record],
    [0], [timer, and part of the memory], [wave unit],
    [2, 3, 10–15], [memory], [wave units],
  )),
  caption: [*Unit roles in the two efforts.* Both working papers are in `original-papers/`.],
) <tab-roles>

= Limitations and open questions

- *No short faithful account.* The idealised account is short but not faithful; the full account is faithful but not short. Searches by the original authors for a short recursion that passes the single-unit and mistakes tests failed. A failed search is not a proof that none exists.
- *No mechanistic accuracy estimate.* ARC's standard for an explanation is a deductive estimate of the network's accuracy whose error matches random sampling at the same compute. Neither account gives one.
- *Fitted thresholds do not replace units.* Threshold formulas in the idealised form, fitted directly to the real comparators, reach $R^2 > 0.99$. Yet replacing any one of units 1, 6, 7 or 8 by its fit keeps only 78–88% of the network's answers (original notes).
- *Error causation is untested.* We know where the mistakes are and which parts the full account needs, but not which parts produce which mistakes.

#v(0.6em)
#text(size: 9pt)[*Reproducing.* `code/verify.py` re-runs Table 1 with NumPy alone. `code/analysis.py` and `code/readout_stats.py` produce the example, clock, near-tie and readout statistics. `code/README.md` lists every file. The two working papers report slightly different numbers (for example 170 constants, 96.6% accuracy and 6% of mistakes for the idealised account). They used other samples and counting conventions; see `original-papers/README.md`. The work was carried out by AI research agents in Goodfire's parameter-decomposition team, at the request of the team's researchers.]
