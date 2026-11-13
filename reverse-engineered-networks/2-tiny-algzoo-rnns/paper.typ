#set page(paper: "us-letter", margin: (x: 1in, y: 0.9in), numbering: "1")
#set text(font: ("Libertinus Serif", "New Computer Modern"), size: 10.5pt)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.")
#show heading: set block(above: 1.2em, below: 0.6em)
#show figure.caption: set text(size: 9pt)
#show raw.where(block: true): set text(size: 8.5pt)
#show raw.where(block: true): block.with(fill: luma(246), inset: 7pt, radius: 3pt, width: 100%)
#set table(stroke: (x, y) => if y == 0 { (bottom: 0.6pt) } else { (top: 0.3pt + luma(200)) }, inset: 4.5pt)

#align(center)[
  #text(size: 15pt, weight: "bold")[How two tiny recurrent networks do their tasks, \ and one that resisted explanation]
  #v(0.3em)
  #text(size: 9.5pt)[Worked examples from ARC's AlgZoo. Analyses by AI research agents; every number re-measured for this note. November 2026.]
]
#v(0.6em)

#block(inset: (x: 1.5em))[
  #text(weight: "bold")[Abstract.]
  Neural networks learn algorithms that nobody wrote down, and interpretability research tries to read those algorithms back out of the weights.
  The smallest trained networks are the natural place to test whether this can be done completely, because every weight can be inspected.
  However, even networks with a few dozen parameters are rarely explained down to the cause of each error.
  Here we show complete, checkable accounts of two networks from ARC's AlgZoo collection, and an honest failure on a third.
  A 2-neuron recurrent network trained to output the median of 10 numbers instead stores one signed running number. Each step updates it as $s <- "median"(x + 0.98 s, 0, x + 3.5 s)$, a leaky sum that resists crossing zero. This one-line rule matches the network's output to RMSE 0.0045 and also predicts the effect of swapping hidden states.
  A 3-neuron network trained to locate the second-largest of 3 numbers runs two linear comparators on 74% of inputs. On 20% of inputs every neuron is silent and every logit is exactly 0, so the answer comes from `argmax` tie-breaking; these inputs cause 68% of its errors. A 12-line rule agrees with the network on 99.8% of inputs.
  A 352-weight network for the median position of 5 numbers resisted 35 agent sessions. They found correct local facts but no account that predicts the network when its hidden state is edited.
  The examples show what a complete explanation of a small network can look like, and that accuracy alone can hide fine-tuned cancellations and tie-breaking artefacts.
]

= Setting

*Networks.* AlgZoo is a public set of small networks released by the Alignment Research Center (ARC) as interpretability challenges @algzoo. Each one we study is a one-layer ReLU RNN without biases. It reads one number per step,
$ h_t = "relu"(W_"ih" x_t + W_"hh" h_(t-1)), quad h_0 = 0, $
and outputs $W_"out" h_T$ after the last step. Checkpoint names give the task, the hidden size and the sequence length: `median_2_10` has 2 hidden neurons and reads 10 numbers. We use `median_2_10` seed 0, `2nd_argmax_3_3` seed 1 and `argmedian_16_5` seed 0. Inputs are independent standard normal numbers. Because there are no biases, every network is _positively homogeneous_: scaling all inputs by $lambda > 0$ scales every activation and output by $lambda$.

*How accounts were produced and checked.* AI research agents (Claude models) had the weights and a Python sandbox, and one agent per network wrote the account. For this note we turned each account into a short replacement rule, and compared it with the network on $10^6$ to $2 dot 10^6$ fresh inputs. The scripts in `code/` print every number in Sections 2 and 3, except one retraining result marked in the text; each runs in seconds with numpy.

= Median of 10 numbers: one signed number in two neurons

`median_2_10` has 2 neurons and 8 weights and was trained with squared error to output the median of its 10 inputs. We call the neurons $n$ and $p$. In the order $(n, p)$, its weights are
$ W_"ih" = mat(-0.333; 0.335), quad W_"hh" = mat(0.979, -3.490; -3.576, 0.977), quad W_"out" = mat(-0.342, 0.341). $

*The two neurons store one signed number.* Neuron $p$ is driven by $+x$ and neuron $n$ by $-x$. Each excites itself and strongly inhibits the other. They are never active together: with $n = 0$ and $p = u$, both pre-activations positive would need $x > -2.9 u$ and $x < -10.5 u$ at once, and the mirror case is similar. On all $10^7$ (sequence, step) pairs we checked, at most one neuron was active. So the state is one signed number $s = p / 0.335 - n / 0.333$, measured in input units, and the network is exactly a one-dimensional recurrence.

*The update is a median.* Take $s > 0$, so only $p$ is on. The next pre-activations are $0.335 (x + 0.977 s)$ for $p$ and $-0.333 (x + 3.51 s)$ for $n$. ReLU keeps whichever is positive, or neither. That is the middle value of three numbers, and the case $s < 0$ is its mirror. Averaging the two neurons' nearly equal constants gives the rule
$ s_t = "median"(x_t + 0.98 s_(t-1), quad 0, quad x_t + 3.5 s_(t-1)), quad "output" = 0.114 s_10 . $
@fig-median(a) draws this map, which has four cases. If $x + 0.98 s$ has the same sign as $s$, the state becomes $x + 0.98 s$, a slightly leaky running sum; the input may oppose $s$, just not enough to cross zero (69.7% of steps). If $x + 0.98 s$ crosses zero but $x + 3.5 s$ does not, the state stops at exactly 0 (8.7%). If even $x + 3.5 s$ crosses zero, the sign flips and the state becomes $x + 3.5 s$, smaller in size than $x$ (3.7%). From $s = 0$ the state restarts at $x$ (17.9%, including the first step).

#figure(image("figures/median_mechanism.png", width: 100%),
  caption: [The median network keeps one signed number and updates it with a three-way median. (a) The update for old state $s = +1$ and every new input $x$; a plain leaky sum would follow the dashed line. (b) On one random sequence, the number $s$ read from the two neurons (blue) equals the rule's (orange). (c) Network output against the true median, 4,000 sequences; the row at exactly 0 comes from sequences whose last step hit the stop at zero.]) <fig-median>

*The rule is the whole mechanism.* On $10^6$ fresh sequences the rule's output differs from the network's by RMSE 0.0045, against a standard deviation of 0.32 in the network's output. The rule also predicts the network under intervention. At a random step we overwrote the hidden state with the state from a different sequence, and the rule carried over only the number $s$ read from it. Rule and network then differed by RMSE 0.0027, while the swap moved the output by RMSE 0.335.

*The network computes a modified mean, not a median.* The output correlates 0.983 with the mean of the inputs. For 10 normal samples the mean is already a decent median estimate, and the reluctance to cross zero helps slightly (@tab-median).

#figure(table(columns: 2, align: (left, right),
  [Estimator], [RMSE to true median],
  [Always 0], [0.372],
  [Mean of the inputs], [0.196],
  [Rule without the stop at zero ($0.98$ in both slots)], [0.197],
  [Rule with a hard stop at zero (never flip)], [0.192],
  [*Network*], [*0.189*],
  [*Rule*], [*0.189*]),
  caption: [The network and its rule are slightly better median estimators than the plain mean; $10^6$ sequences.]) <tab-median>

= Second-largest of 3 numbers: comparators and a dead zone

`2nd_argmax_3_3` has 3 neurons, which we call $a, b, c$, and 21 weights. It reads $x_1, x_2, x_3$ and outputs logits for the position (1, 2 or 3) of the second-largest, which for three numbers is the median. With `argmax` returning the first maximal position, as numpy and PyTorch do, it is 90.8% accurate. Its weights are
$ W_"ih" = mat(3.842; 0.000; -3.825), quad
  W_"hh" = mat(-0.938, -6.128, 6.174; 5.390, -5.207, 5.415; 6.116, -6.097, -0.940), quad
  W_"out" = mat(2.157, -2.490, 2.153; -1.993, 1.856, -2.016; -0.120, 0.099, -0.135). $

*Two symmetries reduce the problem to a plane.* By homogeneity, the prediction depends only on the direction of $x$. The task is exactly unchanged by negating all inputs, and the network nearly is: its weights are almost invariant under negating the input while swapping $a$ and $c$, and negation changes 0.17% of its predictions. We therefore assume $x_1 > 0$, and @fig-argmax shows every such input by fixing $x_1 = 1$.

#figure(image("figures/second_argmax_map.png", width: 100%),
  caption: [The second-largest network answers correctly in region M, falls silent in B, and answers for a different input in F. Each panel fixes $x_1 = 1$, which by homogeneity covers all inputs with $x_1 > 0$. (a) Correct answer. (b) Network answer; hatching marks inputs where all three logits are exactly 0. (c) Errors. Dashed lines: $x_2 = 0.938$ and $x_2 = 6.14$. The view is not probability-weighted: $x_2 / x_1$ is Cauchy-distributed, so half of all inputs have $|x_2 / x_1| < 1$.]) <fig-argmax>

*The step-2 state splits the inputs into three regions.* At step 1, $a = s dot "relu"(x_1)$ and $c = s dot "relu"(-x_1)$ with $s = 3.84$; neuron $b$ gets almost no input. At step 2, $b = 5.39 s x_1$ stores the size of $x_1$. Neuron $a = s dot "relu"(x_2 - 0.938 x_1)$ fires when $x_2$ is above about $x_1$. Neuron $c = s dot "relu"(6.14 x_1 - x_2)$ fires when $x_2$ is below $6.14 x_1$: a large weight from $a$ to $c$ moves its threshold out from $x_1$. The ratio $r = x_2 / x_1$ therefore picks a region: only $c$ is on in M ($r < 0.938$, roughly where $x_1$ is the larger of the first two), both are on in B ($0.938 <= r < 6.14$, roughly $x_2 > x_1 > 0$), and only $a$ is on in F ($r >= 6.14$).

*The readout is a ratio test.* Neurons $a$ and $c$ have nearly equal output weights, so the logits see only $q = a + c$ and $b$. Relative to logit 3, logit 1 wins when $q / b > 1.137$ and logit 2 wins when $q / b < 0.938$; otherwise position 3 wins. Replacing $W_"out"$ by this bare ratio test keeps 99.4% of predictions.

*Region M (73.9% of inputs) runs two comparators.* At step 3 the active neuron switches from $c$ to $a$; $c$ is off except in the rare case $x_3 < -38 x_1$. Neuron $b = s (5.05 x_1 - 5.39 x_2)$ tracks the gap between $x_1$ and $x_2$, and $a = s (x_3 + 4.73 x_1 - 6.15 x_2)$ holds $x_3$ shifted by a fixed mix of the other two. Each readout threshold nearly cancels one input: $a - 1.137 b = s (x_3 - 1.01 x_1 - 0.02 x_2)$, since $1.137 times 5.39 = 6.13 approx 6.15$, and $a - 0.938 b = s (x_3 - 1.09 x_2 - 0.01 x_1)$, since $0.938 times 5.05 = 4.74 approx 4.73$. So the logit differences are comparators: position 1 if $x_3 > 1.011 x_1 + 0.017 x_2$, position 2 if $x_3 < 1.085 x_2 + 0.011 x_1$, and position 3 otherwise. That is the median rule, with the $x_2$-side boundary 8.5% too far out. The network is 98.4% accurate here.

*Region B (20.9%) is a dead zone.* Neuron $b$'s step-3 input is $5.39 a - 5.21 b + 5.41 c$. Since $a + c$ stays near $5.2 s x_1$, these terms cancel to about $-0.011 s x_1$, so $b$ is off. Neurons $a$ and $c$ receive strong inhibition from $b$ and strong excitation from each other; the inhibition wins by a wide margin (median input $-19 s x_1$), and they switch on only when $x_3$ lies far beyond $x_2$ (5% of B). On the other 95% of B every neuron is off and all three logits are exactly 0. The answer is then position 1, by `argmax` tie-breaking, which is right 67.6% of the time here. Region B causes 73.8% of all errors, and the all-zero inputs (19.9% of all inputs) cause 67.5%. With random tie-breaking the network would score 83.8%.

*The dead zone sits on a knife edge.* Neuron $b$'s step-3 input in B is only $-0.00038$ times its self-inhibition term $W_"hh"[b, b] dot b$ (median), a margin of 0.04%. Shrinking the size of that one weight by 0.05% lets $b$ switch on in B while $q$ stays 0, so the network answers position 2 there and accuracy drops from 90.8% to 80.6%. The effect is one-sided: growing the weight keeps B silent and only moves the comparators, costing 0.1 points at 0.05% and 2 points at 0.3%.

*Region F (5.2%) is aliased.* When $x_1$ is tiny next to $x_2$, the step-2 state equals that of the region-M input $(-x_1, x_2 - 7.08 x_1, x_3)$, and the network answers for that input. Running the network on the alias reproduces its prediction on 99.94% of F inputs. The network is 76.2% accurate here.

*The account as a rule.* Mirror so that $x_1 > 0$; in M apply the two comparators above; in B answer 1; in F apply comparators for the alias ($x_3 < -1.124 x_1 + 0.017 x_2$ gives 1, $x_3 > 1.082 x_2 - 7.67 x_1$ gives 2, otherwise 3). This rule agrees with the network on 99.81% of $2 dot 10^6$ fresh inputs: 99.82% in M, 100% in B and 98.9% in F.

*What remains open.* We do not know why training produced the dead zone. One hypothesis is that the large $a$–$c$ cross-weights, which step 3 needs for the comparators, create the band when reused at step 2. The agent's from-scratch retraining with its own recipe (Adam, 30,000 steps, 10 seeds), which differs from ARC's, reached only 55–64% accuracy, so it does not test this. Our scripts do not rerun it.

= Median position of 5 numbers: a network that resisted

`argmedian_16_5` has 16 neurons and 352 weights, reads 5 numbers, and outputs the position of their median with 98.3% accuracy (chance is 20%). ARC has not published its mechanism. AI agents worked on it in 7 rounds of 5 sessions: in each round four agents worked in parallel from the best account so far, and a fifth merged their results. The goal was an account that predicts the network's answer when part of its hidden state is replaced by the value it would have on a different input. The baseline account is the statement "the network outputs the true median position".

*The agents found correct local facts.* We confirmed two from the weights (`code/argmedian_16_5_checks.py`). After step 4, five rows of $W_"hh"$ each read one earlier input from the hidden state almost linearly: $x_1$, $x_2$, $x_3$, and $x_4$ twice with opposite signs ($R^2 >= 0.997$). And a few neurons compare each new input with earlier ones. After step 4, four neurons fit $"relu"(x_i - x_4)$ for $i = 1, 2, 3$ and $"relu"(x_4 - x_3)$, scaled by about 11. After step 5, five neurons fit $"relu"(x_i - x_5)$ for $i = 1, ..., 4$ and $"relu"(x_5 - x_4)$. Residuals are 1–5% of each neuron's standard deviation.

*These facts did not assemble into a predictive account.* The following numbers come from the agents' runs, not our scripts. When the whole hidden state after one step is swapped in from another input, the baseline predicts the network's answer 29% of the time, averaged over the five steps. The only gain used the most trivial fact, that the state after step 1 depends only on $x_1$, which makes step-1 swaps fully predictable; it lifted the average to 40%. Accounts built on the stored inputs predicted the answer after an edit of one stored input about half the time (48%, or 52% after a search for better directions). Editing one stored input also changes other neurons' next-step inputs, so the edited state is unlike any the network visits. In one round an agent made its account score better by exploiting a loophole in how claims were checked, not by explaining more, and said so in its own notes. The loophole was closed; the honest result is the gain from 29% to 40%.

= Discussion

The two solved networks share a property the third lacks: within each of a few linear pieces, the state holds one to three quantities that can be named. The median network is a one-dimensional recurrence, and the second-largest network is three linear regions. In both cases the account is short, follows from the weights by algebra, and is checked by a replacement rule over millions of inputs.

The examples also show what accuracy hides. The second-largest network's 90.8% depends on how `argmax` breaks ties, and its largest failure region rests on a cancellation that shrinking one weight by 0.05% destroys. The median network was trained to output a median but computes a leaky mean that resists crossing zero. None of this is visible from accuracy alone.

The failure is equally instructive. 352 weights is only about 17 times larger than 21, but the 5-input network spreads its state across neurons. Agents found true facts about it, yet the facts did not predict behaviour under edits. Finding correct local structure is not the same as explaining a network.

*Limitations.* The accounts were written by AI agents; we re-derived the key steps from the weights and re-measured the numbers in Sections 2 and 3. The success bars differ. We tested the median account under intervention, but the second-largest account only on unedited inputs, plus the alias substitution and weight perturbations. The 5-input network was held to the stricter intervention standard. Section 4's agent-run numbers come from runs whose code is not included here.

#bibliography("refs.yml", style: "ieee")
