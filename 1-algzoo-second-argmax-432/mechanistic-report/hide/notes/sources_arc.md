# ARC primary sources on the AlgZoo 2nd-argmax RNNs ($M_{16,10}$ and family)

Fetched 2026-11-19. Everything in a blockquote is verbatim from the URL given with it (ARC's LaTeX transcribed into `$...$`; typos are ARC's; `[…]` marks an omission; some lists are run together on one line). Everything marked **[mine]** is my own inference or my own computation, not an ARC statement. Raw copies of sources S1–S10 are in `arc_raw/` beside this file; my checks are `verify_arc_claims.py` and `verify_arc_claims_output.txt` (about 30 s on CPU). This folder is `c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/notes/`; file paths below are relative to it.

## 0. Summary

1. **Provenance.** The blog's $M_{16,10}$ is the file `gs://arc-ml-public/alg/one_layer_16_hidden_94_acc_2nd_argmax.pth` (uploaded 2024-11-28). It is byte-identical (SHA-256 `498bd9a8…18bb`) to the model linked from Zihao Chen's April 2024 notebook and to our `reverse-engineered-networks/1-algzoo-second-argmax-432/code/m16_10.pth` **[mine]**.
2. **What ARC claims to understand** is five units. In rescaled units (ARC's "standardisation"), with $M_k := \max(0, x_0, \dots, x_k)$ and our indexing ($h_s$ = state after $x_s$): unit 2 $\approx M_{s-1}$, unit 4 $\approx M_s - M_{s-1}$, unit 7 $\approx \mathrm{ReLU}(M_{s-1} - x_s)$, unit 6 $\approx \mathrm{ReLU}(M_{s-2} - x_s)$, unit 1 $\approx \mathrm{ReLU}(\max(M_{s-3}, x_{s-1}) - x_s)$. The last three are "leave-one-out maximum minus newest input" features. Chen's 2024 note adds: unit 8 $\approx \mathrm{ReLU}(\text{2nd max} - x)$ at early steps, a "max sub-circuit" (the readout without units 1, 6, 7, 8 finds the argmax), and logits that shift one position per step.
3. **What ARC says it does not understand:** the other 11 units ("more densely connected"), any correspondence between them and a handcrafted solution, the approximation errors, and, in Chen's words, how the four negative-input units, "which seem very entangled", represent the per-position scores. There is no ARC statement about units 0, 3, 5, 10–13, 15, and nothing about rotations or circles beyond one code comment and a remark about linear argmax RNNs (section 7).
4. **Reference algorithm.** ARC's handcrafted 22-unit RNN computes $\text{logit}_i = \max(0, x_i) + 2 \max(0, x_j : j \ne i)$ exactly **[mine: verified to $10^{-14}$; the code comment has the wrong sign]** and is exactly right whenever at least two inputs are positive (accuracy 99.0%).
5. **Training recipe of $M_{16,10}$** (Chen's notebook, training call commented out): fresh iid $\mathrm{Unif}(-1,1)$ inputs, cross-entropy on the final logits, Adam, lr 0.005, weight decay $10^{-14}$, ExponentialLR $\gamma = 0.995$ per epoch, 1000 epochs of 100,000 sequences, PyTorch default init, no seed. **The batch size is not given anywhere.** The zoo models use a different recipe (Gaussian inputs, AdamW, $2^{30}$ sequences).
6. **Accuracy.** ARC: 95.3% on $\mathcal N(0,1)$ (blog), 94% on $\mathrm{Unif}(-1,1)$ (Chen; 0.93792 on $10^5$ samples). **[mine]** 0.9531 and 0.9386 on $2^{20}$ samples.
7. **ARC's standard.** An explanation is a mechanistic (deductive, sampling-free) estimate of the accuracy, judged by squared error against compute (must match sampling) or by surprise accounting. The challenge is exactly that for $M_{16,10}$. In August 2026 Jacob Hilton called outside progress "fairly marginal".
8. **My checks [mine]:** ARC's five formulas hold with $R^2$ 0.94–0.995 on the activations, but the "$\approx \pm 1$" weights are really 0.84–1.31 in magnitude, units 6 and 1 have self-connections of 0.36 and 0.41 that ARC does not mention, and the comparison thresholds sag below the true maxima by about 0.1 input units at late steps.

## 1. Sources

| # | Source | URL | Status |
|---|---|---|---|
| S1 | Blog post, Jacob Hilton, 2026-01-26 | https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/ | read in full |
| S2 | LessWrong / Alignment Forum cross-post, 10 comments (3 by Hilton) | https://www.lesswrong.com/posts/x8BbjZqooS4LFXS8Z/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters | post and all comments read (GraphQL); body is the same as S1; no AF-only comments |
| S3 | GitHub repo `alignment-research-center/alg-zoo` (branches `main`, `mlp`; 4 commits, all by Hilton) | https://github.com/alignment-research-center/alg-zoo | every file read. No notebooks, no PRs, discussions disabled, one issue |
| S4 | Zihao Chen, "RNNs for the 2nd argmax", April 2024, 8 pp. | https://drive.google.com/file/d/1USCD1LPa6hXMTj7PAO6Jmc4dVdxemZY3/view | read in full |
| S5 | Chen's Colab notebook (49 cells) | https://colab.research.google.com/drive/1yRv18_Gp0twoZ-kptMygVhAq-azj2bBq (the copy linked from the blog; identical cells to the original `1KgkLyR-sKB8PZxYtjH9YIdgJPus0I8tW`) | all cells, text outputs and figures read |
| S6 | Jacob Hilton, "Heuristic explanations for 2nd argmax models", 2024-12-21, 6 pp. ($M_{4,3}$) | https://drive.google.com/file/d/1t2rXhDQqWrOXRwjqstbrBPWiCmYlMts_/view | read in full |
| S7 | George Robinson, "Symmetric RNNs", December 2024, 7 pp. | https://drive.google.com/file/d/1qH61_WshTrgYh0YLj5vkONzk8x00zImN/view | read in full |
| S8 | Public bucket `gs://arc-ml-public/alg/` (models, training and validation logs) | https://storage.googleapis.com/storage/v1/b/arc-ml-public/o?prefix=alg/ | listed; four checkpoints and 16 logs downloaded |
| S9 | "Competing with sampling", 2025-11-18 (matching sampling principle) | https://www.alignment.org/blog/competing-with-sampling/ | main sections read |
| S10 | "Formal verification, heuristic explanations and surprise accounting", 2024-06-25 | https://www.alignment.org/blog/formal-verification-heuristic-explanations-and-surprise-accounting/ | read in full |
| S11 | Other ARC posts 2024–2026 (14 downloaded and searched for AlgZoo, argmax, RNN) | https://www.alignment.org/blog/ | only "Announcing the ARC White-Box Estimation Challenge" mentions AlgZoo, in one sentence |
| S12 | David Matolcsi (former ARC), "Obstacles in ARC's agenda: Finding explanations" | https://www.lesswrong.com/s/uYMw689vDFmgPEHrS/p/xtcpEceyEjGqBCHyK | searched; context only, no mechanism |

Nothing I tried to fetch failed. The blog's two figures (the architecture diagram and the $M_{2,2}$ region diagram) were viewed as images. I found no ARC paper, talk or post other than the above that discusses these models; I did not search video or social media.

## 2. Indexing conventions

| | recurrence | "$h_t$" means | unit labels |
|---|---|---|---|
| ARC blog (S1), Hilton (S6) | $h_0 = 0$, $h_{t+1} = \mathrm{ReLU}(W^{hh} h_t + W^{hi} x_t)$, logits $= W^{oh} h_n$ | state after $t$ inputs $x_0..x_{t-1}$ | 0-based |
| Chen (S4) | $h_0 = 0$, $h_{t+1} = \mathrm{ReLU}(W_{hh} h_t + x_t w_{ih})$, $y = W_{oh} h_{d+1}$, inputs $x_1..x_d$ | loosely used | N0–N15, 0-based **[mine: confirmed by reproducing his ablation table]** |
| ours (project CLAUDE.md) | $h_s = \mathrm{ReLU}(W_{ih} x_s + W_{hh} h_{s-1})$, $h_{-1} = 0$, $\ell = W_{out} h_9$ | state after $x_0..x_s$ | 0-based |

So ARC's $h_t$ is our $h_{t-1}$. With $M_k := \max(0, x_0, \dots, x_k)$ and $M_{-1} := 0$, ARC's formulas (section 3.1) translate as follows **[mine]**:

| unit | ARC, in $h_t$ | ours, in $h_s$ ($s = t - 1$) |
|---|---|---|
| 2 | $\max(0, x_0..x_{t-2})$ | $M_{s-1}$ |
| 4 | $\max(0, x_0..x_{t-1}) - \max(0, x_0..x_{t-2})$ | $M_s - M_{s-1} = \mathrm{ReLU}(x_s - M_{s-1})$ |
| 7 | $\max(0, x_0..x_{t-1}) - x_{t-1}$ | $M_s - x_s = \mathrm{ReLU}(M_{s-1} - x_s)$ |
| 6 | $\max(0, x_0..x_{t-3}, x_{t-1}) - x_{t-1}$ | $\mathrm{ReLU}(M_{s-2} - x_s)$ |
| 1 | $\max(0, x_0..x_{t-4}, x_{t-2}, x_{t-1}) - x_{t-1}$ | $\mathrm{ReLU}(\max(M_{s-3}, x_{s-1}) - x_s)$ |

## 3. (a) Every ARC statement about $M_{16,10}$

### 3.1 Blog post (S1; identical text in S2)

Introduction:

> The models are RNNs and transformers trained to perform algorithmic tasks, and range in size from 8 to 1,408 parameters. The largest model that we believe we more-or-less fully understand has 32 parameters; the next largest model that we have put substantial effort into, but have failed to fully understand, has 432 parameters.

Section "Mechanistic estimates as explanations":

> For most of the larger models from AlgZoo (including the 432-parameter model $M_{16,10}$ discussed below), we would consider it a major research breakthrough if we were able to produce a mechanistic estimate that matched the performance of random sampling under the mean squared error versus compute metric.[3] It would be an even harder accomplishment to achieve a full understanding under the surprise accounting metric, but we are less focused on this.

Section "Case study: 2nd argmax RNNs":

> The model $M_{d,n}$ is a 1-layer ReLU RNN with $d$ hidden neurons that takes in a sequence of $n$ real numbers and produces a vector of logit probabilities of length $n$. It has three parameter matrices:
> - the input-to-hidden matrix $W^{hi}\in\mathbb R^{d\times 1}$
> - the hidden-to-hidden matrix $W^{hh}\in\mathbb R^{d\times d}$
> - the hidden-to-output matrix $W^{oh}\in\mathbb R^{n\times d}$
>
> The logits of $M_{d,n}$ on input sequence $x_0,\ldots,x_{n-1}\in\mathbb R$ are computed as follows:
> - $h_0=0\in\mathbb R^d$
> - $h_{t+1}=\mathrm{ReLU}\left(W^{hh}h_t+W^{hi}x_t\right)$ for $t=0,\ldots,n-1$
> - $\mathrm{logits}=W^{oh}h_n$

> Each model in this family is trained to make the largest logit be the one that corresponds to the position of second-largest input, using softmax cross-entropy loss. The models we'll discuss here are $M_{2,2}$, $M_{4,3}$ and $M_{16,10}$. For each of these models, we'd like to understand why the trained model has high accuracy on standard Gaussian input sequences.

Section "Hidden size 16, sequence length 10", complete:

> The model $M_{16,10}$ can be loaded in AlgZoo using `example_2nd_argmax()`.[6] It has 432 parameters and an accuracy of 95.3%.
>
> This model is deep enough that a brute force approach is no longer viable. Instead, we look for "features" in the activation space of the model's hidden state.
>
> After rescaling the neurons of the hidden state, we notice an approximately isolated subcircuit formed by neurons 2 and 4, with no strong connections to the outputs of any other neurons:
>
> $$W^{hi}_{(2,4)}\approx\begin{pmatrix}0\\+1\end{pmatrix},\qquad W^{hh}_{(2,4),(2,4)}\approx\begin{pmatrix}+1&+1\\-1&-1\end{pmatrix}\qquad\text{and}\qquad W^{hh}_{(2,4),(0,1,3,\ldots)}\approx\begin{pmatrix}0&\ldots\\0&\ldots\end{pmatrix}.$$
>
> It follows that after unrolling the RNN for $t$ steps:
>
> - Neuron 2 is approximately $\max\left(0,x_0,\ldots,x_{t-2}\right)$
> - Neuron 4 is approximately $\max\left(0,x_0,\ldots,x_{t-1}\right)-\max\left(0,x_0,\ldots,x_{t-2}\right)$
>
> This can be proved by induction using the identity $\mathrm{ReLU}\left(a-b\right)=\max\left(a,b\right)-b$ for neuron 4.
>
> Next, we notice that neurons 6 and 7 fit into a larger approximately isolated subcircuit together with neurons 2 and 4:
>
> $$W^{hi}_{(6,7)}\approx\begin{pmatrix}-1\\-1\end{pmatrix},\qquad W^{hh}_{(6,7),(2,4)}\approx\begin{pmatrix}+1&0\\+1&+1\end{pmatrix}\qquad\text{and}\qquad W^{hh}_{(6,7),(0,1,3,\ldots)}\approx\begin{pmatrix}0&\ldots\\0&\ldots\end{pmatrix}.$$
>
> Using the same identity, it follows that after unrolling the RNN for $t$ steps:
>
> - Neuron 6 is approximately $\max\left(0,x_0,\ldots,x_{t-3},x_{t-1}\right)-x_{t-1}$
> - Neuron 7 is approximately $\max\left(0,x_0,\ldots,x_{t-1}\right)-x_{t-1}$
>
> We can keep going, and add in neuron 1 to the subcircuit:
>
> $$W^{hi}_{(1)}\approx\begin{pmatrix}-1\end{pmatrix},\qquad W^{hh}_{(1),(2,4,6,7)}\approx\begin{pmatrix}+1&+1&+1&-1\end{pmatrix}\qquad\text{and}\qquad W^{hh}_{(1),(0,1,3,\ldots)}\approx\begin{pmatrix}0&\ldots\end{pmatrix}.$$
>
> Hence after unrolling the RNN for $t$ steps, neuron 1 is approximately
>
> $$\max\left(0,x_0,\ldots,x_{t-4},x_{t-2},x_{t-1}\right)-x_{t-1},$$
>
> forming another "leave-one-out-maximum" feature (minus the most recent input).
>
> In fact, by generalizing this idea, we can construct a model by hand that uses 22 hidden neurons to form all 10 leave-one-out-maximum features, and leverages these to achieve an accuracy of 99%.[7]
>
> Unfortunately, however, it is challenging to go much further than this:
>
> - We have exploited the approximate weight sparsity of 5 of the hidden neurons, but most of the remaining 11 hidden neurons are more densely connected.
> - We have produced a handcrafted model with high accuracy, but we have not produced a correspondence between most of hidden neurons of the trained model and the hidden neurons of the handcrafted model.
> - We have used approximations in our analysis, but have not dealt with the approximation error, which gets increasingly significant as we consider more complex neurons.
>
> Fundamentally, even though we have some understanding of the model, our explanation is incomplete because we not have not turned this understanding into an adequate mechanistic estimate of the model's accuracy.
>
> Ultimately, to produce a mechanistic estimate for the accuracy of this model that is competitive with sampling (or that constitutes a full understanding), we expect we would have to somehow combine this kind of feature analysis with elements of the "brute force after exploiting symmetries" approach used for the models $M_{2,2}$ and $M_{4,3}$, and to do so in a primarily algorithmic way. This is why we consider producing such a mechanistic estimate to be a formidable research challenge.
>
> Some notes with further discussion of this model can be found here: **RNNs for the 2nd argmax** and complementary notebook by Zihao Chen

Footnotes:

> [1] Technically, the model was trained to minimize cross-entropy loss (with a small amount of weight decay), not to maximize accuracy, but the two are closely related, so we will gloss over this distinction.

> [6] We treat this model as separate from the "official" model zoo because it was trained before we standardized our codebase. Credit to Zihao Chen for originally training and analyzing this model. The model from the zoo that can be loaded using `zoo_2nd_argmax(16, 10)` has the same architecture, and is probably fairly similar, but we have not analyzed it.

> [7] This handcrafted model can be loaded in AlgZoo using `handcrafted_2nd_argmax(10)`. Note that this handcrafted model has more hidden neurons than the trained model $M_{16,10}$.

> [8] The specific model we are referring to can be be loaded in AlgZoo using `example_2nd_argmax()`. Additional 2nd argmax models with the same architecture, which a good method should also work well on, can be loaded using `zoo_2nd_argmax(16, 10, seed=seed)` for `seed` equal to 0, 1, 2, 3 or 4.

The challenge statement and footnotes 3 and 9 are in section 6.

**[mine] The induction ARC alludes to.** With the idealised weights, in ARC's indexing and with $m_t := \max(0, x_0..x_{t-1})$, $m_0 = 0$:

$$n_2(t{+}1) = n_2(t) + n_4(t), \qquad n_4(t{+}1) = \mathrm{ReLU}(x_t - n_2(t) - n_4(t)),$$
$$n_7(t{+}1) = \mathrm{ReLU}(n_2(t) + n_4(t) - x_t), \qquad n_6(t{+}1) = \mathrm{ReLU}(n_2(t) - x_t),$$
$$n_1(t{+}1) = \mathrm{ReLU}(n_2(t) + n_4(t) + n_6(t) - n_7(t) - x_t).$$

If $n_2(t) = m_{t-1}$ and $n_2(t) + n_4(t) = m_t$, then $n_4(t{+}1) = \max(x_t, m_t) - m_t = m_{t+1} - m_t$ and $n_2(t{+}1) = m_t$, which closes the induction. Then $n_7(t{+}1) = \mathrm{ReLU}(m_t - x_t)$, $n_6(t{+}1) = \mathrm{ReLU}(m_{t-1} - x_t)$, and since $n_6(t) - n_7(t) = \max(m_{t-2}, x_{t-1}) - m_t$, $n_1(t{+}1) = \mathrm{ReLU}(\max(m_{t-2}, x_{t-1}) - x_t)$. So unit 7 compares the new input with the maximum of all earlier inputs, unit 6 with that maximum leaving out the previous input, and unit 1 with it leaving out the input two steps back. Every feature is floored at 0, so the circuit says nothing about negative maxima.

### 3.2 Repository (S3)

README:

> Check you can load the 432-parameter example 2nd argmax model: `from alg_zoo import example_2nd_argmax` / `model = example_2nd_argmax()`

> The handcrafted 2nd argmax models for sequence lengths 2, 3 and 10 discussed in the blog post can be loaded using: `handcrafted_2nd_argmax(seq_len)`
>
> The trained 2nd argmax model with sequence length 10 discussed in the blog post is not officially part of the zoo, but can be loaded using: `example_2nd_argmax()`

`alg_zoo/loading.py`:

```python
def example_2nd_argmax() -> DistRNN:
    # 2nd argmax model with hidden_size=16, seq_len=10 discussed in blog post
    return load_dist_rnn(
        "gs://arc-ml-public/alg/one_layer_16_hidden_94_acc_2nd_argmax.pth",
        bias=False,
    )
```

`alg_zoo/architectures.py`: `OneLayerRNN` is `th.nn.RNN(input_size=1, hidden_size, nonlinearity="relu", bias=False, batch_first=True)` followed by `th.nn.Linear(hidden_size, output_size, bias=False)` applied to the final state. The repository says nothing else about $M_{16,10}$. The `mlp` branch (August 2026) only adds MLP classes, their loaders, a training option and `handcrafted_kth_argmax_mlp`.

### 3.3 Chen's note (S4), section 5, "Nonlinear RNN for 2nd Argmax"

> For the 2nd argmax task, nonlinearity and more neurons are required. In this section, we focus on our analysis of a 16-neuron RNN that achieves 94% accuracy on the uniform distribution for sequence length d = 10. We sketch some findings of the trained network and a hand-coded network inspired by these findings.

> 5.1 Mechanistic Findings. We summarize a few more definitive findings of the network, which we will use as hints to construct an almost-perfect RNN.
> 1. We have several major input neurons: N1, N4, N6, N7, N8. The weight for N4 is positive, while the others are negative.
> 2. N2 approximately implements the running max value $\max_t$, N4 approximately implements $\mathrm{ReLU}(x_t - \max_t)$, and N7 approximately implements $\mathrm{ReLU}(\max_t - x_t)$.
> 3. N8 approximately implements $\mathrm{ReLU}(\mathrm{2ndmax}_t - x_t)$ in earlier time steps.
> 4. When calculating the logits at each time t, projecting out the negative input neurons (N1, N6, N7, N8) gives an argmax/max sub-circuit that accurately finds the max index.
> 5. At a high level, new values enter through the last logit (m-th), and the previous values (inexactly) shift to earlier logits. For each iteration t, if $x_t > \max_t$ (running max value), the old max logit gets a boost. Note that this restates that the intermediate logits have good accuracies for the running 2nd argmax.

From 5.2, on the trained network:

> The choice of $s_9(x_t)$ is inspired by the trained RNN since the sum of N4 and N7 gives $|\max_t - x_t|$, which is reflected by the weights of the last logit.

"Discrepencies" between the trained network and Chen's hand-coded 21-unit network:

> Overall, the constructed net and the trained net share similar qualitative patterns. N2, N4 and N7 of the trained net have their exact counterparts in the constructed net. Both nets have a max/argmax sub-circuit. However, there are some major discrepancies:
> - The constructed 21-neuron net has an accuracy 99%. In comparison, the trained 16-neuron RNN has an accuracy 94% and empirically we found it hard to train a 21-neuron RNN with 95%+ accuracy.
> - Both the constructed and the trained net have 1 major positive entry in the input weight vector $w_{ih}$. However, the constructed net has 10 negative entries, designated to update each of the $s_k$ features. In comparison, the trained net has 4 negative input neurons which seem very entangled and this has been a major obstacle to understanding how $(s_k - m_k)$ is represented.
> - In the constructed net, the $m_k$ features are stored without updates but the max circuit of the trained net does not store them statically and or preserve the order. We have similar observations for trained RNNs on the argmax task as well.
> - As mentioned above, there are potentially a lot of ways of designing the s and m features. The trained net do not seem to implement the versions described in the code.

> 5.3 Heuristic Estimator. In our attempt, we were primarily focused on complete mechanistic interpretations, which was hindered by understanding the m (max sub-circuit) and (s − m) feature implementations. We shifted our efforts to analyzing RNNs for argmax tasks but did not uncover an explanation for nonlinearity's benefits. An unexplored direction is pursuing a higher-level mechanistic view while leveraging generic end-to-end techniques like bin propagation.

On the weight decay of this model (section 2):

> With strong weight decay, we should expect each column of $W_{oh}$ to sum to 0, implying the average of all output directions is the $\mathbf 0$ vector. However, our main RNN model for the 2nd argmax task was not trained with very strong weight decay for generalization concerns.

### 3.4 Chen's notebook (S5): statements beyond the note

Numbers printed in the notebook (inputs $\mathrm{Unif}(-1,1)$ unless noted):

- Accuracy on $10^5$ samples: `0.93792`. Accuracy by target position 0..9: `[0.9415, 0.9550, 0.9553, 0.9441, 0.9437, 0.9358, 0.9310, 0.9227, 0.9168, 0.9225]`.
- Accuracy after setting $W_{ih}[i] = 0$, $i = 0..15$: `[0.93812, 0.15397, 0.93737, 0.93777, 0.18228, 0.92804, 0.15089, 0.12008, 0.20156, 0.93207, 0.93833, 0.93522, 0.92956, 0.93808, 0.85871, 0.93791]`, with the comment
  > This shows marginally neurons (1, 4, 6 ,7, 8) are indeed the major input neurons
- Code comments on pruning: `w_ih = np.where(np.abs(w_ih) < 0.1, 0, w_ih) # no accuracy drop` and `w_hh = np.where(np.abs(w_hh) < 0.005, 0, w_hh)` under "The following ablation does not really affect the accuracy"; zeroing $W_{ih}$ outside `[1, 4, 6, 7, 8, 14]` is "The following cause 3% accuracy drop"; `w_hh = np.where(np.abs(w_hh) < 0.01, 0, w_hh)` and `w_hh[:,9] = 0` are under "The following ablations decreases the model accuracy".
- Column sums of $W_{oh}$: `[-0.23 -0.3 -0.58 -0.17 -0.78 0.05 -1.08 -0.53 0.42 -0.04 0.63 0.21 -0.09 0.07 -0.03 -0.42]`, with the comment "This model didn't use large enough WD, otherwise we should expect to see the weights are 0". Singular values of $W_{oh}$: `[19.2 17.92 8.64 6.15 5.97 4.6 3.42 2.4 2.31 0.31]`.
- Fraction of nonzero activations per unit, $\mathrm{Unif}(0,1)$ inputs: `[0.8 0.63 1. 0.69 0.31 0.54 0.59 0.7 0.56 0.28 0.94 0.93 0.98 0.9 0.88 0.63]`.

Markdown cells:

> We separate the neurons into two groups, input neurons (1,4,6,7,8) along with neuron 2 which is closely related to neuron 4 and 7, and all other neurons as the other group.

> **Identifying the Max sub-circuit.** Note that at any time t, if the incoming input value xt is large enough (xt = 1), then the old running argmax becomes the running 2nd max. Neurons [1,6,7,8] will be unactivated. This implies the sub circuit excluding these neurons forms a circuit for the max and argmax tasks. More precisely, max sub-circuit is just zeroing out a few neurons when calculating the logits. It has a decent accuracy for finding the argmax among the first d-1 numbers.

(The code gives `0.8684` on $10^4$ samples for the readout without units 1, 6, 7, 8, with comments `# excluded_indices = [1, 4, 6, 7, 8, 9] # doesn't hurt` and `# excluded_indices = [1, 4, 6, 7, 8, 9, 14] # hurt a bit`.)

> We calculate the argmax accuracy of this network. It has a decent accuracy for finding the argmax among the first d-1 numbers. Notably, further removing N4 or N9 doesn't hurt the accuracy. However, further removing N14 hurts the accuracy. We did not show the analysis of where the errors come from in this notebook, but a major part of the error is when the model output the 2nd argmax since the 2nd max value can be close to the max value at times.

> **Individual neurons analysis.** Through extensively running the logit and neurons analysis above, we have the following hypothesis that we are interested in test: 1. N2 approximately implements max_t, N4 approximately implements ReLU(x_t - max_t) and N7 approximately implements ReLU(max_t - x_t) 2. N8 represents the 2nd max value well during the earlier time steps. 3. N9 is mostly unactive.

> **Probing.** Instead of looking the activations values h_t, it is more reasonable to investigate pre-input neurons values W_hh @ h_t. […] It seems that the no-intercept R2 is the best metric for our setting, since our RNN has no bias term.

> N9 is actually not really a dead neuron. This can be further confirmed throught ablating it, which results in a 30% accuracy drop.

Code comment beside the heatmap of the cosine similarity between rows of $W_{oh} W_{hh}$ and rows of $W_{oh}$:

> `# the rotation tends to overshoot a bit, so Av_k = 1.1 v_{k-1} or something`

Figures (values read by eye from the published outputs). The first two are no-intercept $R^2$ of single pre-input values $(W_{hh} h_t)_k$ on 1000 $\mathrm{Unif}(0,1)$ sequences; the last two depend on the weights only.

- Running max from units 2, 4, 7: about 1.00 at $t \le 2$, falling to about 0.77 (unit 2), 0.87 (unit 4), 0.80 (unit 7) at $t = 9$.
- Running 2nd max from unit 8: 0.94, 0.95, 0.91, 0.73, 0.47, 0.28, 0.24, 0.23 for $t = 2..9$. Unit 1: 0.61 falling to 0.17. Unit 6: 0.49, dipping to 0.36, rising to 0.51.
- $\cos\big((W_{oh} W_{hh})[i], W_{oh}[i{+}1]\big)$ for $i = 0..8$: 0.73, 0.92, 0.98, 0.94, 0.63, 0.69, 0.94, 0.88, 0.18. This is the evidence for "previous values (inexactly) shift to earlier logits".
- The rows of $W_{oh}$ projected on its top two singular vectors lie on a loop in the order 0, 1, …, 9 (figure only; the notebook does not comment on it).

### 3.5 ARC staff comments on LessWrong (S2)

Jacob Hilton, 2026-01-28, replying to Nicholas Goldowsky-Dill's partial analysis (section 9): "Good start!"

Jacob Hilton, 2026-02-04, replying to Martin Keane:

> Can you turn this argument into a mechanistic estimate of the model's accuracy? (You'd need to do things like deduce correlations from the weights, rather than just observe them empirically - but it seems like you're getting close.)

Jacob Hilton, 2026-08-17, replying to "How is the progress six months later?":

> I don't recall seeing any specific effort on these models in the last six months outside of the comments on this post. To me, that progress is fairly marginal and we are not much closer to succeeding at the challenge given at the end of the post. (I'm excluding other research at ARC that doesn't pertain to these models specifically, but may be on the path to success.)

## 4. (b) Training recipes and how accuracy is measured

### 4.1 $M_{16,10}$ = `example_2nd_argmax()` (S5 cells 0, 3, 5, 6, 8; S4 section 2)

| item | value | where |
|---|---|---|
| inputs | "Train data is iid sampled from $\text{Unif}(-1,1)^{10}$"; fresh every step: `torch.rand(samples, seq_len, dim)*2*factor - factor`, `factor = 1` | cells 0, 5 |
| target | `src.sort(dim = 1)[1][:,-2]` (index of the second-largest) | cell 5 |
| model | `nn.RNN(input_size=1, hidden_size=16, batch_first=True, nonlinearity='relu', bias=False)` + `nn.Linear(16, 10, bias=False)` | cell 3 |
| loss | `F.cross_entropy` on the logits from the final state only | cell 5 |
| optimiser | `torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)` (Adam with L2 decay, not AdamW) | cell 5 |
| learning rate | `LR = 0.005`; `ExponentialLR(optimizer, gamma=0.995)` stepped once per epoch | cells 5, 6 |
| weight decay | `WEIGHT_DECAY = 1e-14` | cell 6 |
| length | `EPOCHS = 1000`, `total_samples_per_epoch=100000`, `steps_per_epoch = total_samples_per_epoch // samples_per_batch` | cells 5, 6 |
| batch size | **not given**: the call uses `samples_per_batch=batch_size`, and `batch_size` is never defined in the notebook | cell 6 |
| initialisation | not overridden, so PyTorch defaults **[mine: $\mathrm{Unif}(-1/\sqrt{16}, 1/\sqrt{16})$ for all three matrices]** | cell 3 |
| seed, model selection | none set (`# np.random.seed(123)` is commented out); selection not described | cell 1 |

Caveats **[mine]**. The training call in cell 6 is commented out and the notebook loads the finished weights from a file, so these are the hyperparameters shown next to the model, not a logged run. They imply about $10^8$ training sequences and a final learning rate of $0.005 \cdot 0.995^{1000} \approx 3.3 \times 10^{-5}$. The blog's footnotes 1 and 6 ("a small amount of weight decay", "trained before we standardized our codebase") agree with this. A third party asked ARC for the recipe in GitHub issue #1 (2026-10-19); ARC has not answered.

### 4.2 Zoo models `zoo_2nd_argmax(d, n)` (S3 `alg_zoo/training.py`, README)

README: "The models in the zoo were trained using `train(task_name, hidden_size=hidden_size, seq_len=seq_len, seed=seed)`, with all other arguments set to their default values." "Models were trained using PyTorch version `2.9.1+cu128` on an A100 (80G)."

| item | value |
|---|---|
| inputs | `th.randn(batch_size, seq_len)`: iid $\mathcal N(0,1)$, fresh every step |
| target, loss | `x.argsort(-1)[..., -2]`; `th.nn.functional.cross_entropy(logits, targets)` |
| optimiser | `th.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)` |
| learning rate | `lr = lr_base / hidden_size**0.5` with `lr_base=0.04`, so 0.01 for $d = 16$. Multiplier at step $k$ of $K$: $\min(k / 2^{14}, 1) \cdot \max(0.01^{2k/K}, 0.01)$: linear warm-up over $2^{14}$ steps, exponential decay to $0.01\times$ at half-time, then constant |
| batch size, length | 1024; `n_train=2**30` sequences, so $K = 2^{20}$ steps |
| bias, init | `bias=False`; PyTorch defaults |
| seeds, selection | `th.manual_seed(seed)`, seeds 0–4. "The models in the zoo are the best of 5 random seeds, since local minima can be hard to avoid for tiny models." Best seeds are hard-coded in `zoo.py`: $(2,2) \to 2$, $(4,3) \to 0$, $(16,10) \to 2$ |
| checkpoints | "Model checkpoints were saved after 0, 2^20, 2^21, ..., 2^28 and 2^29 out of 2^30 training sequences." |

### 4.3 How accuracy is measured

- Zoo: after training, the same loop runs under `th.no_grad()` on `n_val=2**24` fresh $\mathcal N(0,1)$ sequences, with `accuracy = (logits.argmax(dim=-1) == targets).float().mean()`.
- Blog: accuracies are on "standard Gaussian input sequences"; $M_{16,10}$ is 95.3%. The sample size is not stated.
- Chen: 94% "on the uniform distribution", from $10^5$ $\mathrm{Unif}(-1,1)$ samples scored against `test_x.argsort()[-2]`. The "94_acc" in the file name presumably refers to this number **[mine]**.
- Validation logs in the bucket (S8), mean over the 16 logged blocks **[mine]**: $(16,10)$ seeds 0–4: 0.9099, 0.8802, **0.9484**, 0.9320, 0.8896. $(4,3)$: **0.9845**, 0.8929, 0.7590, 0.9044, 0.9451. $(2,2)$: 0.4999, 0.99990, **0.999919** (1 error in 12,363), 0.8749, 0.5001. Bold is the zoo's best-of-5.

### 4.4 Zoo accuracies, 2nd argmax (README figure `images/zoo_2nd_argmax.svg`, read from the labels embedded in the SVG)

| hidden size \ seq len | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|
| 32 | 100.0% | 100.0% | 99.9% | 99.7% | 99.4% | 99.3% | 99.4% |
| 16 | 100.0% | 100.0% | 99.8% | 99.2% | 98.5% | 97.6% | 94.8% |
| 8 | 100.0% | 99.9% | 98.7% | 88.2% | 78.9% | 46.6% | 42.7% |
| 6 | 100.0% | 99.9% | 86.5% | 70.3% | 48.4% | 40.5% | 37.5% |
| 5 | 100.0% | 99.9% | 74.6% | 54.2% | 45.0% | 38.4% | 34.7% |
| 4 | 100.0% | 98.5% | 60.2% | 42.8% | 39.4% | 34.6% | 31.8% |
| 3 | 100.0% | 90.8% | 43.3% | 37.5% | 34.1% | 28.5% | 23.9% |
| 2 | 100.0% | 41.0% | 32.8% | 30.6% | 26.9% | 21.4% | 20.2% |

## 5. (c) How ARC explains the smaller models

### 5.1 $M_{2,2}$: 10 parameters, "fully understood" (S1)

> It has 10 parameters and almost perfect 100% accuracy, with an error rate of roughly 1 in 13,000. This means that the difference between the model's logits, $\Delta(x_0,x_1):=\mathrm{logits}(x_0,x_1)_1-\mathrm{logits}(x_0,x_1)_0$, is almost always negative when $x_1>x_0$ and positive when $x_0>x_1$.

> To do this, note first that because the model uses ReLU activations and there are no biases, $\Delta$ is a piecewise linear function of $x_0$ and $x_1$ in which the pieces are bounded by rays through the origin in the $x_0$-$x_1$ plane.
>
> Now, we can "standardize" the model to obtain an exactly equivalent model for which the entries of $W^{hi}$ lie in $\{\pm 1\}$, by rescaling the neurons of the hidden state. Once we do this, we see that
>
> $$W^{hi}=\begin{pmatrix}+1\\-1\end{pmatrix},\qquad W^{hh}\in\begin{pmatrix}[-1,0)&[1,\infty)\\ [1,\infty)&[-1,0)\end{pmatrix}\qquad\text{and}\qquad W^{oh}\in\begin{pmatrix}(0,\infty)&(-\infty,0)\\(-\infty,0)&(0,\infty)\end{pmatrix}.$$
>
> From these observations, we can prove that, on each linear piece of $\Delta$, $\Delta(x_0,x_1)=a_0x_0-a_1x_1$ with $a_0,a_1>0$ […]

> - In the two blue regions, we have $a_0<a_1$
> - In the two green regions, we have $a_0>a_1$
> - In the two yellow regions, we have $a_0\approx a_1$ to within around 1 part in $2^{11}$

> Together, these imply that the model has almost 100% accuracy. More precisely, the error rate is the fraction of the unit disk lying between the model's decision boundary and the line $x_0=x_1$, which is around 1 in $2\pi\times 2^{11}\approx 13,000$.

Surprise accounting: 14 bits (signs of all 10 parameters, and whether each of the 4 entries of $W^{hh}$ is above or below 1 in magnitude) + 4 bits (which coefficient is larger in the 4 blue and green regions) + about 22 bits (two equalities to 1 part in $2^{11}$) $\approx 40$ bits, which "plausibly matches the number of bits of optimization used to select the model". ARC stresses that this is brute force over linear regions and would not scale: "the number of regions would grow exponentially with depth".

> Finally, it is interesting to note that our analysis allows us to construct a model by hand that gets exactly 100% accuracy, by taking
> $$W^{hi}=\begin{pmatrix}+1\\-1\end{pmatrix},\qquad W^{hh}=\begin{pmatrix}-1&+1\\+1&-1\end{pmatrix}\qquad\text{and}\qquad W^{oh}=\begin{pmatrix}+1&-1\\-1&+1\end{pmatrix}.$$

**[mine] The mechanism in one line.** In the handcrafted model $h_1 = (\mathrm{ReLU}(x_0), \mathrm{ReLU}(-x_0))$ is a sign-split copy of $x_0$, $h_2 = (\mathrm{ReLU}(x_1 - x_0), \mathrm{ReLU}(x_0 - x_1))$, and $\text{logits} = (x_1 - x_0,\ x_0 - x_1)$. The numbers for the zoo model (seed 2), standardised: $W^{hh} = \begin{pmatrix}-0.382 & 2.085\\ 1.142 & -0.752\end{pmatrix}$, $W^{oh} = \begin{pmatrix}39.82 & -173.53\\ -39.81 & 173.16\end{pmatrix}$. For $x_0 > 0$ unit 0 is active iff $x_1 > 0.382\,x_0$ and unit 1 iff $x_1 < 1.142\,x_0$; for $x_0 < 0$ the slopes are 2.085 and 0.752. These four rays and the $x_1$ axis are the region boundaries of the blog's figure. The two regions where both units are active (yellow) contain the line $x_0 = x_1$ and have $a_0 / a_1 = 0.999673$ and $1.000646$, which gives an error rate of $(0.000327 + 0.000646)/(4\pi) = 1/12{,}900$. Sampling $2^{24}$ inputs gives 1 in 13,128. Where only one unit is active, $a_0/a_1$ is 0.382 and 0.752 in the two regions next to the $x_1$ axis (blue) and 1.142 and 2.085 in the other two (green).

### 5.2 $M_{4,3}$: 32 parameters, 98.5% (S1, S6, S7)

Blog:

> The model $M_{4,3}$ can be loaded in AlgZoo using `zoo_2nd_argmax(4, 3)`. It has 32 parameters and an accuracy of 98.5%.
>
> Our analysis of $M_{4,3}$ is broadly similar to our analysis of $M_{2,2}$, but the model is already deep enough that we wouldn't consider a fully brute force explanation to be adequate. To deal with this, we exploit various approximate symmetries in the model to reduce the total number of computational operations as well as the surprise of the explanation.

> In the second set of notes, we provide two different mechanistic estimates for the model's accuracy that use different amounts of compute, depending on which approximate symmetries are exploited. […] We find that we are able to roughly match the computational efficiency of sampling,[4] and we think we more-or-less have a full understanding, although this is less clear. Finally, our analysis once again allows us to construct an improved model by hand, which has 99.99% accuracy.[5]

> [4] To assess the mean squared error of our method, we add noise to the model's weights and check the squared error of our method on average over the choice of noisy model.

**The symmetry (Robinson, S7, section 1; Hilton applies it to $M_{4,3}$).** For $n = 3$ the 2nd argmax is the argmedian, so Robinson's argmedian analysis is about the same task.

> The symmetries we've noticed of the argmedian and argoutlier models (of sequence length 3 and width 4) are equivalent to the equations $w_{hh} = U^T w'_{hh} U$, $w_{ih} = U^T w'_{ih}$, $w_{ho} = w'_{ho} V$ where $U = \begin{pmatrix}1&0&0&-1\\0&1&-1&0\end{pmatrix}$, $V = \begin{pmatrix}1&0&0&1\\0&1&1&0\end{pmatrix}$ and $w'_{hh}, w'_{ih}, w'_{ho}$ are 2x2, 2x1, 3x2 matrices respectively. Notice that $U\,\mathrm{ReLU}(U^T v) = v$, $V\,\mathrm{ReLU}(U^T v) = |v|$ where by $|v|$ I mean coordinate-wise absolute value rather than norm. Therefore the function defined by the RNN is […] $= w'_{ho}\,\big|\,w'_{ih} x_2 + w'_{hh}(w'_{ih} x_1 + w'_{hh}(w'_{ih} x_0))\,\big|$ […] So, RNNs of this size with the symmetries above are generically equivalent to a (3,2,3) neural network with activation function given by absolute value.

So the four units are two sign-split pairs. Each pair carries one signed linear function of the inputs through the recurrence without any nonlinearity, and the only nonlinearity left is one absolute value before the readout: $\text{logits} = \mathrm{fc2}\,|\mathrm{fc1}\,x|$.

**Robinson's normal form for the argmedian abs-network (S7, section 3).** These are feedforward abs-activation networks trained directly.

> $$\mathrm{fc1} = \begin{pmatrix}1 & -(\lambda+1) & \lambda\\ 1 & \lambda & -(\lambda+1)\end{pmatrix},\qquad \mathrm{fc2} = \begin{pmatrix}\epsilon & \epsilon\\ -\mu & \mu-\epsilon\\ \mu-\epsilon & -\mu\end{pmatrix}.$$ The specific values we get are $\lambda \approx 7.808$, $\epsilon \approx 0.7$, and $\mu \approx 17.79$.

- Both pre-activations positive $\iff x_0 > \frac{x_1+x_2}{2} + \frac{2\lambda+1}{2}|x_1 - x_2|$ ("With our values the coefficient here is approximately 8.308"). There $x_0$ is the largest and the median is the larger of $x_1, x_2$. The second logit is the largest "precisely when $x_1 - x_2 > \frac{6\epsilon}{(2\lambda+1)(2\mu-\epsilon)}\left(x_0 - \frac{x_1+x_2}{2}\right)$. This ratio is 0.007", so "the only interval of failure here is when $|x_1 - x_2| < \frac{6\epsilon}{(2\lambda+1)(2\mu-\epsilon)}\left(x_0 - \frac{x_1+x_2}{2}\right)$".
- In the $(+,-)$ quadrant ($x_2 > x_1$ and $\left|x_0 - \frac{x_1+x_2}{2}\right| < \frac{2\lambda+1}{2}|x_1 - x_2|$) the second logit beats the first iff $x_2 - x_1 < \frac{2(2\mu-\epsilon)}{3\epsilon(2\lambda+1)}\left(\frac{x_1+x_2}{2} - x_0\right)$. "This ratio is almost exactly 2, which gives perfect accuracy." **[mine: with ratio exactly 2 the condition is $x_0 < x_1$, which is the true condition for $x_1$ to be the median there.]**

**Hilton's estimate without the symmetry (S6, "Original version").** ARC indexing, $z_{t+1} = W^{hh} h_t + W^{hi} x_t$.

> $M_{4,3}$ happens to have the property that $W^{hi}_0, W^{hi}_1 > 0$ and $W^{hi}_0, W^{hi}_1 < 0$.

(A typo in the source: the second pair must be $W^{hi}_2, W^{hi}_3 < 0$, as the next equation shows.) Then $h_1(x_0) = (W^{hi}_0 x_0, W^{hi}_1 x_0, 0, 0)$ if $x_0 \ge 0$ and $(0, 0, W^{hi}_2 x_0, W^{hi}_3 x_0)$ if $x_0 \le 0$, and $z_2 = (a_j x_0 + c_j x_1)_j$ or $(b_j x_0 + c_j x_1)_j$, with

$$a_j = W^{hh}_{j0} W^{hi}_0 + W^{hh}_{j1} W^{hi}_1,\qquad b_j = W^{hh}_{j2} W^{hi}_2 + W^{hh}_{j3} W^{hi}_3,\qquad c_j = W^{hi}_j.$$

> As it turns out, we have $a_3 \approx -a_0 > 0$, $a_2 \approx -a_1 > 0$, $b_3 \approx -b_0 > 0$, $b_2 \approx -b_1 > 0$, $c_0 \approx -c_3 > 0$, $c_1 \approx -c_2 > 0$

and "moreover $a_j \approx b_j$". With $\alpha_3 := \frac{a_3 + (-a_0)}{2}$ and so on, $h_2$ is approximated as linear on 4 regions of the $(x_0, x_1)$ plane, bounded by $x_0 = \frac{\gamma_0}{\alpha_3} x_1$ and $x_0 = \frac{\gamma_1}{\alpha_2} x_1$ and their $\beta$ versions ("it happens that $\frac{\gamma_0}{\alpha_3}, \frac{\gamma_0}{\beta_3} < 1$ and $\frac{\gamma_1}{\alpha_2}, \frac{\gamma_1}{\beta_2} > 1$"). Crossing with the 6 orderings of $(x_0, x_1, x_2)$ gives "14 likely subregions plus 4 unlikely subregions". In each likely subregion the last ReLU is handled by a presumption of independence: $\mathbb 1_{z_3 > 0}$ is replaced by a two-valued variable that matches its mean and variance and depends on a latent $Z \sim \mathrm{Bernoulli}(q)$, "where $q = 0.2$ is a hyperparameter". Unlikely subregions get accuracy $\frac13$. All probabilities are solid angles of intersections of half-spaces through the origin, computed with "the polygonal generalization of Girard's theorem": $18 + 56 + 28 = 102$ volumes.

**Hilton's second estimate ("Version based on George's approach").** Write $\varepsilon_{hi} = W_{hi} - U^T W'_{hi}$, $\varepsilon_{hh} = W_{hh} - U^T W'_{hh} U$, $\varepsilon_{oh} = W_{oh} - W'_{oh} V$, with the primed matrices fitted by averaging, and propagate the mean and covariance of the deviation from the symmetric network through the layers (covariance propagation, every variable treated as a zero-mean Gaussian). Then there are 24 subregions (4 sign patterns of the two absolute values times 6 orderings); their volumes in 6 dimensions were obtained by sampling, "We avoid the problem of figuring out how to do this exactly for now".

**Results (S6).**

> To test these methods of estimating the accuracy of $M_{4,3}$, we added zero-mean Gaussian noise with a standard deviation of 0.05 to the weights of network and ran the two estimators on 256 such models. The original version achieves a RMS error of around 3%, and the version based on George's approach achieves a RMS error of around 5%, although it appears to have lower bias.

Costs: about $5(18 \cdot 4^3 + 56 \cdot 5^3 + 28 \cdot 6^3) = 71000$ and $5 \cdot 24 \cdot 4^3 = 7680$ operations. Sampling costs $72k$ operations for $k$ samples with RMS error about $24\%/\sqrt k$, so "with the same computational budget as our two methods, sampling should achive an RMS error of around 0.8% and 2.3% respectively. So we are underperforming sampling by around a factor of 2 in terms of error (i.e, a factor of 4 in terms of compute)." Surprise accounting: "roughly 16 bits" for the inequalities plus an unclear amount for the 102 volumes, and "This all seems confusing enough that we should perhaps just stick to quality versus compute for evaluating explanations".

**[mine] Numbers for the zoo $M_{4,3}$ (seed 0).** Standardised, in its stored unit order, $W^{hi} = (1, 1, -1, -1)$. Hilton's quantities are $a = (-1.268, -0.807, 0.788, 1.239)$, $b = (-1.240, -0.786, 0.809, 1.265)$, $\gamma_0/\alpha_3 = 0.798$, $\gamma_1/\alpha_2 = 1.254$, so all his stated properties hold. Robinson's symmetry holds with relative residuals 0%, 10%, 0.1% for $W^{hi}, W^{hh}, W^{oh}$, and the exactly symmetric network has accuracy 0.9840 against 0.9846. In Robinson's normal form $\mathrm{fc1} = \begin{pmatrix}1 & -4.76 & 3.80\\ 1 & 4.16 & -5.22\end{pmatrix}$, $\mathrm{fc2} = \begin{pmatrix}1.38 & 0.89\\ -16.43 & 13.85\\ 15.06 & -14.74\end{pmatrix}$, so $\lambda \approx 4.0$, $\epsilon \approx 1.13$, $\mu \approx 15.6$; the two ratios are 0.025 and 1.97. 87% of the symmetric network's errors lie in the $(+,+)$ and $(-,-)$ quadrants (12% of inputs, accuracy 0.886 there), and nearly all of them answer position 0. That is Robinson's failure sliver: when $x_0$ is the extreme value and $x_1 \approx x_2$, the network says $x_0$ is the median.

### 5.3 Handcrafted models (S3 `alg_zoo/handcrafted.py`)

**$n = 3$** (footnote 5: "Credit to Michael Sklar for correspondence that led to this construction"). The code, condensed (tensor wrappers removed):

```python
# accuracy converges to 100% as lam -> infinity, but only at high precision
# lam = 100 works well for float32
lam = 100
fc1 = [[1, -lam - 1, lam], [-1, -lam, lam + 1]]
fc2 = [[1, 1], [-lam, lam + 1], [lam + 1, -lam]]
U = [[1, 0, 0, -1], [0, 1, -1, 0]];  V = [[1, 0, 0, 1], [0, 1, 1, 0]]
weight_ih = U.T @ fc1[:, 2:]
weight_hh = U.T @ fc1[:, :2] @ inv(fc1[:, 1:]) @ U
linear.weight = fc2 @ V
```

**[mine]** This is $\text{logits} = \mathrm{fc2}\,|\mathrm{fc1}\,x|$ with $u_0 = (x_0 - x_1) + \lambda(x_2 - x_1)$ and $u_1 = (x_2 - x_0) + \lambda(x_2 - x_1)$. When both have the sign $\sigma$ of $x_2 - x_1$, which fails only in a sliver of relative width about $1/\lambda$,

$$\text{logits} = (2\lambda + 1)\,\big(|x_2 - x_1|,\ \sigma(x_2 - x_0),\ \sigma(x_0 - x_1)\big),$$

whose argmax is the median exactly. It is the same normal form as the trained model, with $\lambda = 100$ in place of about 4. Measured in float64: the signs agree on 99.45% of $\mathcal N(0,1)$ inputs, the formula holds there to $10^{-11}$, and the overall accuracy is 0.99997.

**$n = 10$, 22 units.** The comments in the code, in ARC's indexing:

```python
# neurons 0-9 are max(0, x_{t-10}), ..., max(0, x_{t-1})
# neuron 19 is max(0, x_0, ..., x_{t-1}) - max(0, x_0, ..., x_{t-2})
# neuron 20 is max(0, x_0, ..., x_{t-2})
# neuron 21 is max(0, x_0, ..., x_{t-1}) - x_{t-1}
# neuron 18 is max(0, x_0, ..., x_{t-3}, x_{t-1}) - x_{t-1}
# neurons 10-17 are max(0, x_1, ..., x_{t-1}) - x_{t-1}, ..., max(0, x_0, ..., x_{t-4}, x_{t-2}, x_{t-1}) - x_{t-1}
# the ith logit is max(0, x_i) - 2 * max(0, x_0, ..., x_{i-1}, x_{i+1}, ..., x_9)
```

Weights: units 0–9 are a shift register ($W_{ih}[9] = 1$, $W_{hh}[i, i{+}1] = 1$). Units 19, 20, 21, 18 are the trained network's units 4, 2, 7, 6 with ARC's idealised weights. Units 10–17 form a chain, $W_{hh}[i, i{+}1] = 1$, $W_{hh}[i, 21] = -1$, $W_{hh}[i, 20] = W_{hh}[i, 19] = 1$, $W_{ih}[i] = -1$, which is exactly the rule ARC gives for the trained unit 1 ($+n_2 + n_4 + n_6 - n_7 - x$), repeated. Readout: $W_{out}[i, i] = 1$; for $i < 9$, $W_{out}[i, 10{+}i] = 2$, $W_{out}[i, 21] = -2$, $W_{out}[i, 20] = W_{out}[i, 19] = 2$; $W_{out}[9, 20] = 2$.

**[mine]** Writing $L_i := \max(0, x_j : j \ne i)$ for the leave-one-out maximum, the weights give

$$\text{logit}_i = \max(0, x_i) + 2 L_i$$

(checked numerically to $7 \times 10^{-15}$; the "- 2 *" in the comment is a sign typo, and with a minus sign the network would compute the argmax). If the top three values are $a > b > c$ with $b > 0$, the logits are $a + 2b$ at the argmax, $b + 2a$ at the 2nd argmax and at most $c + 2a$ elsewhere, so the 2nd argmax wins. If fewer than two inputs are positive, nine logits tie and the answer is right about 1 time in 10. Measured: accuracy 1.00000 with $\ge 2$ positive inputs, 0.098 with 1 or 0; $P(\text{fewer than 2 positive}) = 11/1024$; overall 0.9902 on both $\mathcal N(0,1)$ and $\mathrm{Unif}(-1,1)$. Correspondence with the trained network: 20 ↔ 2, 19 ↔ 4, 21 ↔ 7, 18 ↔ 6, 17 ↔ 1. For the other seven chain units and the ten shift-register units ARC gives no counterpart in the trained network; that is their second "unfortunately" bullet.

### 5.4 Chen's distilled algorithm and 21-unit network (S4 section 5.2, S5 cells 42–48)

> The high-level idea is to maintain a set of features, $m_k$ and $s_k$, for representing the scores of each index as the max and the 2nd max index, respectively. They correspond to the intermediate logit values and the projected logit values in the trained network. In other words, $\arg\max m_k$ is the max index, and $\arg\max s_k$ is the 2nd max index at each time step.
>
> When the new input $x_t$ is larger than the running max, i.e., $x_t > \max_t$, we need to update the s value for the current argmax logit. Mimicking the trained network can be achieved by shifting the m values to the s values.

Listing 1 of the note, with line breaks condensed and no code changed:

```python
def distilled_algo(test_x):
    s = [0 for _ in range(10)]; m = [0 for _ in range(10)]; max_t = 0
    for t in range(10):
        for k in range(9):            # rotate
            s[k] = s[k+1]; m[k] = m[k+1]
        xt = test_x.squeeze()[t]      # update
        m[9] = xt
        s[9] = max_t - abs(max_t - xt)
        # other choices can be
        # m[9] = max_t -  max(xt - max_t, 0)
        # m[9] = max(xt, 0)
        # s[9] = max(max_t - xt, 0)
        if xt > max_t:                # exact shifting version
            for k in range(9):
                s[k] = m[k]
        # approximate shifting version
        # for k in range(9):
        #     s[k] = m[k] - 2 * max(max_t - xt, 0)
        #                 + 2 * max(max_t - m[k] / 2 + s[k] / 2 - xt, 0)
        max_t = max(xt, max_t)
    return max(enumerate(s), key=lambda x: x[1])[0]
```

Accuracy 0.99781 on $10^5$ uniform samples; Chen's count is $1 - (2^{-10} + 0.1 \cdot 10 \cdot 2^{-10}) \approx 0.998$, with all failures on sequences with at most one positive entry ("This algorithm always computes the argmax instead of the 2nd argmax when the entire sequence is negative"). "The approximate shifting version can be exactly implemented by a ReLU RNN"; his 21-unit construction gets 0.9886 on $10^4$ samples.

**[mine]** How the approximate shift works, with $g := \max_t - x_t$: if $g \le 0$ (new maximum) both ReLUs vanish and $s_k \leftarrow m_k$; if $g \ge (m_k - s_k)/2$ then $s_k \leftarrow s_k$; in between $s_k \leftarrow m_k - 2g$. So it is a soft step in $g$ built from two ReLUs. The first, $\mathrm{ReLU}(\max_t - x_t)$, is shared by all $k$ (it is the counterpart of unit 7); the second needs one input-driven unit per older position. These are the "10 negative entries" of the constructed $w_{ih}$ ($1 + 9$), against the trained network's 4.

### 5.5 Linear RNNs for argmax (S4 section 3)

Without the ReLU, $y = Ax$ with $A = W_{oh}\,[\,W_{hh}^{d-1} w_{ih}\ \ W_{hh}^{d-2} w_{ih}\ \cdots\ w_{ih}\,]$, and argmax is solved iff $A = cI + \mathbf 1 u^T$.

> The specific values of the RNN weights $W_{hh}$ and $w_{ih}$ are not crucial, as long as the concatenated matrix […] has full rank. […] Empirically, $W_{hh}$ tends to be a cyclical shift in the directions specified by the output matrix $W_{oh}$.

On nonlinear argmax RNNs (section 4) Chen reports no explanation: the heuristic estimators he tried "fail to explain the advantages provided by nonlinearity, instead, viewing it as a distortion from the linear case".

## 6. (d) ARC's standard for an explanation, and the challenge

S1:

> Our current perspective at ARC is that, given a model that has been optimized for a particular loss, an "explanation" of the model amounts to a **mechanistic estimate of the model's loss**. We evaluate mechanistic estimates in one of two ways. We use *surprise accounting* to determine whether we have achieved a full understanding; but for practical purposes, we simply look at mean squared error as a function of compute, which allows us to compare the estimate with sampling.

> By "mechanistic", we mean that the estimate reasons deductively based on the structure of the model, in contrast to a sampling-based estimate, which makes inductive inferences about the overall performance from individual examples.

> **Mean squared error versus compute.** […] Our matching sampling principle is roughly the following conjecture: there is a mechanistic estimation procedure that (given suitable advice) performs at least as well as random sampling in mean squared error for any given computational budget.

> **Surprise accounting.** This is an information-theoretic metric that asks: how surprising is the model's actual accuracy, now that we have access to the mechanistic estimate? We accrue surprise in one of two ways: either the estimate itself performed some kind of calculation or check with a surprising result, or the model's actual accuracy is still surprising even after accounting for the mechanistic estimate and its uncertainty.

> Surprise accounting is useful because it gives us a notion of "full understanding": a mechanistic estimate with as few bits of total surprise as the number of bits of optimization used to select the model.

The challenge:

> **Challenge**: Design a method for mechanistically estimating the accuracy of the 432-parameter model $M_{16,10}$[8] that matches the performance of random sampling in terms of mean squared error versus compute. A cheap way to measure mean squared error is to add noise to the model's weights (enough to significantly alter the model's accuracy) and check the squared error of the method on average over the choice of noisy model.[9]

> [3] An estimate for a single model could be close by chance, so the method should match sampling on average over random seeds.

> [9] A better but more expensive way to measure mean squared error is to instead average over random seeds used to train the model.

> The analysis we have presented here is relatively traditional mechanistic interpretability, but we think of this analysis mainly as a warm-up. Ultimately, we seek a scalable, algorithmic approach to producing mechanistic estimates […] Thus our approach is best described as "ambitious" and "mechanistic", but perhaps not as "interpretability".

S9, the matching sampling principle. For an estimator $\mathbb G_M(\theta, \pi, c, \varepsilon)$ with explanation $\pi$, $|\pi| \le O(|\theta|)$: it "runs in time $O\left(\frac{1}{\varepsilon^2}\mathrm{Time}(M_\theta)\right)$", is "competitive with sampling, on average over random $c$": $\mathbb E_c[(\mathbb G_M(\theta,\pi,c,\varepsilon) - \mathbb E_x[M_\theta(c,x)])^2] \le \varepsilon^2\,\mathbb E_c[\mathrm{Var}_x[M_\theta(c,x)]]$, and "is mechanistic". On that word:

> We do not have a formal definition of "mechanistic." But, loosely speaking, we mean that $\mathbb G_M$ estimates the expected output of $M_\theta$ **deductively, based on the structure of $M_\theta$.** […] A simple, though not entirely correct, heuristic for whether an estimation algorithm is deductive, is whether it avoids any random or pseudorandom sampling.

S10: "**Total surprise = surprise of explanation + surprise given explanation**", where the first term asks "how many times did the explanation just 'work out' for no apparent reason? Or put another way, how many free parameters did the explanation have?", and "we can compare the total surprise of an explanation with the amount of optimization applied to produce the phenomenon. If the two match, then there is no reason to expect there to be a better explanation."

Two further ARC remarks that set the bar. Hilton to Keane (section 3.5): correlations must be deduced "from the weights, rather than just observe[d] empirically". Hilton on the White-Box Estimation Challenge post, 2026-06-03 (https://www.lesswrong.com/posts/Kben8CzS4awCwNw5c), on why fitted predictors are not the goal:

> once you have a well-defined distribution over trained models, people can spend a lot of compute offline empirically fitting predictors for the expected output as a function of the model parameters, and it would be much harder to outperform these with mechanistic approaches.

**[mine]** For sampling, $k$ samples give squared error $p(1-p)/k$, with $p(1-p) = 0.045$ at $p = 0.953$, at a cost of $k$ forward passes of 10 steps of a $16 \times 16$ matrix-vector product. A mechanistic account of how the network works is therefore necessary but not sufficient for ARC: it must also yield the number 0.953 deductively and cheaply.

## 7. (e) Other ARC-authored hints that bear on $M_{16,10}$

1. **Positive homogeneity.** S1 (for $M_{2,2}$, but general): the output is piecewise linear with pieces "bounded by rays through the origin". S4:
   > Since ReLUs are positive homogeneous functions, i.e., $\mathrm{ReLU}(\alpha x) = \alpha\,\mathrm{ReLU}(x)$ for any $\alpha > 0$, the network's computations are equivariant to positive scaling of the input sequence. […] This implies that the network's behavior generalizes beyond the [−1, 1] input range, maintaining the same accuracy across all symmetric uniform distributions.

   **[mine]** Consequently $h_s(\alpha x) = \alpha h_s(x)$: no direction of state space can hold a constant, and any offset of a state-space structure has to be proportional to stored input values. ARC says nothing more specific about this.
2. **Unit-rescaling symmetry** (S4): for a positive diagonal $D$, $W'_{hh} = D W_{hh} D^{-1}$, $w'_{ih} = D w_{ih}$, $W'_{oh} = W_{oh} D^{-1}$ gives $h'_t \equiv D h_t$ and the same output. ARC's "standardization" is the choice $|w'_{ih}| = 1$. **Softmax shift**: $W_{oh} + \mathbf 1 u^T$ has the same output distribution.
3. **Leading zeros** (S4):
   > Since the network does not have bias terms, the model output after receiving the input sequence $(0, 0, ..., 0, x_1, x_2, ..., x_t)$ is identical to receiving $(x_1, x_2, ..., x_t)$ at time t. This implies the intermediate logit values should reflect the running classification result. […] However, the intermediate accuracies may not be monotonically decreasing since we don't explicitly train for the accuracy when the inputs start with 0s

   **[mine]** So $W_{oh} h$ read at any step is the answer in end-aligned coordinates, and the stored answer has to move one output position per step. Chen's finding 5 and the cosine similarities in section 3.4 say that the network does this.
4. **Negative inputs** (S4):
   > Note that with high probability, the max and the 2nd max values are positive. Therefore, the neural networks are not incentivized to perform well on purely negative sequences, an effect further enhanced by the ReLU activation. Consequently, we expect the trained networks to perform well on benign distribution shifts like $\mathrm{Unif}[0, 1]^d$ but not on $\mathrm{Unif}[-1, 0]^d$.

   **[mine]** All of ARC's feature formulas contain $\max(0, \cdot)$, and both handcrafted networks fail only when fewer than two inputs are positive.
5. **Failure cases.** ARC gives no error analysis of $M_{16,10}$. The only remarks: accuracy is lower for late target positions (list in section 3.4), and, for the max sub-circuit, "a major part of the error is when the model output the 2nd argmax since the 2nd max value can be close to the max value at times".
6. **Rotation.** The only ARC text is the code comment "the rotation tends to overshoot a bit, so Av_k = 1.1 v_{k-1} or something", the variable names `rotation_correlations` and `shifting_simiarity`, the remark about linear argmax RNNs ("$W_{hh}$ tends to be a cyclical shift in the directions specified by the output matrix $W_{oh}$"), and the unlabelled loop of $W_{oh}$ rows in a figure. ARC never describes oscillating modes, phases, circles or clocks.
7. **Capacity** (table in 4.4): at $n = 10$, 8 units reach 42.7%, 16 units 94.8%, 32 units 99.4%; Chen "found it hard to train a 21-neuron RNN with 95%+ accuracy". **[mine]** So a 16-unit network at about 95% has to be a compressed, lossy solution: ARC's exact construction needs 22 units.
8. **The zoo's own $(16,10)$ models** (seeds 0–4, 88.0–94.8%) are "probably fairly similar, but we have not analyzed it". A good method "should also work well on" them.

## 8. My checks of ARC's statements against the weights [mine]

All from `verify_arc_claims.py`; inputs $\mathcal N(0,1)$ unless noted.

**Standardised weights.** With $D_i = 1/|W_{ih,i}|$ for units 1, 4, 6, 7, 8 and $D_2$ chosen so that $W'_{hh}[2,4] = 1$ (ARC does not say how unit 2 is scaled). $W_{ih} = (0.069, -10.565, 0.015, -0.123, 10.157, -0.281, -10.996, -13.169, -12.312, -0.228, 0.060, 0.148, 0.298, -0.047, -1.321, -0.119)$.

| block | ARC | actual |
|---|---|---|
| $W'_{hh}[(2,4),(2,4)]$ | $\begin{pmatrix}+1&+1\\-1&-1\end{pmatrix}$ | $\begin{pmatrix}0.969&1.000\\-0.852&-0.986\end{pmatrix}$ |
| $W'_{hh}[(6,7),(2,4)]$ | $\begin{pmatrix}+1&0\\+1&+1\end{pmatrix}$ | $\begin{pmatrix}0.855&-0.044\\0.899&1.016\end{pmatrix}$ |
| $W'_{hh}[1,(2,4,6,7)]$ | $(+1,+1,+1,-1)$ | $(0.838, 0.838, 1.036, -1.306)$ |
| not stated by ARC | | $W'_{hh}[6,6] = 0.361$, $W'_{hh}[1,1] = 0.409$, $W'_{hh}[6,7] = -0.151$, $W'_{hh}[7,6] = -0.152$ |
| unit 8 (not in the blog) | | $W'_{hh}[8,(2,4,6,7,1,8)] = (0.893, -0.034, 0.000, -1.170, 0.544, 0.615)$ |

The product $W_{hh}[2,4]\,W_{hh}[4,2]$ is scale-invariant and equals $-0.852$, not $-1$, so no rescaling makes both off-diagonal entries of the (2,4) block $\pm 1$. If unit 8 continued ARC's leave-one-out chain, its weights on units (2, 4, 6, 7, 1, 8) would be $(+1, +1, 0, -1, +1, 0)$; it shares the signs on 2, 7 and 1 but has no weight on unit 4 and a self-loop of 0.62.

**How isolated is the subcircuit?** Root-mean-square size of each term of the pre-activation over steps 1–9 (scale-invariant):

| target | input term | largest recurrent terms (source unit: rms) | total |
|---|---|---|---|
| 2 | 0.01 | 2: 20.1, 4: 7.8, 8: 2.2, 6: 1.5, 1: 1.4 | 22.6 |
| 4 | 10.2 | 2: 10.2, 4: 4.4, 6: 1.4, 8: 1.2 | 16.0 |
| 7 | 13.2 | 2: 14.0, 4: 5.9, 6: 2.7, 8: 2.4, 7: 1.5, 1: 1.4 | 21.0 |
| 6 | 11.0 | 2: 11.1, 6: 5.3, 7: 2.4, 1: 1.9, 8: 1.2, 14: 1.1 | 16.5 |
| 1 | 10.6 | 7: 20.0, 6: 14.6, 2: 10.4, 1: 5.8, 4: 3.9, 8: 1.7 | 15.9 |
| 8 | 12.3 | 7: 20.9, 2: 13.0, 1: 9.0, 8: 8.9 | 17.0 |

The summed recurrent input from the other ten units has an rms of 4% (unit 2), 5% (7), 7% (4), 8% (6), 10% (1) and 11% (8) of the rms of the unit's total pre-activation. So $\{1, 2, 4, 6, 7, 8\}$ is a nearly closed subsystem: ARC's "no strong connections" holds at the 4–11% level, and the larger deviations from ARC's idealised weights are inside the six (unit 8 into 2, 4 and 7; the self-loops of 6, 1 and 8). Unit 9 is driven by the six alone (the other ten supply 1.6% of its pre-activation). For the remaining nine units the same ratio is 0.36–1.27, so they mix input from both groups.

**Feature formulas on the activations** (best no-intercept fit $h_s[u] \approx c f$; centred $R^2$ pooled over $s$, and its range over $s$):

| unit | feature $f$ (our indexing) | $c$ | $R^2$ | range over $s$ |
|---|---|---|---|---|
| 2 | $M_{s-1}$ | 17.32 | 0.979 | 1.00 for $s \le 4$, 0.91–0.96 for $s \ge 6$ |
| 4 | $M_s - M_{s-1}$ | 10.30 | 0.995 | 0.98–1.00 |
| 7 | $\mathrm{ReLU}(M_{s-1} - x_s)$ | 12.71 | 0.987 | 0.97–1.00 |
| 6 | $\mathrm{ReLU}(M_{s-2} - x_s)$ | 10.49 | 0.974 | 0.95–1.00 |
| 1 | $\mathrm{ReLU}(\max(M_{s-3}, x_{s-1}) - x_s)$ | 9.90 | 0.943 | 0.91–1.00 |
| 8 | $\mathrm{ReLU}(\max(0, \text{2nd max of } x_0..x_{s-1}) - x_s)$ (Chen) | 13.63 | 0.887 | 0.96–1.00 for $s \le 5$, then 0.90, 0.81, 0.76, 0.75 |

Chen tests "N2 approximately implements max_t" on the pre-input value $(W_{hh} h_t)_2$, which is consistent with the blog's lagged formula. Read as a statement about the activation $h_s[2]$ and the running max including $x_s$, it would give $R^2 = 0.73$.

**The same claims as thresholds.** Unit $i$ compares $x_s$ with $\theta_i(h_{s-1}) := -(W_{hh} h_{s-1})_i / W_{ih,i}$, a linear readout of the previous state, in input units.

| claim | slope | $R^2$ | mean $\lvert\theta - f\rvert$ | mean $\theta - f$ at $s = 1, 5, 9$ |
|---|---|---|---|---|
| $\theta_4 \approx M_{s-1}$ | 0.935 | 0.983 | 0.067 | $-0.004, -0.070, -0.112$ |
| $\theta_7 \approx M_{s-1}$ | 0.948 | 0.969 | 0.067 | $+0.008, -0.049, -0.112$ |
| $\theta_6 \approx M_{s-2}$ | 0.929 | 0.945 | 0.115 | $-0.022, -0.016, -0.151$ |
| $\theta_1 \approx \max(M_{s-3}, x_{s-1})$ | 0.902 | 0.869 | 0.168 | $-0.065, -0.106, -0.125$ |
| $\theta_8 \approx$ 2nd max (Chen) | 1.207 | 0.727 | 0.183 | $-0.016, +0.118, +0.339$ |

The thresholds for the maximum sag below it as steps pass, and unit 8's threshold drifts above the 2nd max. For the prefix $(0.77, 0.90, 0.79)$ of the notebook's example the thresholds for $x_3$ are 0.875 (unit 4), 0.907 (7), 0.830 (6), 0.771 (1), 0.882 (8), against the idealised 0.90, 0.90, 0.90, 0.79 and Chen's 0.79.

**Notebook numbers reproduced.** The $W_{ih}$ ablation table agrees within 0.003 everywhere; the cosine row $(0.73, 0.92, 0.98, 0.94, 0.63, 0.69, 0.94, 0.88, 0.18)$, the column sums and singular values of $W_{out}$ and the activation frequencies agree to the printed digits. Max sub-circuit: 0.8645 without units 1, 6, 7, 8; 0.8676 without 1, 4, 6, 7, 8, 9; **0.4262** when unit 14 is removed too (the notebook's "hurt a bit" is an understatement). Unit 9: clamping it to 0 gives 0.652 against 0.938 (the "30% accuracy drop"); removing only its readout column gives 0.930.

**Chen's general remarks.** Reading $W_{out} h_s$ at step $s$ and shifting by $9 - s$ gives the 2nd argmax of the prefix with probability 0.93–0.97 for every $s = 1..9$, given at least two positive inputs so far. Accuracy by number of positive inputs: 0.134 (none), 0.239 (one), 0.9604 (two or more). Accuracy on $\mathrm{Unif}(0,1)$ is 0.891 and on $\mathrm{Unif}(-1,0)$ 0.120, so the "benign" shift costs 5 points.

## 9. Non-ARC material attached to ARC's venues (pointers only; not verified)

- Nicholas Goldowsky-Dill, LessWrong comment 2026-01-28: for units 1, 6, 7, 8, "$W_{hh} h_{n,t} \approx a_n(\delta) \cdot M_t + b_n(\delta) \cdot \max(S_t, 0)$ where $M_t$ is the running max, $S_t$ is the second running max, and $\delta$ represents how long ago the max-value occurred", $R^2 = 0.992$. Notebook: https://gist.github.com/NixGD/2651be79bfd22bcb13d1f609f23439a4 (not read).
- Martin Keane, LessWrong comments 2026-01-31 to 02-05: groups "Comparators", "Wave neurons", "Bridge neurons", "Special cases: n2, n4, n9"; "Zeroing out all the inputs except the largest two gets the network to 100%"; a "Phase Wheel" ("a spiral trajectory through time, implemented by a rotating phase in the hidden state. The W_out projection converts phase angle to position logits") plus a "Discrimination Offset" $h_{\text{forward}} = h_{\text{reverse}} + f(m) - f(s)$ of effective rank about 2. Gist: https://gist.github.com/mrsirrisrm/d6850ff8647d1ed2f67cc92d5bce3ed0 (not read).
- GitHub issue #1, "How was example_2nd_argmax() trained?" (user jsnider3, 2026-10-19, open, no ARC reply): 1,024 networks trained with the zoo recipe; the example model's median logit margin (8.2) is larger than all of them (median 4.35). The same author's post: https://www.lesswrong.com/posts/sAhFp6gYoyw3xeCaF/brute-force-progress-on-algzoo-s-432-param-interpretability, which guesses batch size 128.

## 10. Caveats and open points

- The batch size, seed and selection criterion of $M_{16,10}$ are unpublished, and the notebook's hyperparameters sit next to a commented-out training call.
- ARC's "approximately" is never quantified in the sources; the numbers in section 8 are mine.
- Hilton's and Robinson's notes date from December 2024, a year before the zoo was retrained. The zoo's $M_{4,3}$ has the structure Hilton describes (section 5.2), but I cannot confirm it is the same checkpoint. The bucket also holds an unreferenced `alg/zoo_old/` folder dated 2025-12-11.
- Robinson's parameter values ($\lambda \approx 7.808$ and so on) are for directly trained abs-activation networks, not for the RNN $M_{4,3}$.
- The values I report from Chen's figures are read by eye, except those I reproduced.

## Files in this folder

- `sources_arc.md`: this file.
- `verify_arc_claims.py`, `verify_arc_claims_output.txt`: the checks of sections 5 and 8.
- `arc_raw/blog_algzoo.txt`, `arc_raw/blog_figures/`, `arc_raw/lesswrong_post_and_comments.txt`, `arc_raw/github_issue_1_and_commits.txt`.
- `arc_raw/pdf_notes/`: the three ARC notes (PDF and text).
- `arc_raw/chen_notebook_cells_and_outputs.txt`, `arc_raw/notebook_figures/`: all cells and the 15 output figures.
- `arc_raw/alg-zoo-repo/` (branch `main`), `arc_raw/alg-zoo-repo_mlp-branch.diff`.
- `arc_raw/gcs_logs/`: validation logs for $(16,10)$, $(4,3)$, $(2,2)$, all seeds, and one training log.
- `arc_raw/models/`: `one_layer_16_hidden_94_acc_2nd_argmax.pth` and the zoo's best $(2,2)$, $(4,3)$, $(16,10)$ checkpoints.
- `arc_raw/other_arc_posts/`: text of S9, S10 and the White-Box Challenge post.
