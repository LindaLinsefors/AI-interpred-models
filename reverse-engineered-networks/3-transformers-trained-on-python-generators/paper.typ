#set document(title: "Recovering a known program from transformers trained on its output")
#set page(paper: "us-letter", margin: (x: 1in, y: 0.9in), numbering: "1")
#set text(font: "Libertinus Serif", size: 10.5pt)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.")
#show heading: set block(above: 1.3em, below: 0.7em)
#show heading.where(level: 1): set text(size: 12pt)
#show raw.where(block: true): set text(size: 7.6pt)
#show raw.where(block: true): block.with(fill: luma(246), inset: 6pt, radius: 3pt, width: 100%)
#show figure.caption: set text(size: 9pt)
#show figure.caption: set align(left)
#show link: set text(fill: rgb("#1a5fb4"))

#align(center)[
  #text(size: 16pt, weight: "bold")[Recovering a known program from transformers trained on its output]
  #v(0.4em)
  #text(size: 10pt)[The Goodfire parameter-decomposition team · experiments of October 2026]
  #v(0.2em)
  #text(size: 9pt, fill: luma(90))[Written for readers outside the team; assumes familiarity with transformers and KL divergence.]
]
#v(0.6em)

#block(inset: (x: 1.2em))[
#text(weight: "bold")[Abstract.]
Interpretability research tries to explain how neural networks compute their outputs. Claims that a network has been "reverse engineered" are hard to check, because for most trained networks nobody knows the true algorithm. If a network is trained on text written by a program we wrote, the target algorithm is known, and we can ask how much of it an interpretability pipeline recovers. Here we show that small transformers trained on a Python story generator contain rank-one weight pieces that mark the generator's decision points, and that LLM agents who never see the generator's code can write Python programs that reproduce the network's outputs closely. In a 2-layer model, single pieces found by parameter decomposition mark where the generator may copy a character's name or close a quote with F1 of 1.000. Swapped in for the network's parts, a merged program built on the decomposition reaches a KL of 0.044 nats per token from the network, and a program written without a decomposition 0.011; a 4-gram model scores 0.40. These programs are readable parsers of the grammar plus large fitted tables, and the tokenizer's class-ordered vocabulary let them read off the generator's word lists. Character-name identity is a dense code that no short rule describes, and on a harder generator with hidden variables the programs recover the visible world but not the hidden rules. Known-program models give a concrete test bed for interpretability, and they show that matching a network's outputs is not yet explaining it.
]

= Introduction

A common goal in mechanistic interpretability is to replace a trained network, or parts of it, with a description a person can read, such as a short program, and to show that the description computes what the network computes. Judging such work is hard because the ground truth is usually unknown: if a program matches a language model's outputs, it is unclear whether it captured the model's algorithm or only imitated its behavior.

Known-algorithm test beds address this. Tracr compiles programs in the RASP language directly into transformer weights @tracr @rasp, and Learning Transformer Programs trains transformers constrained to be readable as programs @ltp. We instead train an ordinary transformer on text from a program we wrote, so the network is free to implement the program however it likes, and then try to recover the program from the trained network.

We ask three questions: do the network's internal parts correspond to identifiable steps of the program; can programs written from the network, without the generator's code, reproduce it; and where do they fail? We study a simple generator (Sections 2–4) and a harder one with hidden variables (Section 5). Programs are written by LLM agents (Claude and GPT models) that see the trained network, its training text and, in one route, a decomposition of its weights.

= Setup

*The generator.* The simple generator is about 450 lines of Python (grammar plus word lists). It writes lowercase children's stories over a 1,358-word vocabulary. Each grammar rule is a coroutine that yields the exact distribution of the next word and receives the word drawn. Word classes never overlap within a distribution, so replaying any text recovers the generator's state and its exact next-word distribution. Its entropy is 2.26 nats per word. Besides a Zipf-weighted grammar with subject–verb agreement, the text has four features that need memory beyond a few words: character names introduced at a story's start and copied later; "he/she/his/her" bound to the most recently named character; quotes opened and closed by the same token; and numbered lists whose announced length decides when they end. An example:

#block(inset: (x: 1em))[#text(size: 8.5pt, font: "DejaVu Sans Mono")[once there was a girl named nadia and a boy named paul and a girl named ella . both paul and ella play and love nadia . she rests . … she made a list of four things : 1 a hammer , 2 the cat , 3 a fox , 4 the mice .]]

The tokenizer is word-level, so a token is a word or punctuation mark. It numbers the vocabulary class by class (function words, girl names, boy names, nouns, plural nouns, places, adjectives, verb forms, …), so a token's id reveals its word class. This matters for Section 4.

*The models.* We trained a 2-layer and a 4-layer Llama-style transformer @llama (width 192, MLP width 768, 6 attention heads) on 500M words of generator text. The 2-layer model has about 1.1–1.3 million weights. Both models learned the generator almost exactly: KL(generator || model) is 0.0017 nats per word for the 2-layer model and 0.0018 for the 4-layer one. Positions whose next word is a copied name, a pronoun, a list number or list-ending punctuation, or a closing quote are 15% of the text but carry 54% of a trigram model's KL from the network.

*Parameter decomposition.* Parameter decomposition @apd @spd @goodfire writes every weight matrix as a sum of many rank-one pieces ("components"). Each piece reads one direction of its input and writes one fixed direction. For each token, the method also learns which pieces are needed to reproduce the network's output there (their causal importance), and it is trained so that few are. For the 2-layer model, about 54 of 4,224 pieces are needed per token, and 816 pieces in the attention and MLP output matrices are needed somewhere. We call a piece _active_ at a token where it is needed; its _activation_ is the scalar it reads. Layers are numbered from 0.

*Scoring.* To score a set of programs, we replace the parts they cover (all 816 pieces, or whole layer outputs) with the programs' predictions at once, run the hybrid network, and report
$ "KL"(p_"network" "||" p_"hybrid") = "mean over positions of" sum_w p_"network" (w) log frac(p_"network" (w), p_"hybrid" (w)) $
in nats per token, over 2.09M held-out positions. Two references use the same formula with another model in place of the hybrid. A Kneser–Ney 4-gram model @kn trained on the same text scores 0.40. The true generator scores 0.010 after mixing it with a uniform distribution at weight $10^(-6)$, which is needed because it gives most words probability zero. (The 0.0017 above is the KL in the other direction, which needs no smoothing.) These are reference points, not floors: a program fitted to the network can match the generator's score without being closer to the generator.

= What the network's pieces mark

We replayed the generator on 261,632 held-out positions and labelled each with the choice the generator was facing. A _slot_ label marks positions where the generator gives an option nonzero probability, for instance "a character's name may come next". For each label we found the single piece whose being active best predicts it.

#figure(placement: auto, image("paper-src/slots.png", width: 100%),
  caption: [*Single pieces mark most of the generator's decision points, but not its persistent state.* F1 of the best single piece for each label. Left of the dashed line: slot labels ("this choice is open now"). Right: state variables that hold over many positions. In the 2-layer model, list counting is split across pieces.]) <slots>

@slots shows the result. In the 2-layer model, one piece (in layer 1's MLP input matrix) is active exactly where a character name may be copied (precision 1.000, recall 1.000; 15.4% of positions), and another exactly where an open quote may close (0.999 / 1.000). Separate pieces mark where "she/her" or "he/his" may come next (precision 1.00, recall 0.85); each is active only when the most recently named character has that gender, sometimes about 100 tokens back. List counting is split across pieces in the 2-layer model (best F1 0.67), while the 4-layer model has one piece for it (0.95).

These pieces encode that a choice is open, not which outcome follows. The quote-closing piece predicts a closing quote with precision 0.61, close to the generator's own 0.6 probability of closing at that point. Persistent states, such as being inside a quote or a list, have no single always-on piece (best F1 0.51–0.71).

*Programs for single pieces.* For each active piece, an agent (Claude Haiku 5.5) that saw the network, its training text and that piece's activations wrote a Python program predicting the activation token by token. Three examples follow; "fit" is the variance of the activation explained on held-out text.

- *Quotes.* A layer-1 attention-output piece has activation −0.29 at a sentence end inside a quote, the point where the generator chooses `mix((0.6, one('"')), (0.4, QUOTE_SUBJECT))`. Its program carries one bit, toggled by each quote mark (fit 0.97; @quote).
- *Pronouns.* A layer-1 MLP-output piece is high on a narrated sentence end when the current character is female, where the generator may write `PRONOUN[s.last.gender]`. Its program tracks the sex of the latest character cue (fit 0.94). This agent guessed names' genders from US birth records rather than the vocabulary order.
- *Lists.* A layer-1 MLP-output piece is active on each list item's noun, and its program finds those nouns (fit 0.92). What decides "," against "." is `k < s.list_length`. In an audit of ten per-piece programs, none tracked both the item number and the announced length.

#figure(
  grid(columns: (1fr, 1fr), gutter: 8pt,
```python
# Generator (stories.py)
def speech(s: State) -> Program:
    yield from say(', "')
    s.in_quote = True
    w = yield QUOTE_SUBJECT
    while w != '"':
        ...   # one quoted sentence,
              # ending in "." or "!"
        w = yield mix((0.6, one('"')),
                      (0.4, QUOTE_SUBJECT))
    s.in_quote = False
```,
```python
# Agent's program, layer-1 attention piece 59
def step(carry, prefix, upstream):
    inside_quote = bool(carry)
    token = prefix.token_strings[-1].strip()
    value = 0.0
    if token == '"':
        inside_quote = not inside_quote
    elif token in {".", "!"} and inside_quote:
        value = -0.29
    return inside_quote, {SUBJECT: value}
```),
  caption: [*A piece's program mirrors the generator's state.* Left: the generator's quote rule. Right: the program an agent wrote, without seeing the generator, for the piece that is active where the generator decides whether to close a quote.]) <quote>

= One program for the whole model

#figure(placement: auto, image("paper-src/scoreboard.png", width: 80%),
  caption: [*Programs written without the generator reproduce the network closely.* Joint KL from the 2-layer network for each program set (all parts it covers replaced at once), n-gram baselines (which replace the whole network) and the smoothed true generator. Log scale.]) <scoreboard>

@scoreboard collects the joint scores.

*One program per piece* (816 programs) reaches 0.317, already better than every n-gram model. Its error concentrates in two places.

_List endings._ The positions where the count decides the next token (a list number, or the "," or "." after an item) are 6% of the text. Five of the ten audited agents noticed that their piece changed with list position; none kept it, and at least four said it barely improved their fit. With all programs in place these positions cost 0.88 nats each, mostly through two programs (pieces 658 and 442, fit 0.98 each). Adding one feature, "this is the last item", to those two cut the KL there from 0.88 to 0.12 and the joint KL from 0.317 to 0.273, while their fit changed by 0.001. A fit averaged over all tokens cannot see a signal that the output depends on at a few positions.

_Names._ At name positions, layer-1 attention writes each of the 195 names as its own dense vector. The centred per-name vectors show no clusters: 59 directions hold 99% of their variance, and none holds more than 4%. About 106 pieces carry parts of this name code; at the layer-1 attention output, about 65 of them each respond to roughly 30 names. The agents' instructions discouraged lookup tables. In 12 of 13 audited sessions the agent fitted a table over names and then dropped it, and the median of a related fit score on these pieces was 0.07. Allowing tables over words raised it to 0.85.

*Merged programs.* A team of two stronger agents (an OpenAI GPT-6 model and Claude Opus 5.5) merged the per-piece programs into one, so that shared state such as a list counter is computed once. With tables banned, the merged program got list endings right (KL 0.000) and ordinary tokens nearly so (0.002) but failed at name positions, for a joint 0.837. In a second attempt with tables over words allowed, the team kept the earlier per-piece list programs instead of its list counter and reached *0.044*. That program is about 8.1k lines; 591 of its 816 outputs reproduce earlier per-piece programs and 225 come from refitted tables of story characters and roles. Its remaining error splits evenly between list endings and name positions.

*A program without a decomposition.* The same kind of agent team, given the network and its text but no decomposition, wrote one program for the four layer outputs directly, in about an hour (effort was not equalised with the decomposition route). It reached *0.011* (name positions 0.074, list counts 0.000). It consists of a 230-line streaming parser of the story grammar (word class, grammatical role, agreement, quotes, list counter, cast of characters, recency) and about 16M fitted coefficients (82 MB) that map the parser's 20.8k named features linearly onto the layer outputs: more than ten times as many numbers as the network has weights. Its word lists match the generator's lexicon word for word and in the same order, consistent with reading them off the class-ordered vocabulary. Its parser lists exactly the generator's five name positions (@names). Which name is copied, and how each name is written, lives in the fitted vectors.

#figure(
  grid(columns: (1fr, 1fr), gutter: 8pt,
```python
# Generator: where a name may be written
# subject_start (after "." or closing quote):
    (0.35, s.names())
# object_phrase (after a transitive verb,
#                outside quotes):
    w = yield mix((0.6, determiners), *owner,
                  (0.15, s.names()))
# subject_rest, "both mia and leo":
    first = yield [(1.0, s.names())]
    yield from say("and")
    yield [(1.0, s.names(exclude=first))]
```,
```python
# Recovered program (excerpt, simplified)
if word in ('.', '!') and not q:
    slot = 'sentence_end'
elif word == '"' and quote:
    slot = 'closing_quote'
elif c in ('vt', 'vt_s') and not q:
    slot = 'object_name'
elif word == 'both':
    slot = 'first_joint_name'
elif word == 'and' and joint_names == 1:
    slot = 'second_joint_name'
# then: one fitted vector per eligible name
```),
  caption: [*The program written without a decomposition recovered the generator's name positions, but not how names are encoded.* `q` and `quote` mean "inside a quote" after and before the current token; `vt`, `vt_s` are transitive verbs.]) <names>

= A harder generator with hidden variables

The second generator (about 750 lines) writes stories in a small visible world: characters move between places, take, drop and give things, find and lose coins, talk, list what they see, and answer questions about the story. It also has four families of hidden variables that the text never states: a theme per story (16 options); a temperament per character (4); a kind per noun (12) that restricts which adjectives and verbs fit; and 8 "laws", XOR rules over hidden halves of the place and name sets. Ignoring the hidden variables costs 0.154 nats per word. Of the sizes we trained, a 6-layer, width-384 transformer learned this world best (KL 0.012 from the generator); 4- and 6-layer models of width 256 learned the laws less well.

#figure(
  table(columns: (auto, auto, auto), align: (left, right, right), stroke: none,
    table.hline(),
    [*Program for all 12 layer outputs*], [*joint KL*], [*size*],
    table.hline(stroke: 0.5pt),
    [No decomposition, best accuracy (fourth refining session)], [*0.043*], [1.51 MB],
    [No decomposition, best small version], [0.061], [0.74 MB],
    [Decomposition, one program per active piece (5,123), not merged], [0.633], [1.95 MB],
    [Decomposition, pieces' true activations (best case)], [0.003], [–],
    [Baseline: each layer output from the current token alone], [4.36], [–],
    [_True generator source, for comparison_], [–], [14.8 KB],
    table.hline(),
  ),
  caption: [*On the harder generator, a program written without a decomposition is accurate but large.* Sizes are gzipped source with comments stripped. The decomposition route was not matched in effort: it had one inexpensive agent per piece and no merging round (one was started and stopped early).]) <v2>

@v2 shows the scores. The best program's readable part recovers the visible world (cast, genders, places, possessions, coins, lists, questions and answers, pronoun referents) and infers that each story belongs to one of 16 themes from word-to-theme affinities. It does not recover the hidden families as rules: temperament is approximated by features of each character's event history, noun kinds by a per-noun lookup, and the laws have no explicit representation. In an earlier version, 94% of the source was packed numeric tables (@bytes).

#figure(placement: auto, image("paper-src/v2-bytes.png", width: 70%),
  caption: [*The accurate program for the harder generator is mostly fitted numbers.* Source composition of an earlier no-decomposition version (joint KL 0.054, 4.85 MB before compression).]) <bytes>

= Limits

*Fitting outputs is not explaining the computation.* Every program here reads only the text, never the network's internal values. When one of the network's pieces is edited (held at its mean, or given its value from another text), the hybrid receives the edited piece, but the downstream programs cannot respond. For the 12 most influential pieces, the hybrid's KL from the edited network was 0.36 (mean) and 0.50 (swap), about the same as ignoring the edit entirely (0.41 and 0.60). This is expected by construction, and it shows that KL on ordinary text cannot distinguish a program that tracks the network's computation from one that only predicts its text.

*The programs are partly tables.* In every accurate program, the readable part is a parser of the grammar, and the map from parsed facts to the network's numbers is fitted. On the harder generator, the hidden rules end up inside fitted numbers.

*The vocabulary gave away the word classes.* Because token ids are grouped by class, a program can recover the generator's word lists without inferring them. How much of the grammar the agents found independently is therefore unclear; the parsing logic and the name positions are their own.

*Scope.* These are small models on two synthetic languages, with one or a few attempts per setting and unequal effort between routes.

= Conclusion

Training networks on a known program makes "has this network been reverse engineered?" a measurable question. On a simple generator, single rank-one pieces line up with the program's decision points, and programs written without the generator's code reproduce the network's outputs closely. The remaining error is where the network's representation is dense (character names) or where an averaged fit score hid a rare but decisive signal (list endings). On a harder generator, readable code recovered the visible world and fitted tables absorbed the hidden rules. Two immediate next steps are a tokenizer that does not reveal word classes, and scoring programs on how the network responds to edits of its parts, not only on ordinary text.

#v(0.6em)
#text(size: 8.5pt)[*Materials.* The generators, the agent-written programs quoted here and the readable parts of the whole-model programs are in the `code/` folder beside this paper. `explainer.html` gives the same story with side-by-side code.]

#set text(size: 9pt)
#bibliography("refs.yml", title: "References", style: "ieee")
