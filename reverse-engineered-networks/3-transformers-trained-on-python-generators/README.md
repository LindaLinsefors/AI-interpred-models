# Transformers trained on a known Python program

From the Goodfire parameter-decomposition team (experiments of October 2026).

Start with `explainer.html`, which opens offline in any browser. `paper.pdf` covers the same results as a short paper with references. Its source is `paper.typ`, with `refs.yml` and the figures in `paper-src/`.

**Setup.** We wrote Python programs that generate simple stories, trained small transformers on their text, and then tried to recover the programs from the trained networks. The programs in `code/` were written by LLM agents. The agents saw the trained network and its training text, but never the generator's code.

One caveat applies throughout: the tokenizer numbers the vocabulary class by class (see `code/generator-v1/corpus.py`), so token ids reveal word classes.

**Scores.** "Joint KL" is KL(network || hybrid), in nats per token, averaged over held-out positions. The hybrid is the network with the program's predictions in place of the parts the program covers. Lower is better. On the simple generator, a 4-gram language model scores 0.40. The true generator, mixed with a uniform distribution at weight 1e-6, scores 0.010. These are reference points, not floors.

## code/

| Path | What it is | Measured score |
|---|---|---|
| `generator-v1/stories.py`, `lexicon.py`, `corpus.py` | The simple generator, kept verbatim: grammar (each rule a coroutine yielding the exact next-word distribution), word lists, and the tokenizer and dataset code. `corpus.py` imports our internal training packages and will not run elsewhere. `stories.py` and `lexicon.py` run with plain Python. | 2- and 4-layer models trained on it reach KL(generator \|\| model) 0.0017 / 0.0018 |
| `v1-component-programs/*.py` | Seven agent-written programs. Each predicts the activation of one rank-1 weight piece of the 2-layer model from the text so far. Each file starts with its held-out score and the agent's own description. Files are named by layer (counted from 0), matrix, piece number and what the piece does. | Variance explained: quote piece 59: 0.97; she piece 259: 0.94; he piece 318: 0.94; list-noun piece 755: 0.92; pieces 658 and 442 (which carry most list-ending error): 0.98 each; name piece 189: 0.00 (the agent found no rule) |
| `v1-whole-model-program/no_decomposition_program_readable_part.py` | The readable part of the best program for the 2-layer model: a ~230-line streaming parser of the story grammar, written without a decomposition. The omitted remainder of the 82 MB file is ~16M fitted coefficients that map the parser's features onto the four layer outputs. | Joint KL 0.011 for the full program |
| `generator-v2/stories.py` | The harder generator: a visible world plus hidden variables (theme, temperament, noun kinds, laws). Its word-list file is not included; the full source is viewable in `extras/v2-all-program-versions.html`. | A 6-layer, width-384 model reaches KL 0.012 from it |
| `v2-whole-model-program/v2_program_tables_elided.py` | An accurate program for that 6-layer model, written without a decomposition. Its packed numeric tables (94% of its 4.85 MB) and long word lists are elided. | Joint KL 0.054 on held-out development text. The best later version scores 0.043 at 1.51 MB gzipped. |

**The program interface.** Every program defines `step(carry, prefix, upstream) -> (carry, outputs)`. The team's harness calls it once per token, in order:
- `carry` is the program's own state from the previous token (`None` at the start).
- `prefix.token_strings` is the list of tokens so far. Some programs strip a leading space from tokens and some do not, so `'"'` and `' "'` both occur.
- `upstream` would hold network values the program declared in `READS`. It is empty here: every program reads only text.
- `outputs` maps each name in `PREDICTS` to a predicted value.

The files parse, but none run without that harness and the trained network. The decomposition and the network weights are not included. Neither is the 0.044 program built on the decomposition, which exists only in internal storage; the paper and explainer describe it.

## extras/

These are original pages written by the team for colleagues, unedited. All five render offline. Some links in them point to internal pages that will not open.

| File | What it is |
|---|---|
| `v1-generator-source.html` | The simple generator, with an overview, a 260-word sample and its full source. |
| `v1-name-code-geometry.html` | An interactive explainer of why each name piece responds to many character names. Its piece counts are for a second decomposition of the same model. |
| `v1-list-last-item-problem.html` | How a missing "last item" feature broke list endings, even though the programs fit their pieces well. |
| `v2-program-beside-generator.html` | An accurate program for the harder generator (tables elided), side by side with the true generator. |
| `v2-all-program-versions.html` | Every version of the harder-generator program from one merging round, each with its own account, plus a detailed comparison of the best version with the generator. |

### Glossary of internal terms in the extras

| Term | Meaning |
|---|---|
| VPD | The team's parameter-decomposition method; "decomposition" for short. |
| component, subcomponent | One rank-1 piece of a weight matrix. |
| CI, causal importance | Whether a piece is needed at a token. "Active" means CI > 0. |
| VE | Variance explained: the share of a piece's activation variance a program explains. |
| CVE | Causal variation explained: a related fit score used while writing programs. |
| Explain, explain2, explain programs | The team's pipeline in which agents write Python programs for pieces or layers. |
| brief | The instructions given to the agents. |
| no-VPD, target-only | Agents saw only the trained network, with no decomposition. |
| Haiku, Opus, Sol, Astra | Agent models: Claude Haiku 5.5 and Claude Opus 5.5 (Anthropic); Sol and Astra are internal names for GPT-6 models (OpenAI). |
| h.0, h.1 | Layers 0 and 1 (counted from 0). |
| o_proj, down_proj, c_fc | Attention output matrix, MLP output matrix, MLP input matrix. |
| L0 | The number of pieces active per token. |
| PGD | A robustness check of the decomposition under adversarially chosen masks. |
| e2e_recon | An end-to-end reconstruction score that agents did not see during these sessions. |
| p-xxxxxxxx, t-xxxxxxxx | Internal IDs for decomposition runs and trained-model runs. |
| task:NNNN, lore:…, agent:xxxx, human:name | Internal task threads, internal notes, AI agents on the team, and team members. |
