# Agent-written program for one rank-1 weight piece of the 2-layer model: layer 1 attention output, piece 59
# (layers are numbered from 0, so layer 1 is the second layer).
# It predicts the piece's activation at each token from the text so far.
# Held-out variance explained: 0.97 (a per-token lookup table gets 0.33).
# The agent never saw the generator. The imported `names` (US birth-name counts) and
# `wordnet` modules came with the agent's sandbox and are not included. The program
# needs the team's harness to run; see the README for its interface.
#
# The agent's own account (it uses some internal terms: 'CI' is the piece's causal
# importance, i.e. whether the piece is needed at that token; 'the brief' is the agent's
# instructions; 'eval' and 'train rows' are held-out and fitting data):
#   The program tracks one bit as it reads the row: whether a double quotation mark is currently
#   open (each `"` token toggles it). At a sentence-ending `.` or `!` read while a quote is
#   open, it predicts about -0.29, the typical value of this component there; at every other
#   position it predicts 0. The component is a sentence-end-inside-quotes detector: on 1024
#   training rows, every `.`/`!` inside quotes has CI about 0.99 and value -0.288 (std 0.007),
#   while periods outside quotes have value about 0 and CI about 0. Eval: variance explained
#   0.971, causal variation explained 0.998, Pearson r 0.985, variance explained where CI > 0 is
#   0.943. What the simple version leaves out is the diffuse, low-CI activity (about ±0.01 on
#   ordinary periods and commas, and small values on content words), which makes up about 3% of
#   variance and which I did not model.

"""Component 59 of layer 1's attention output projection: sentence-end punctuation inside quotes.

What it does: the program walks the row token by token and keeps one bit, whether a
double quotation mark is currently open (each `"` token toggles the bit). When a
sentence-ending mark (`.` or `!`) comes while the bit is set, i.e. at the end of a
sentence spoken inside quotes, the component is strongly negative (about -0.29).
Everywhere else it predicts 0: ordinary periods outside quotes, commas, and content
words all sit near zero for this component.

Why: in the corpus, quoted speech is a run of short sentences that close with `.` or `!`
before the closing quote. The component's large negative output is written at exactly
those positions, and its causal importance is about 1 there and about 0 everywhere else.
"""

from collections.abc import Mapping

PREDICTS: tuple[str, ...] = ("h.1.attn.o_proj:59",)
READS: tuple[str, ...] = ()

QUOTE_MARK = '"'
SENTENCE_END = frozenset({".", "!"})
INSIDE_QUOTE_VALUE = -0.29


def step(carry, prefix, upstream: Mapping):
    inside_quote = bool(carry)
    token = prefix.token_strings[-1].strip()
    value = 0.0
    if token == QUOTE_MARK:
        inside_quote = not inside_quote
    elif token in SENTENCE_END and inside_quote:
        value = INSIDE_QUOTE_VALUE
    return inside_quote, {"h.1.attn.o_proj:59": value}
