# Agent-written program for one rank-1 weight piece of the 2-layer model: layer 1 attention output, piece 189
# (layers are numbered from 0, so layer 1 is the second layer).
# It predicts the piece's activation at each token from the text so far.
# Held-out variance explained: -0.00 (a per-token lookup table gets 0.02).
# The agent never saw the generator. The imported `names` (US birth-name counts) and
# `wordnet` modules came with the agent's sandbox and are not included. The program
# needs the team's harness to run; see the README for its interface.
#
# The agent's own account (it uses some internal terms: 'CI' is the piece's causal
# importance, i.e. whether the piece is needed at that token; 'the brief' is the agent's
# instructions; 'eval' and 'train rows' are held-out and fitting data):
#   This program predicts 0 at every position for component 189 of layer 1's attention output
#   projection; it carries no state. What the component does: on about half of periods, closing
#   quotes and some transitive verbs (where its CI is near 1), heads 4 and 5 of layer 1 attend
#   to the story's protagonist name token and copy that token's value into the component. Its
#   sign and size there are set by which name it is (for example "julia" gives about -0.19 and
#   "mabel" about +0.12 at the same kind of period), so the component's activation keeps one
#   sign within a story. I could not find a short rule for that per-name sign: name length,
#   final letter, initial vowel, popularity, gender, word frequency and the word before "named"
#   each correlate at most 0.15 with the per-name mean, and a per-name lookup would explain at
#   most about 12% of the variance at those positions, while the brief rules out such tables.
#   Predicting 0 gives eval variance explained about -0.005 and causal variation explained about
#   -0.01, which is essentially no fit. What the simple version misses is the name-keyed sign; a
#   constant at periods was also no better (variance explained about -0.003 offline), so 0 is
#   the simplest honest prediction.

"""Component 189 of h.1.attn.o_proj (layer 1 attention output projection).

What it does in the model: at sentence ends (periods), closing quotes and some
transitive verbs, heads 4 and 5 of layer 1 attend to the story's protagonist name
token and copy its value into this component. The component's sign there is set by
which name it is (for example "julia" gives about -0.19, "mabel" about +0.12), and
that sign is not a property I could describe with a short rule: name length,
final letter, initial vowel, popularity, gender and word frequency each correlate
at most 0.15 with the per-name mean.

What this program predicts: zero everywhere. Its variance explained is about zero;
a constant at periods gives no gain either, and only knowing the name's sign would.
"""

PREDICTS = ("h.1.attn.o_proj:189",)
READS = ()


def step(carry, prefix, upstream):
    """Predict 0.0 for the component at every position (no state is carried)."""
    return carry, {PREDICTS[0]: 0.0}


if __name__ == "__main__":
    print("This program predicts 0 everywhere; run it through eval or submit.")
