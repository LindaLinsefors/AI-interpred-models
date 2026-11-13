# Agent-written program for one rank-1 weight piece of the 2-layer model: layer 1 MLP output, piece 259
# (layers are numbered from 0, so layer 1 is the second layer).
# It predicts the piece's activation at each token from the text so far.
# Held-out variance explained: 0.94 (a per-token lookup table gets 0.38).
# The agent never saw the generator. The imported `names` (US birth-name counts) and
# `wordnet` modules came with the agent's sandbox and are not included. The program
# needs the team's harness to run; see the README for its interface.
#
# The agent's own account (it uses some internal terms: 'CI' is the piece's causal
# importance, i.e. whether the piece is needed at that token; 'the brief' is the agent's
# instructions; 'eval' and 'train rows' are held-out and fitting data):
#   Component 259 of layer 1's MLP down-projection is a "female protagonist, sentence end"
#   feature. It is high (about 0.9) on the full stop that ends a narrated sentence, and on the
#   closing quote after a sentence of dialogue, when the story's current character is female
#   (the next sentence is likely to start with "she" or a girl's name). It is near 0 for a male
#   character, near 0 for full stops inside quoted speech, and about 0.3 on the word "she"
#   itself. The program tracks the sex of the latest character cue (he/she/his/her/him,
#   girl/boy, or a common given name from the SSA counts in `names` that is at least 5000 births
#   and at least 10x one sex, minus five common words the list misreads) and uses quote-mark
#   parity to know when it is inside dialogue. On eval it gets variance explained 0.942, causal
#   variation explained 0.948, Pearson r 0.971. A simpler version that tracks only pronouns
#   scores about 0.60 on training rows, and dropping the quote tracking drops it to about 0.68,
#   so both parts matter. The NOT_NAMES list and the name thresholds are heuristics, and the
#   remaining spread within female-protagonist sentence ends (std about 0.19) is not modelled.
#   The values where CI is high are tightly clustered, so the within-CI-positive variance
#   explained is slightly negative (-0.17) even though the causal score is high.

"""Sentence ends in a story about a girl or woman (layer 1 MLP, component 259).

What it responds to: the full stop (or the closing quote after a sentence of
dialogue) that ends a narrated sentence, when the story's current character is
female. There the component is high (about 0.9), because the next sentence is
likely to start with "she" or a girl's name. For a male character the same
sentence ends are near 0. Full stops inside quoted speech are near 0 for either
sex. The word "she" itself has a weak, steady value of about 0.3. Before any
character cue has appeared, sentence ends are predicted as 0.

How the program knows the character's sex: it keeps the sex of the latest
character cue it has read. A cue is a he/she/his/her/him pronoun, the words
boy/girl, or a common given name that is clearly one sex (from the SSA birth
counts in `names`: at least 5000 births, and at least 10 times as many of one
sex as the other), except five common words the name list misreads as names
(NOT_NAMES). It also counts quote marks, so it knows when a sentence is inside
dialogue.
"""

import names

PREDICTS = ("h.1.mlp.down_proj:259",)
READS = ()

FEMALE_END = 0.9  # sentence end, character is female
MALE_END = 0.05  # sentence end, character is male
SHE_VALUE = 0.3  # the word "she" itself
PRONOUN_SEX = {"she": 1, "her": 1, "he": -1, "his": -1, "him": -1}
SEX_WORDS = {"girl": 1, "boy": -1}
MIN_BIRTHS = 5000
SKEW = 10
# Common words that the name list reads as a (male) name but that are not
# character names in these stories.
NOT_NAMES = {"forest", "stone", "chase", "rocky", "berry"}


def cue_sex(word):
    """+1 for a female cue, -1 for a male cue, 0 for no cue."""
    if word in PRONOUN_SEX:
        return PRONOUN_SEX[word]
    if word in SEX_WORDS:
        return SEX_WORDS[word]
    if not word.isalpha() or word in NOT_NAMES:
        return 0
    given = names.given_name(word.capitalize())
    if given is None or given.female + given.male < MIN_BIRTHS:
        return 0
    if given.female >= SKEW * given.male:
        return 1
    if given.male >= SKEW * given.female:
        return -1
    return 0


def step(carry, prefix, upstream):
    # carry: (sex of the latest cue: 1, -1 or 0; inside quotes; previous word)
    sex, in_quote, previous = carry if carry is not None else (0, False, "")
    word = prefix.token_strings[-1].strip().lower()
    value = 0.0
    if word == "." and not in_quote:
        value = FEMALE_END if sex == 1 else (MALE_END if sex == -1 else 0.0)
    elif word == '"' and in_quote and previous in (".", "!"):
        value = FEMALE_END if sex == 1 else (MALE_END if sex == -1 else 0.0)
    elif word == "she":
        value = SHE_VALUE
    if word == '"':
        in_quote = not in_quote
    cue = cue_sex(word)
    if cue:
        sex = cue
    return (sex, in_quote, word), {PREDICTS[0]: float(value)}
