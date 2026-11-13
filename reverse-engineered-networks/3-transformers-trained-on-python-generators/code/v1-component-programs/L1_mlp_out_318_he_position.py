# Agent-written program for one rank-1 weight piece of the 2-layer model: layer 1 MLP output, piece 318
# (layers are numbered from 0, so layer 1 is the second layer).
# It predicts the piece's activation at each token from the text so far.
# Held-out variance explained: 0.94 (a per-token lookup table gets 0.37).
# The agent never saw the generator. The imported `names` (US birth-name counts) and
# `wordnet` modules came with the agent's sandbox and are not included. The program
# needs the team's harness to run; see the README for its interface.
#
# The agent's own account (it uses some internal terms: 'CI' is the piece's causal
# importance, i.e. whether the piece is needed at that token; 'the brief' is the agent's
# instructions; 'eval' and 'train rows' are held-out and fitting data):
#   Component 318 of layer 1's MLP down projection fires at the end of a narrative sentence when
#   the story's current character is male, and its value is near 0 otherwise. The program reads
#   the text so far. It counts quote marks to know whether it is inside speech. At a full stop
#   outside speech, or at the closing quote that ends speech, it predicts 0.75 if the most
#   recent gendered reference is male (he/him/his, boy, or a first name the names library gives
#   as at least 97% male, with 5000+ counts), 0 if it is female (she/her, girl, or a female
#   first name), and 0 everywhere else. On the eval rows this gives variance explained 0.942,
#   causal variation explained 0.955 and Pearson r 0.971. It captures the on/off pattern at
#   sentence ends and the male/female split: male-context ends average about 0.75 and female-
#   context ends about 0.01. In the model's terms it is a "the next sentence will begin with he"
#   signal: on training data its value at these positions is about 0.77 when the next token is
#   "he" and about 0 when it is "she". What it misses: the spread within male-context ends (SD
#   about 0.15, not explained by the preceding word or by distance to the last cue), and small
#   values elsewhere. I left out role nouns (man, king, woman, queen) and family nouns, which
#   are weaker cues in these stories and added no fit. A few common words that the first-name
#   list also contains (meadow, star, candy, and similar) are excluded by hand.

"""Component 318 of h.1.mlp.down_proj (layer 1 MLP down projection).

What it does: it fires at the end of a narrative sentence when the story's
current character is male. A narrative sentence end is a full stop outside
quoted speech, or the closing quote that ends a piece of speech. There it is
about 0.75 when the latest gendered reference is male (he/him/his, boy, or a
male first name) and about 0 when it is female (she/her, girl, or a female
first name). Everywhere else it is 0. In effect it signals "the next sentence
will probably begin with 'he'": at these positions its value tracks whether
the next token is 'he' (about 0.77) or 'she' (about 0).

The program reads only the text so far. It counts quote marks to know whether
it is inside speech, and it keeps the gender of the most recent gendered
reference.
"""

from collections.abc import Mapping

import names

PREDICTS: tuple[str, ...] = ("h.1.mlp.down_proj:318",)
READS: tuple[str, ...] = ()

MALE_WORDS = frozenset("he him his himself boy boys".split())
FEMALE_WORDS = frozenset("she her hers herself girl girls".split())
# common English words that the first-name list also contains; not used as names
NOT_NAMES = frozenset(
    "meadow candy star cherry merry treasure chase rocky stone".split()
)
MALE_LEVEL = 0.75  # the typical value at a male-context sentence end
NAME_MIN_COUNT = 5000  # first names common enough to trust
NAME_SHARE = 0.97  # a first name at least this share one sex


def gender_cue(token: str) -> int:
    """+1 for a male reference, -1 for a female one, 0 for anything else."""
    if not token.startswith(" "):
        return 0
    word = token[1:].lower()
    if not word.isalpha():
        return 0
    if word in MALE_WORDS:
        return 1
    if word in FEMALE_WORDS:
        return -1
    if word in NOT_NAMES:
        return 0
    record = names.given_name(word.capitalize())
    if record is None:
        return 0
    total = record.male + record.female
    if total < NAME_MIN_COUNT:
        return 0
    if record.male >= NAME_SHARE * total:
        return 1
    if record.female >= NAME_SHARE * total:
        return -1
    return 0


def step(carry, prefix, upstream: Mapping):
    # carry: (gender of the latest gendered reference, whether inside speech)
    gender, in_speech = carry if carry is not None else (0, False)
    token = prefix.token_strings[-1]
    value = 0.0
    # a full stop in narration, or a closing quote, ends a narrative sentence
    if (token == "." and not in_speech) or (token == ' "' and in_speech):
        value = MALE_LEVEL if gender > 0 else 0.0
    if token == ' "':
        in_speech = not in_speech
    cue = gender_cue(token)
    if cue:
        gender = cue
    return (gender, in_speech), {PREDICTS[0]: value}
