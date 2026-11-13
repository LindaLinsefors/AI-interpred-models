# Agent-written program for one rank-1 weight piece of the 2-layer model: layer 1 MLP output, piece 755
# (layers are numbered from 0, so layer 1 is the second layer).
# It predicts the piece's activation at each token from the text so far.
# Held-out variance explained: 0.92 (a per-token lookup table gets 0.27).
# The agent never saw the generator. The imported `names` (US birth-name counts) and
# `wordnet` modules came with the agent's sandbox and are not included. The program
# needs the team's harness to run; see the README for its interface.
#
# The agent's own account (it uses some internal terms: 'CI' is the piece's causal
# importance, i.e. whether the piece is needed at that token; 'the brief' is the agent's
# instructions; 'eval' and 'train rows' are held-out and fitting data):
#   Component 755 of layer 1's MLP down projection is a "list-item noun" feature. In
#   enumerations such as "made a list of three things: 1 a hen, 2 a short dry witch, 3 a messy
#   crab", it fires on the noun naming each numbered item, which sits two to four tokens after
#   the number (a determiner and adjectives fall between). The program predicts 0.76 for such a
#   noun ending in -s (plural) and 0.62 otherwise, when: the noun is concrete (WordNet
#   categories animal, artifact, body, food, object, person or plant) and WordNet lists it
#   mainly as a noun (at least twice as many noun as adjective senses), so colour and size
#   adjectives are excluded; the number follows a comma or colon; and only plain alphabetic
#   words lie between number and noun, so a full stop ends the item. It predicts 0 everywhere
#   else, including nouns outside such lists and the adjectives inside them, which the component
#   leaves near zero. Measured on eval: variance explained 0.921, causal variation explained
#   0.965, Pearson r 0.960. What the simple version leaves out: a further ~0.02 of variance from
#   a 'list' header raising magnitude, a few nouns WordNet lists as adjectives (occasional
#   misses such as 'dress' and 'gift'), and the small activations on character names, which
#   carry little CI.

"""Component 755 of h.1.mlp.down_proj (layer 1 MLP): the noun that names a numbered list item.

When the text enumerates things, as in "made a list of three things: 1 a hen, 2 a short dry
witch, 3 a messy crab", this component switches on at the noun that names each item. That
noun sits two to four tokens after the item's number (a determiner and any adjectives come
in between), the number follows a comma or colon, and nothing but plain words lies between
the number and the noun, so a full stop ends the item. Only concrete nouns fire (animals,
objects, food, plants, people, body parts), and only when WordNet lists the word mainly as
a noun, so adjectives such as "small" or "red" stay off. Plural nouns fire a little harder
than singular ones. Everywhere else the program predicts 0: nouns outside such a list, and
the words in between, are ~0 for this component.

The program keeps no state: at each token it looks back at most six tokens.
"""

import functools

import wordnet

CONCRETE = {
    "noun.animal",
    "noun.artifact",
    "noun.body",
    "noun.food",
    "noun.object",
    "noun.person",
    "noun.plant",
}

SUBJECT = "h.1.mlp.down_proj:755"
PLURAL_VALUE = 0.76
SINGULAR_VALUE = 0.62

PREDICTS: tuple[str, ...] = ("h.1.mlp.down_proj:755",)
READS: tuple[str, ...] = ()


@functools.lru_cache(maxsize=None)
def is_item_noun(word: str) -> bool:
    """A concrete noun that WordNet lists mainly as a noun (at least twice as many noun senses
    as adjective senses)."""
    if not word.isalpha():
        return False
    senses = wordnet.senses(word)
    nouns = sum(".n." in s for s in senses)
    adjectives = sum(".a." in s or ".s." in s for s in senses)
    cats = set(wordnet.categories(word))
    return bool(cats & CONCRETE) and nouns >= 2 * adjectives


def step(carry, prefix, upstream):
    words = [s.strip() for s in prefix.token_strings]
    value = 0.0
    word = words[-1]
    if is_item_noun(word):
        for gap in (2, 3, 4):  # the list number sits `gap` tokens back
            if len(words) < gap + 2 or not words[-1 - gap].isdigit():
                continue
            # the number follows a comma or colon, and only plain words lie between
            # it and this token (a full stop ends the item)
            between = words[-gap:-1]
            if words[-2 - gap] in (",", ":") and all(w.isalpha() for w in between):
                value = PLURAL_VALUE if word.endswith("s") else SINGULAR_VALUE
                break
    return carry, {SUBJECT: value}
