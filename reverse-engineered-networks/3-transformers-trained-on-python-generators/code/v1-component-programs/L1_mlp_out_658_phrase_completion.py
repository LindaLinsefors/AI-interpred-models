# Agent-written program for one rank-1 weight piece of the 2-layer model: layer 1 MLP output, piece 658
# (layers are numbered from 0, so layer 1 is the second layer).
# It predicts the piece's activation at each token from the text so far.
# Held-out variance explained: 0.99 (a per-token lookup table gets 0.55).
# The agent never saw the generator. The imported `names` (US birth-name counts) and
# `wordnet` modules came with the agent's sandbox and are not included. The program
# needs the team's harness to run; see the README for its interface.
#
# The agent's own account (it uses some internal terms: 'CI' is the piece's causal
# importance, i.e. whether the piece is needed at that token; 'the brief' is the agent's
# instructions; 'eval' and 'train rows' are held-out and fitting data):
#   The component fires on the word that completes a phrase after a verb, a preposition, or
#   "is/are/am/was": the noun, name, adjective or adverb in "sees the dog", "in the park",
#   "named tara", "is happy", "runs quickly". The completing word fires, not the determiner or
#   an attributive adjective before it ("the big dog": "dog" fires). Strength depends on what it
#   completes: adverbs about 4.4, an object noun phrase after a verb about 3.9, a place after a
#   preposition about 3.2, a bare name or predicate after a verb or copula about 3.4, and a name
#   right after sentence-initial "both" about 0.9. Subject phrases (after a full stop, comma,
#   "and", a number or a pronoun), verbs, function words, girl/boy and attributive adjectives
#   stay at 0. The program reads only earlier words, using short word lists (verbs in base and
#   -s forms, determiners, copulas, prepositions) and WordNet to tell adjectives from nouns. On
#   eval it scores variance explained 0.986, causal variation explained 0.987, Pearson r 0.993,
#   and 0.874 on CI-positive positions. Not modelled: numbered list items (small activations
#   rising with the item number) and some nouns WordNet also tags as adjectives (e.g. "beetle",
#   "dress"), which are sometimes missed.

"""Object and predicate words: the component fires on the word that completes a phrase after
a verb, a preposition, or "is/are/am/was": the noun, name, adjective or adverb in
"sees the dog", "in the park", "named tara", "is happy", "runs quickly".

The word that completes the phrase fires, not the determiner or attributive adjective
before it ("the big dog": "dog" fires, "the" and "big" do not). How strongly it fires
depends on what it completes: manner and time adverbs about 4.4 ("runs quickly"), an
object noun phrase after a verb about 3.9 ("sees the dog"), a place after a preposition
about 3.2 ("in the park"), and a bare name or predicate after a verb or copula about 3.4
("named tara", "is happy"). A phrase that opens a sentence, a clause or a list item stays
near 0 ("The dog runs", "and the cat", "2 a dog"), and so do verbs, function words and
attributive adjectives. Girl and boy never fire here. After a sentence-initial "both",
the first name is weakly active (0.9).

The rule reads only the words before the current one, so it is causal.
"""

import functools

import wordnet

PREDICTS = ("h.1.mlp.down_proj:658",)
READS = ()

ADVERB_LEVEL = 4.4  # "runs quickly", "sleeps again"
OBJECT_LEVEL = 3.9  # "sees the dog"
PLACE_LEVEL = 3.2  # "in the park"
PREDICATE_LEVEL = 3.4  # "named tara", "is happy"
BOTH_LEVEL = 0.9  # "both ana and ben ..." at the start of a sentence

DETERMINERS = {"the", "a", "some", "many", "both", "his", "her", "their", "my", "your", "our"}
COPULAS = {"is", "are", "am", "was"}
PREPOSITIONS = {"in", "at", "near", "behind", "under", "beside", "inside", "across"}
OTHER_FUNCTION_WORDS = {
    "and", "he", "she", "they", "i", "you", "we", "of", "there", "once", "list", "things"}
ADVERBS = {"today", "again", "always", "often", "away", "together", "alone", "everywhere",
           "outside"}  # adverbs that do not end in -ly
NEVER_FIRE = {"girl", "boy"}  # nouns that stay silent in these stories
VERBS = set(
    """run jump walk sleep sing dance laugh cry smile play swim sit stand wait rest shout
    hide fly fall hop skip spin wave nod yawn sneeze giggle wander rush hurry stop grow
    shine glow float sink roll slide crawl march skate bounce shiver blink frown sigh cheer
    clap dream wake travel return listen think bow kneel wobble tumble stumble dash race
    leap paddle drift howl bark purr chirp buzz roar quack squeak snore scream grin pout
    wiggle sway rise see like love find hold take want need help hug kick push pull carry
    throw catch eat drink make build paint draw read write open close wash fix break chase
    follow meet visit watch touch keep lose bring buy sell cook bake cut fill drop lift
    share give feed pet tickle kiss call greet thank miss remember forget hear smell taste
    grab squeeze shake borrow collect count fold hang pack pick plant pour protect save
    scare search steal tie trade trust wrap admire ask bump chew comb cover crush dig
    discover enjoy explore guard hit hunt inspect join lick mend mix notice obey pinch poke
    polish praise rescue rub scrub sniff stack stir stroke sweep teach tease toss warn weigh
    named say made""".split()
)


def is_verb(w):
    """A listed verb in its base, -s, -es or -ies form."""
    forms = [w]
    if w.endswith("s"):
        forms.append(w[:-1])
    if w.endswith("es"):
        forms.append(w[:-2])
    if w.endswith("ies"):
        forms.append(w[:-3] + "y")
    return any(form in VERBS for form in forms)


def is_opener(w):
    """A word after which an object or a predicate begins: a verb, a preposition or a copula."""
    return is_verb(w) or w in PREPOSITIONS or w in COPULAS


@functools.lru_cache(maxsize=None)
def is_adjective(w):
    """WordNet lists w as an adjective about as often as a noun ("chicken" is mainly a noun)."""
    senses = [sense.split(".")[1] for sense in wordnet.senses(w)]
    adjectives = senses.count("a") + senses.count("s")
    return adjectives > 0 and adjectives >= 0.5 * senses.count("n")


def is_content(w):
    """A word that can fire: an alphabetic word that is not a function word or a verb."""
    function = DETERMINERS | COPULAS | PREPOSITIONS | OTHER_FUNCTION_WORDS
    return w.isalpha() and w not in function and w not in NEVER_FIRE and not is_verb(w)


def strength(w, anchor, has_determiner):
    """How strongly a completing word fires, given the word it completes (its anchor)."""
    if w.endswith("ly") or w in ADVERBS:
        return ADVERB_LEVEL
    if anchor in PREPOSITIONS:
        return PLACE_LEVEL
    return OBJECT_LEVEL if has_determiner else PREDICATE_LEVEL


def fires(w, history):
    """Value of word w, given the words before it (history, nearest last, stripped)."""
    if not is_content(w):
        return 0.0
    p1 = history[-1] if history else ""
    p2 = history[-2] if len(history) >= 2 else ""
    if p1 in DETERMINERS:
        # "the big dog" is attributive, but "in the forest" is a place: fire after a preposition
        if is_adjective(w) and p2 not in PREPOSITIONS:
            return 0.0
        if is_opener(p2):
            return strength(w, p2, True)
        return BOTH_LEVEL if p1 == "both" and not is_adjective(w) else 0.0
    if is_adjective(w):
        # predicate adjectives follow "is/are/am/was"; after a verb, a word that WordNet also
        # lists as an adjective is a name or a noun
        return strength(w, p1, False) if (p1 in COPULAS or is_verb(p1)) else 0.0
    if is_adjective(p1) and not is_verb(p1):
        # a noun after attributive adjectives: look back past them to a determiner, then its anchor
        j = len(history) - 1
        while j >= 0 and is_adjective(history[j]):
            j -= 1
        if j >= 0 and history[j] in DETERMINERS:
            j -= 1
        anchor = history[j] if j >= 0 else ""
        return strength(w, anchor, True) if is_opener(anchor) else 0.0
    return strength(w, p1, False) if is_opener(p1) else 0.0


def step(carry, prefix, upstream):
    history = [] if carry is None else carry
    word = prefix.token_strings[-1].strip().lower()
    value = fires(word, history)
    return (history + [word])[-8:], {PREDICTS[0]: value}
