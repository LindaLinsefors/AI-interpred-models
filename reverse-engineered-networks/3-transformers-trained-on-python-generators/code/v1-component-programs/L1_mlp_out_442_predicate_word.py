# Agent-written program for one rank-1 weight piece of the 2-layer model: layer 1 MLP output, piece 442
# (layers are numbered from 0, so layer 1 is the second layer).
# It predicts the piece's activation at each token from the text so far.
# Held-out variance explained: 0.98 (a per-token lookup table gets 0.65).
# The agent never saw the generator. The imported `names` (US birth-name counts) and
# `wordnet` modules came with the agent's sandbox and are not included. The program
# needs the team's harness to run; see the README for its interface.
#
# The agent's own account (it uses some internal terms: 'CI' is the piece's causal
# importance, i.e. whether the piece is needed at that token; 'the brief' is the agent's
# instructions; 'eval' and 'train rows' are held-out and fitting data):
#   The component is a "predicate word in narration" feature. The program walks each row token
#   by token, tracking whether it is inside quotation marks, whether a verb has come up in the
#   current clause, whether an object slot is open after a transitive verb, and whether the
#   previous word was a copula. Outside quotes it predicts about 4.8 on intransitive verbs (base
#   and third-person forms, from a hand list; third-person forms are derived by spelling rules),
#   about 4.0 on -ly or listed manner/time adverbs after such a verb, about 2.9 on place nouns
#   (from a list), about 3.2 on the first non-function word after is/are/am/was/were, about 3.2
#   on the object noun after a transitive verb (adjectives in between keep the slot open, so
#   "sees the big dog" fires on "dog"), and about 2.9 on the name after "named". It predicts 0
#   for subjects, determiners, attributive adjectives, function words, transitive verbs
#   themselves, and everything inside quotation marks. Function-word and adjective tests use
#   closed_class and WordNet. Why: the data show a clear split. Intransitive verbs, adverbs and
#   clause-final objects or predicates fire at CI near 1, and subjects and quoted speech stay at
#   0. Inside quotes the feature is switched off for every word class; first- and second-person
#   subjects appear almost only in quotes, which is why the quote state is the real switch. On
#   eval it scores variance explained 0.984, causal variation explained 0.986, Pearson r 0.992.
#   Local ablations on 3072 training rows (in-sample, approximate) show what each part adds:
#   dropping the quote gate costs about 0.24, dropping the object rule about 0.27, dropping the
#   predicate-adjective rule about 0.02, dropping the adverb rule about 0.07, dropping the place
#   list about 0.09. A plain token-identity lookup reaches only about 0.65 in-sample, so the
#   clause-state rule is what carries the fit. Lists are needed for intransitive and transitive
#   verbs and for places, because WordNet has no transitivity and its location category misses
#   forest, ocean and river.

"""Component 442 of layer 1's MLP down projection: the predicate word of a narrated clause.

In narration (text outside quotation marks) this component is active on the word that
completes a clause's predicate, the word the clause hangs on:

- an intransitive verb (runs, sleeps, flies): about 4.8
- a manner or time adverb after such a verb (quickly, happily, always): about 4.0
- a place noun (park, forest): about 2.9
- an adjective right after is/are/was (is big): about 3.2
- the object noun after a transitive verb (sees the dog): about 3.2, and the name after "named": about 2.9

It is zero for subjects, determiners, attributive adjectives ("the big dog"), function words,
the transitive verbs themselves, and every word inside quotation marks, where it switches off.
Verb forms come from a base list: the third-person singular (runs, sees, flies) is derived.
The state carried along is: inside quotes, a verb seen in this clause, an open object slot,
the previous word, and whether the previous word was a copula.
"""

from collections.abc import Mapping

import closed_class
import wordnet

PREDICTS: tuple[str, ...] = ("h.1.mlp.down_proj:442",)
READS: tuple[str, ...] = ()
SUBJECT = "h.1.mlp.down_proj:442"

INTRANSITIVE = set("""
    bark blink bounce bow buzz cheer chirp clap crawl cry dance dash
    dream drift fall float fly frown giggle glow grin grow hide hop howl
    hurry jump kneel laugh leap listen march nod paddle play pout purr
    quack race rest return rise roar roll run rush scream shine shiver
    shout sigh sing sink sit skate skip sleep slide smile sneeze snore
    spin squeak stand stop stumble sway swim think travel tumble wait
    wake walk wander wave wiggle wobble yawn
""".split())

TRANSITIVE = set("""
    admire ask bake borrow break bring build bump buy call carry catch
    chase chew close collect comb cook count cover crush cut dig
    discover draw drink drop eat enjoy explore feed fill find fix fold
    follow forget give grab greet guard hang hear help hit hold hug hunt
    inspect join keep kick kiss lick lift like lose love make meet mend
    miss mix need notice obey open pack paint pet pick pinch plant poke
    polish pour praise protect pull push read remember rescue rub save
    say scare scrub search see sell shake share smell sniff squeeze
    stack steal stir stroke sweep take taste teach tease thank throw
    tickle tie toss touch trade trust visit want warn wash watch weigh
    wrap write
""".split())

PLACES = set("""
    attic bakery barn beach bridge castle cave cellar city classroom
    desert farm field forest garden harbor hill hospital house island
    jungle kitchen lake library lighthouse market meadow mountain museum
    nest ocean orchard palace park playground pond river room school
    shop station street swamp tower tunnel valley village yard zoo
""".split())

COPULAS = {"is", "are", "am", "was", "were"}
ADVERBS = {"always", "often", "again", "today", "together", "alone", "away", "outside", "everywhere"}
DETERMINERS = {"a", "the", "some", "many", "both", "his", "her", "their", "my", "your", "our"}
CLAUSE_ENDS = {'"', ".", "!", "?", ",", ":", ";", "[EOS]"}

VALUE_INTRANSITIVE = 4.8
VALUE_ADVERB = 4.0
VALUE_PLACE = 2.9
VALUE_PREDICATE_ADJECTIVE = 3.2
VALUE_OBJECT = 3.2
VALUE_NAMED = 2.9


def third_person(verb: str) -> str:
    """The third-person singular of a base verb: run, runs; fly, flies; wash, washes."""
    if verb.endswith("y") and verb[-2] not in "aeiou":
        return verb[:-1] + "ies"
    if verb.endswith(("s", "sh", "ch", "x", "z", "o")):
        return verb + "es"
    return verb + "s"


INTRANSITIVE_FORMS = {form for verb in INTRANSITIVE for form in (verb, third_person(verb))}
TRANSITIVE_FORMS = {form for verb in TRANSITIVE for form in (verb, third_person(verb))}


def is_function_word(word: str) -> bool:
    return bool(closed_class.classes(word))


def is_adverb(word: str) -> bool:
    """Manner and time adverbs: -ly words that are not nouns (names like Holly), plus a few."""
    if word in ADVERBS:
        return True
    return word.endswith("ly") and "noun" not in wordnet.parts_of_speech(word)


def has_adjective_sense(word: str) -> bool:
    return "adjective" in wordnet.parts_of_speech(word)


START = (False, False, False, "", False)


def step(carry, prefix, upstream: Mapping):
    in_quotes, verb_seen, object_open, previous, after_copula = carry or START
    word = prefix.token_strings[-1].strip()
    value = 0.0
    if word == '"':
        in_quotes = not in_quotes
    if word in CLAUSE_ENDS:
        verb_seen = object_open = after_copula = False
    elif word in INTRANSITIVE_FORMS:
        value, verb_seen = VALUE_INTRANSITIVE, True
    elif word in COPULAS:
        verb_seen = True
    elif word in TRANSITIVE_FORMS:
        verb_seen = object_open = True
    elif after_copula and not is_function_word(word):
        value = VALUE_PREDICATE_ADJECTIVE
    elif previous == "named" and not is_function_word(word):
        value = VALUE_NAMED
    elif is_adverb(word) and previous not in DETERMINERS:
        value = VALUE_ADVERB if verb_seen else 0.0
    elif word in PLACES:
        value = VALUE_PLACE
    elif is_function_word(word):
        pass
    elif object_open and not has_adjective_sense(word):
        value, object_open = VALUE_OBJECT, False
    if in_quotes:
        value = 0.0
    carry = (in_quotes, verb_seen, object_open, word, word in COPULAS)
    return carry, {SUBJECT: value}
