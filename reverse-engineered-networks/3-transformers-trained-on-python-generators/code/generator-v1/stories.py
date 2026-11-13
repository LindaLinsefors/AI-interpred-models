"""The true program: a story generator whose next-word distribution is exact.

Each grammar rule is a coroutine. At every step it yields the distribution of the next word
and is sent the word that was drawn; it holds no randomness of its own. Its classes of words
never overlap within one distribution, so the drawn word says which choice was made. Replaying
any text the generator could have produced therefore recovers its state and gives the exact
p(next word | text so far).

What the text exercises:
- a grammar over about 1.3k words;
- subject-verb number agreement ("the dog runs", "the dogs run", "i am", "we are");
- he/she/his/her bound to the gender of the most recently named character;
- characters' names, introduced at the start of a story and copied later;
- quotes, opened and closed by the same token;
- numbered lists whose length is announced, so the counter says when they end.
"""

import bisect
import random
from collections.abc import Generator
from dataclasses import dataclass, field
from functools import cache

import lexicon as lx

EOS = "[EOS]"


class Words:
    """A class of words and the probability of each within it."""

    def __init__(self, name: str, words: list[str], weights: list[float] | None = None):
        weights = weights or [1.0] * len(words)
        total = sum(weights)
        self.name = name
        self.words = tuple(words)
        self.probs = tuple(w / total for w in weights)
        self.cumulative = [sum(self.probs[: i + 1]) for i in range(len(words))]
        self._set = frozenset(words)

    def __contains__(self, word: str) -> bool:
        return word in self._set


def zipf(name: str, words: list[str]) -> Words:
    return Words(name, words, [1 / (rank + 1) for rank in range(len(words))])


@cache
def one(word: str) -> Words:
    return Words(word, [word])


Dist = list[tuple[float, Words]]
Program = Generator[Dist, str, None]


def mix(*parts: tuple[float, "Words | Dist"]) -> Dist:
    """A weighted mixture of word classes and distributions; zero weights drop out."""
    total = sum(p for p, _ in parts)
    out: Dist = []
    for p, part in parts:
        if p == 0:
            continue
        if isinstance(part, Words):
            out.append((p / total, part))
        else:
            out.extend((p * q / total, words) for q, words in part)
    return out


def say(text: str) -> Program:
    for word in text.split():
        yield [(1.0, one(word))]


NAMES = {"girl": Words("girl names", lx.FEMALE_NAMES), "boy": Words("boy names", lx.MALE_NAMES)}
PRONOUN = {"girl": "she", "boy": "he"}
POSSESSIVE = {"girl": "her", "boy": "his"}
NOUN = {
    "sg": zipf("nouns", lx.NOUNS),
    "pl": zipf("plural nouns", [lx.plural(n) for n in lx.NOUNS]),
}
ADJECTIVE = zipf("adjectives", lx.ADJECTIVES)
ADVERB = zipf("adverbs", lx.ADVERBS)
PREPOSITION = zipf("prepositions", lx.PREPOSITIONS)
PLACE = zipf("places", lx.PLACES)
NUMBER = Words("number words", lx.NUMBER_WORDS)

# A subject asks its verb for one of three forms: "s" (third person singular), "base"
# (plural, you, we) or "i".
INTRANSITIVE = {
    "s": zipf("intransitive verbs -s", [lx.third_person(v) for v in lx.INTRANSITIVE_VERBS]),
    "base": zipf("intransitive verbs", lx.INTRANSITIVE_VERBS),
}
TRANSITIVE = {
    "s": zipf("transitive verbs -s", [lx.third_person(v) for v in lx.TRANSITIVE_VERBS]),
    "base": zipf("transitive verbs", lx.TRANSITIVE_VERBS),
}
INTRANSITIVE["i"], TRANSITIVE["i"] = INTRANSITIVE["base"], TRANSITIVE["base"]
COPULA = {"s": one("is"), "base": one("are"), "i": one("am")}
SAY = {"s": one("says"), "base": one("say"), "i": one("say")}

SUBJECT_DETERMINER = Words("determiners", ["the", "a", "some", "many"], [4, 3, 1, 1])
QUOTE_SUBJECT = mix((0.3, one("i")), (0.2, one("you")), (0.1, one("we")), (0.4, SUBJECT_DETERMINER))
QUOTE_END = Words("quote sentence ends", [".", "!"], [3, 1])


@dataclass
class Character:
    name: str
    gender: str  # "girl" or "boy"


@dataclass
class Subject:
    form: str  # the verb form it asks for: "s", "base" or "i"
    possessive: str | None  # the determiner that refers back to it: "his", "her", "their"


@dataclass
class State:
    """What the generator tracks. Alignment labels read it as the next word is chosen."""

    characters: list[Character] = field(default_factory=list)
    last: Character | None = None  # the most recently named character: he and she refer to it
    in_quote: bool = False
    list_item: int = 0  # the number of the item being listed; 0 outside a list
    list_length: int = 0
    form: str | None = None  # the verb form the current subject asks for
    sentences: int = 0

    def character(self, name: str) -> Character:
        return next(c for c in self.characters if c.name == name)

    def names(self, exclude: str | None = None) -> Words:
        return Words("characters", [c.name for c in self.characters if c.name != exclude])


def stream(states: list[State] | None = None) -> Program:
    """Stories one after another, each ended by [EOS]: the text rows are cut from. Each
    story's state is appended to `states` when it starts."""
    while True:
        s = State()
        if states is not None:
            states.append(s)
        yield from story(s)


def story(s: State) -> Program:
    yield from say("once there was")
    yield from introduce(s)
    while True:
        end = 0.0 if s.sentences < 4 else 0.1
        w = yield mix((end, one(EOS)), (1 - end, subject_start(s)))
        if w == EOS:
            return
        subject = yield from subject_rest(s, w)
        yield from predicate(s, subject)
        s.form = None
        s.sentences += 1


def introduce(s: State) -> Program:
    """'a girl named mia and a boy named leo .': one to three characters."""
    while True:
        yield from say("a")
        gender = yield mix((0.5, one("girl")), (0.5, one("boy")))
        yield from say("named")
        taken = {c.name for c in s.characters}
        name = yield [(1.0, Words("names", [n for n in NAMES[gender].words if n not in taken]))]
        s.characters.append(Character(name, gender))
        s.last = s.characters[-1]
        more = 0.4 if len(s.characters) < 3 else 0.0
        if (yield mix((more, one("and")), (1 - more, one(".")))) == ".":
            return


def subject_start(s: State) -> Dist:
    assert s.last is not None
    several = len(s.characters) > 1
    return mix(
        (0.25, one(PRONOUN[s.last.gender])),
        (0.35, s.names()),
        (0.05 * several, one("they")),
        (0.05 * several, one("both")),
        (0.30, SUBJECT_DETERMINER),
    )


def subject_rest(s: State, w: str) -> Generator[Dist, str, Subject]:
    if w in ("he", "she"):
        assert s.last is not None
        return Subject("s", POSSESSIVE[s.last.gender])
    if w == "they":
        return Subject("base", "their")
    if w == "both":  # 'both mia and leo'
        first = yield [(1.0, s.names())]
        yield from say("and")
        s.last = s.character((yield [(1.0, s.names(exclude=first))]))
        return Subject("base", "their")
    if w in SUBJECT_DETERMINER:
        number = yield from noun_rest(w)
        return Subject("s" if number == "sg" else "base", None)
    s.last = s.character(w)
    return Subject("s", POSSESSIVE[s.last.gender])


def noun_rest(determiner: str) -> Generator[Dist, str, str]:
    """Up to two adjectives and a noun after `determiner`; returns the noun's number."""
    numbers = {"a": ["sg"], "some": ["pl"], "many": ["pl"]}.get(determiner, ["sg", "pl"])
    for adjectives in range(3):
        more = 0.3 if adjectives < 2 else 0.0
        nouns = [((1 - more) / len(numbers), NOUN[n]) for n in numbers]
        w = yield mix((more, ADJECTIVE), *nouns)
        if w not in ADJECTIVE:
            return "sg" if w in NOUN["sg"] else "pl"
    raise AssertionError("unreachable")


def predicate(s: State, subject: Subject) -> Program:
    s.form = f = subject.form
    named = subject.possessive in ("his", "her")
    w = yield mix(
        (0.45, INTRANSITIVE[f]),
        (0.25, TRANSITIVE[f]),
        (0.10, COPULA[f]),
        (0.12, SAY[f]),
        (0.08 * named, one("made")),
    )
    if w in SAY[f]:
        yield from speech(s)
    elif w == "made":
        yield from make_list(s)
    else:
        yield from verb_phrase(s, subject, w, one("."))


def verb_phrase(s: State, subject: Subject, verb: str, end: Words) -> Program:
    """The rest of a verb phrase after `verb`: its object or adjective, then perhaps an
    adverb, a place and 'and' another verb phrase, then `end`."""
    f = subject.form
    if verb in TRANSITIVE[f]:
        yield from object_phrase(s, subject)
    elif verb in COPULA[f]:
        yield [(1.0, ADJECTIVE)]
    adverb = 0.3 if verb in INTRANSITIVE[f] else 0.0
    w = yield mix((adverb, ADVERB), (0.2, PREPOSITION), (0.1, one("and")), (0.7 - adverb, end))
    if w in ADVERB:
        w = yield mix((0.2, PREPOSITION), (0.1, one("and")), (0.7, end))
    if w in PREPOSITION:
        yield from say("the")
        yield [(1.0, PLACE)]
        w = yield mix((0.1, one("and")), (0.9, end))
    if w == "and":
        verb = yield mix((0.5, INTRANSITIVE[f]), (0.35, TRANSITIVE[f]), (0.15, COPULA[f]))
        yield from verb_phrase(s, subject, verb, end)


def object_phrase(s: State, subject: Subject) -> Program:
    if s.in_quote:
        determiners = Words("determiners", ["the", "a", "some", "my", "your", "our"])
        w = yield [(1.0, determiners)]
    else:
        determiners = Words("determiners", ["the", "a", "some", "many"], [4, 3, 1, 1])
        owner = [(0.25, one(subject.possessive))] if subject.possessive else []
        w = yield mix((0.6, determiners), *owner, (0.15, s.names()))
        if w in s.names():
            s.last = s.character(w)
            return
    yield from noun_rest(w)


def speech(s: State) -> Program:
    """', " i like the dogs . the cat is happy ! "': one or more sentences in quotes."""
    yield from say(', "')
    s.in_quote = True
    w = yield QUOTE_SUBJECT
    while w != '"':
        if w in SUBJECT_DETERMINER:
            subject = Subject("s" if (yield from noun_rest(w)) == "sg" else "base", None)
        else:
            subject = Subject("i" if w == "i" else "base", None)
        s.form = f = subject.form
        verb = yield mix((0.45, INTRANSITIVE[f]), (0.4, TRANSITIVE[f]), (0.15, COPULA[f]))
        yield from verb_phrase(s, subject, verb, QUOTE_END)
        s.form = None
        w = yield mix((0.6, one('"')), (0.4, QUOTE_SUBJECT))
    s.in_quote = False


def make_list(s: State) -> Program:
    """'a list of three things : 1 a red ball , 2 the cats , 3 some cakes .'"""
    yield from say("a list of")
    s.list_length = lx.NUMBER_WORDS.index((yield [(1.0, NUMBER)])) + 2
    yield from say("things :")
    for k in range(1, s.list_length + 1):
        s.list_item = k
        yield from say(str(k))
        yield from noun_rest((yield [(1.0, Words("determiners", ["the", "a", "some"]))]))
        yield from say("," if k < s.list_length else ".")
    s.list_item = s.list_length = 0


def sample(program: Program, n_words: int, rng: random.Random) -> list[str]:
    """The first `n_words` words `program` produces, drawn with `rng`."""
    words: list[str] = []
    dist = next(program)
    while True:
        u = rng.random()
        for p, part in dist:
            if u < p or part is dist[-1][1]:
                break
            u -= p
        words.append(part.words[min(bisect.bisect(part.cumulative, rng.random()), len(part.words) - 1)])
        if len(words) == n_words:
            return words
        dist = program.send(words[-1])


def replay(program: Program, words: list[str]) -> list[Dist]:
    """The distribution `program` gives each of `words`, the first included, as it is fed
    them in turn."""
    dists = [next(program)]
    for w in words[:-1]:
        assert any(w in part for _, part in dists[-1]), f"{w!r} is impossible here"
        dists.append(program.send(w))
    return dists
