"""The true program, v2: stories in a small world, with an exact next-word distribution.

Characters move between places, take, drop, give and use things, find and lose coins, talk,
list what they see and answer questions, and the text stays true to the world. Beneath that
are values the text never states:
- each noun has a hidden kind, and adjectives and transitive verbs fit only some kinds;
- each story has a hidden theme that favours some nouns, places and adjectives;
- each character has a hidden temperament that sets what they tend to do, and how;
- laws: whether an actor goes or acts depends on whether exactly one of two facts about them
  holds, some of the facts hidden halves of the places and names.

Every choice is a `choose` among options, each a weight, the words that say it and what it
stands for; the generator holds no randomness of its own. Options' words may coincide, and
`choose` keeps the posterior over the options its words fit. A latent value enters a choice as
one option per value it may take, weighted by the belief in that value times the option's
probability given it, each value's distribution normalised on its own; the posterior over the
options then updates the belief by Bayes' rule. Sampling knows each latent value (its belief is
certain), so the same code samples stories; replaying a text from the prior gives the exact
p(next word | text so far) and the generator's state.
"""

import bisect
import itertools
import random
from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass, field
from functools import cache

import numpy as np

import lexicon as lx

EOS = "[EOS]"


class Words:
    """A class of words and the probability of each within it."""

    def __init__(self, name: str, words: Sequence[str], weights: Sequence[float] | None = None):
        weights = [1.0] * len(words) if weights is None else weights
        total = sum(weights)
        self.name = name
        self.words = tuple(words)
        self.probs = tuple(w / total for w in weights)
        self.cumulative = list(itertools.accumulate(self.probs))
        self._prob = dict(zip(self.words, self.probs))
        assert len(self._prob) == len(self.words), f"{name} repeats a word"

    def __contains__(self, word: str) -> bool:
        return word in self._prob

    def prob(self, word: str) -> float:
        return self._prob.get(word, 0.0)

    def sample(self, rng: random.Random) -> str:
        return self.words[min(bisect.bisect(self.cumulative, rng.random()), len(self.words) - 1)]

    def without(self, excluded: Iterable[str]) -> "Words":
        gone = frozenset(w for w in excluded if w in self)
        return Without(self, gone) if gone else self


class Without(Words):
    """`base` without the words `gone`, renormalised; sampled by drawing from `base` again."""

    def __init__(self, base: Words, gone: frozenset[str]):
        self.base, self.gone = base, gone
        self.name = base.name
        self.mass = 1 - sum(base.prob(w) for w in gone)
        assert self.mass > 1e-9, f"{base.name} without {sorted(gone)} is empty"

    @property
    def words(self) -> tuple[str, ...]:  # type: ignore[override]
        return tuple(w for w in self.base.words if w not in self.gone)

    @property
    def probs(self) -> tuple[float, ...]:  # type: ignore[override]
        return tuple(self.base.prob(w) / self.mass for w in self.words)

    def __contains__(self, word: str) -> bool:
        return word in self.base and word not in self.gone

    def prob(self, word: str) -> float:
        return 0.0 if word in self.gone else self.base.prob(word) / self.mass

    def sample(self, rng: random.Random) -> str:
        while (word := self.base.sample(rng)) in self.gone:
            pass
        return word


def zipf(name: str, words: Sequence[str], boosted: frozenset[str] = frozenset(), boost: float = 1.0) -> Words:
    return Words(name, words, [(boost if w in boosted else 1.0) / (rank + 1) for rank, w in enumerate(words)])


@cache
def one(word: str) -> Words:
    return Words(word, [word])


Dist = list[tuple[float, Words]]


@dataclass(frozen=True)
class Option:
    weight: float
    words: tuple[Words, ...]
    value: object


def option(weight: float, words: Iterable[Words | str], value: object = None) -> Option:
    """`words` given as strings are said as they are, split at spaces."""
    said = tuple(w for part in words for w in ((one(x) for x in part.split()) if isinstance(part, str) else (part,)))
    return Option(weight, said, value)


@dataclass
class Said:
    words: list[str]
    values: list[tuple[float, object]]  # the options the words fit, with their posteriors

    def value(self, at: int | None = None) -> object:
        """The value the words said (or its `at`-th part), which they must leave in no doubt."""
        values = {v if at is None else v[at] for _, v in self.values}  # type: ignore[index]
        assert len(values) == 1, values
        return values.pop()

    def posterior(self, at: int) -> dict[object, float]:
        out: dict[object, float] = {}
        for p, v in self.values:
            out[v[at]] = out.get(v[at], 0.0) + p  # type: ignore[index]
        return out


Program = Generator[Dist, str, None]
Choice = Generator[Dist, str, Said]


def choose(options: Iterable[Option]) -> Choice:
    """One of `options`, said word by word: each word is drawn from the options whose words so
    far fit, by weight. Returns the words and the options they fit."""
    live = [(o.weight, o) for o in options if o.weight > 0]
    assert live, "nothing to choose from"
    words: list[str] = []
    for i in itertools.count():
        if any(len(o.words) == i for _, o in live):
            assert all(len(o.words) == i for _, o in live), "an option's words begin another's"
            total = sum(p for p, _ in live)
            return Said(words, [(p / total, o.value) for p, o in live])
        total = sum(p for p, _ in live)
        word = yield [(p / total, o.words[i]) for p, o in live]
        live = [(p * o.words[i].prob(word), o) for p, o in live if word in o.words[i]]
        assert live, f"{word!r} is impossible here"
        words.append(word)
    raise AssertionError("unreachable")


def say(text: str) -> Choice:
    return choose([option(1.0, [text])])


class Latent:
    """A value the text never states. Sampling knows it; replay starts from the uniform prior,
    and each choice whose options depend on it updates the belief by Bayes' rule."""

    def __init__(self, n: int, value: int | None, frozen: bool = False):
        self.belief = np.full(n, 1 / n) if value is None else np.eye(n)[value]
        self.frozen = frozen  # a replay ignoring it keeps the prior (`ablate`)

    def support(self) -> list[tuple[int, float]]:
        return [(k, float(b)) for k, b in enumerate(self.belief) if b > 0]

    def update(self, said: Said, at: int) -> None:
        """Condition on the words of a choice whose values hold this latent's value at `at`."""
        if not self.frozen:
            posterior = np.zeros_like(self.belief)
            for k, p in said.posterior(at).items():
                posterior[k] = p  # type: ignore[index]
            self.belief = posterior / posterior.sum()


# The world's hidden tables, drawn once.

HIDDEN = random.Random(2432)
N_KINDS, N_THEMES, N_TEMPERAMENTS = 12, 16, 4
KIND = {noun: HIDDEN.randrange(N_KINDS) for noun in lx.NOUNS}
ADJECTIVE_KINDS = {a: frozenset(HIDDEN.sample(range(N_KINDS), 2)) for a in lx.ADJECTIVES}
VERB_KINDS = {v: frozenset(HIDDEN.sample(range(N_KINDS), 2)) for v in lx.TRANSITIVE_VERBS}
THEME_BOOST = 60.0
THEMED = {
    cls: [frozenset(HIDDEN.sample(words, len(words) // N_THEMES)) for _ in range(N_THEMES)]
    for cls, words in (("girl", lx.FEMALE_NAMES), ("boy", lx.MALE_NAMES), ("noun", lx.NOUNS), ("place", lx.PLACES),
                       ("adjective", lx.ADJECTIVES))
}
EVENTS = "go take drop give look use find lose say answer see act".split()
BASE_EVENT_WEIGHT = dict(go=3.0, take=2.0, drop=1.0, give=1.5, look=1.5, use=1.5, find=1.0, lose=0.7, say=1.5,
                         see=0.7, act=3.0)
TEMPERAMENT_BOOST = 40.0
TEMPERAMENT_EVENT_WEIGHT = [
    {e: w * HIDDEN.gammavariate(0.35, 1 / 0.35) for e, w in BASE_EVENT_WEIGHT.items()} for _ in range(N_TEMPERAMENTS)
]
TEMPERAMENT_VERBS = [frozenset(HIDDEN.sample(lx.INTRANSITIVE_VERBS, len(lx.INTRANSITIVE_VERBS) // 8))
                     for _ in range(N_TEMPERAMENTS)]
TEMPERAMENT_ADVERBS = [frozenset(HIDDEN.sample(lx.ADVERBS, len(lx.ADVERBS) // 8)) for _ in range(N_TEMPERAMENTS)]
BRIGHT_PLACES = frozenset(HIDDEN.sample(lx.PLACES, len(lx.PLACES) // 2))
MARKED_NAMES = frozenset(HIDDEN.sample(lx.FEMALE_NAMES + lx.MALE_NAMES, (len(lx.FEMALE_NAMES) + len(lx.MALE_NAMES)) // 2))
FACTS = ("girl", "company", "holding", "bright", "marked", "early")
LAW_BOOST = 6.0


@dataclass(frozen=True)
class Law:
    """Tilts whether the actor goes or acts: LAW_BOOST times toward `plus` when exactly one of
    `facts` holds of them, toward the other when both or neither do."""

    facts: tuple[str, str]
    plus: str  # "go" or "act"

    def favoured(self, holds: dict[str, bool]) -> str:
        a, b = self.facts
        return self.plus if holds[a] != holds[b] else {"go": "act", "act": "go"}[self.plus]


LAWS = tuple(Law(pair, HIDDEN.choice(["go", "act"])) for pair in HIDDEN.sample(list(itertools.combinations(FACTS, 2)), 8))


@dataclass(frozen=True)
class Config:
    """Which parts of the world a corpus has: the learnability ladder trains a target on each."""

    name: str
    kinds: bool = True  # adjectives and transitive verbs fit only some kinds of thing
    laws: bool = True
    describe: bool = True  # "the girl with the kite"
    questions: bool = True
    lists: bool = True  # "sees three things : ..." and "takes the second one"
    arithmetic: bool = True  # coins
    dialogue: bool = True
    theme: bool = True
    temperament: bool = True

    def has(self, event: str) -> bool:
        needs = {"find": self.arithmetic, "lose": self.arithmetic, "say": self.dialogue, "answer": self.dialogue,
                 "see": self.lists}
        return needs.get(event, True)


FULL = Config("full")
CONFIGS = {
    c.name: c
    for c in (
        Config("core", lists=False, arithmetic=False, dialogue=False, theme=False, temperament=False),
        Config("core_theme", lists=False, arithmetic=False, dialogue=False, temperament=False),
        Config("core_temperament", lists=False, arithmetic=False, dialogue=False, theme=False),
        Config("visible", theme=False, temperament=False),
        FULL,
    )
}

# Word classes.

NAMES = {"girl": Words("girl names", lx.FEMALE_NAMES), "boy": Words("boy names", lx.MALE_NAMES)}
SUBJECT = {"girl": "she", "boy": "he"}
OBJECT = {"girl": "her", "boy": "him"}
VERB = dict(go="goes", take="takes", drop="drops", give="gives", look="looks", find="finds", lose="loses",
            say="says", answer="answers", see="sees")
ACTS = [zipf(f"verbs t{t}", [lx.third_person(v) for v in lx.INTRANSITIVE_VERBS],
             frozenset(lx.third_person(v) for v in verbs), TEMPERAMENT_BOOST)
        for t, verbs in enumerate(TEMPERAMENT_VERBS)]
ADVERBS = [zipf(f"adverbs t{t}", lx.ADVERBS, adverbs, TEMPERAMENT_BOOST) for t, adverbs in enumerate(TEMPERAMENT_ADVERBS)]
PLAIN_ACTS = zipf("verbs", [lx.third_person(v) for v in lx.INTRANSITIVE_VERBS])
PLAIN_ADVERBS = zipf("adverbs", lx.ADVERBS)
NUMBER = {k: lx.NUMBER_WORDS[k - 2] for k in range(2, 21)}
COUNTABLES = Words("countables", lx.COUNTABLES)


@cache
def themed(cls: str, theme: int | None) -> Words:
    """Names, nouns, places or adjectives, favouring `theme`'s (None: no theme)."""
    words = {"girl": lx.FEMALE_NAMES, "boy": lx.MALE_NAMES, "noun": lx.NOUNS, "place": lx.PLACES,
             "adjective": lx.ADJECTIVES}[cls]
    return zipf(f"{cls}s {theme}", words, frozenset() if theme is None else THEMED[cls][theme], THEME_BOOST)


@cache
def adjectives(kind: int | None, theme: int | None) -> Words:
    """The adjectives that fit a thing of `kind` (None: any thing), favouring `theme`'s."""
    every = themed("adjective", theme)
    fits = [(w, p) for w, p in zip(every.words, every.probs) if kind is None or kind in ADJECTIVE_KINDS[w]]
    return Words(f"adjectives k{kind} {theme}", [w for w, _ in fits], [p for _, p in fits])


@cache
def uses(kind: int | None) -> Words:
    """The transitive verbs that fit a thing of `kind` (None: any thing)."""
    return zipf(f"uses k{kind}", [lx.third_person(v) for v in lx.TRANSITIVE_VERBS if kind is None or kind in VERB_KINDS[v]])


# The world.


@dataclass(eq=False)
class Character:
    name: str
    gender: str  # "girl" or "boy"
    temperament: Latent
    place: str = ""
    coins: int = 0  # never 1: counts are said in the plural


@dataclass(eq=False)
class Thing:
    noun: str
    place: str  # where it lies when no one holds it
    holder: Character | None = None

    def where(self) -> str:
        return self.holder.place if self.holder else self.place


@dataclass
class State:
    """What the generator tracks. Labels read it as each word is chosen."""

    config: Config
    rng: random.Random | None  # draws the latent values while sampling; None while replaying
    ignore: frozenset[str] = frozenset()  # families a replay pretends not to know (`ablate`)
    theme: Latent = field(init=False)
    cast: list[Character] = field(default_factory=list)
    things: list[Thing] = field(default_factory=list)
    places: list[str] = field(default_factory=list)
    last: dict[str, Character] = field(default_factory=dict)  # the latest mention of each gender
    countable: str | None = None
    setting: bool = True  # still telling what lies where
    actor: Character | None = None  # the last sentence's
    addressed: Character | None = None  # whom the last sentence spoke to
    counted: Character | None = None  # whose coins the last sentence changed
    listed: list[Thing] = field(default_factory=list)  # what the last sentence listed
    sentences: int = 0
    slot: str = "start"  # label: the kind of choice being made
    families: frozenset[str] = frozenset()  # label: the hidden families it depends on

    def __post_init__(self) -> None:
        self.theme = self.latent(N_THEMES, "theme")

    def latent(self, n: int, family: str) -> Latent:
        if not getattr(self.config, family):
            return Latent(1, 0)
        return Latent(n, None if self.rng is None else self.rng.randrange(n), frozen=family in self.ignore)

    def at(self, slot: str, *families: str) -> None:
        self.slot, self.families = slot, frozenset(families)

    def themed(self, cls: str, theme: int) -> Words:
        return themed(cls, theme if self.config.theme else None)

    def kind(self, thing: Thing) -> int | None:
        return KIND[thing.noun] if self.config.kinds and "kinds" not in self.ignore else None

    def mention(self, c: Character) -> None:
        self.last[c.gender] = c

    def held(self, c: Character) -> list[Thing]:
        return [t for t in self.things if t.holder is c]

    def visible(self, c: Character) -> list[Thing]:
        return [t for t in self.things if t.where() == c.place]

    def others_here(self, c: Character) -> list[Character]:
        return [o for o in self.cast if o is not c and o.place == c.place]

    def unused(self, words: Words) -> Words:
        return words.without(t.noun for t in self.things).without(self.places)


def stream(config: Config = FULL, rng: random.Random | None = None, states: list[State] | None = None,
           ignore: frozenset[str] = frozenset()) -> Program:
    """Stories one after another, each ended by [EOS]: the text rows are cut from. Sampling
    passes `rng`, which draws the latent values; each story's state is appended to `states`."""
    while True:
        s = State(config, rng, ignore)
        if states is not None:
            states.append(s)
        yield from story(s)


def story(s: State) -> Program:
    s.at("setup")
    yield from say("once there was")
    yield from introduce(s)
    for c in s.cast:
        s.at("setup")
        yield from say(f"{c.name} is in the")
        s.at("place", "theme")
        c.place = yield from place(s, c, new=1.5 if s.places else 1.0)
        s.at("setup")
        yield from say(".")
    s.at("thing", "theme")
    yield from thing_there(s, (yield from choose(thing_options(s, 1.0))))
    while True:
        s.at("sentence")
        said = yield from choose(sentence_options(s))
        match said.value(0):
            case "end":
                return
            case "thing":
                yield from thing_there(s, said)
                continue
            case "now":
                yield from say(".")
                s.actor = s.addressed = s.counted = None
                s.listed = []
            case "question":
                yield from answer(s, said.value(1))
                s.actor = s.addressed = s.counted = None
                s.listed = []
            case "actor":
                actor = said.value(1)
                s.mention(actor)  # type: ignore[arg-type]
                yield from event(s, actor)  # type: ignore[arg-type]
        s.setting = False
        s.sentences += 1


def introduce(s: State) -> Program:
    """'a girl named mia and a boy named leo .': two to four characters."""
    while True:
        s.at("setup")
        gender = (yield from choose([option(0.5, ["a girl named"], "girl"), option(0.5, ["a boy named"], "boy")])).value()
        s.at("name", "theme")
        said = yield from choose(option(b, [s.themed(gender, z).without(c.name for c in s.cast)], (None, z))  # type: ignore[arg-type]
                                 for z, b in s.theme.support())
        s.theme.update(said, at=1)
        name = said.words[0]
        c = Character(name, gender, s.latent(N_TEMPERAMENTS, "temperament"))  # type: ignore[arg-type]
        s.cast.append(c)
        s.mention(c)
        more = {1: 1.0, 2: 0.5, 3: 0.3}.get(len(s.cast), 0.0)
        s.at("setup")
        if (yield from choose([option(more, ["and"], True), option(1 - more, ["."], False)])).value() is False:
            return


def place(s: State, c: Character, new: float) -> Generator[Dist, str, str]:
    """A place `c` is not at: one of the story's, or (by weight `new`, while it has fewer than
    four) a new one, favouring the theme's."""
    known = [p for p in s.places if p != c.place]
    new = new if len(s.places) < 4 else 0.0
    said = yield from choose([
        *(option(1.0, [p], ("known", None)) for p in known),
        *(option(b * new, [s.unused(s.themed("place", z))], ("new", z)) for z, b in s.theme.support()),
    ])
    if said.value(0) == "new":
        s.theme.update(said, at=1)
        s.places.append(said.words[0])
    return said.words[0]


def thing_options(s: State, weight: float) -> list[Option]:
    """'there is a kite': a new thing, favouring the theme's nouns."""
    return [option(weight * b, ["there is a", s.unused(s.themed("noun", z))], ("thing", z)) for z, b in s.theme.support()]


def thing_there(s: State, said: Said) -> Program:
    """'... in the barn .': where the thing `said` lies."""
    s.theme.update(said, at=1)
    noun = said.words[-1]
    s.at("place")
    where = (yield from choose([option(1.0, ["in the", p, "."], p) for p in s.places])).value()
    s.things.append(Thing(noun, where))  # type: ignore[arg-type]


def sentence_options(s: State) -> list[Option]:
    """How the next sentence starts: the end of the story, another thing (while setting the
    scene), 'now she has ...' after coins change, a question, or a character who acts."""
    end = 0.0 if s.sentences < 8 else 1.0 if s.sentences >= 24 else 0.1
    rest = 1 - end
    options = [option(end, [EOS], ("end",))]
    if s.setting and len(s.things) < 3:
        options += thing_options(s, rest * 0.6)
        rest *= 0.4
    if s.counted is not None:
        c = s.counted
        has = ["has", NUMBER[c.coins] if c.coins else "no", s.countable]
        refs = references(s, c, pronoun=0.6, name=0.4)
        options += [option(rest * 0.5 * p, ["now", *ref, *has], ("now",)) for p, ref in refs]
        rest *= 0.5
    asks = questions(s)
    if s.config.questions and s.sentences >= 2 and asks:
        options += [option(rest * 0.15 * p, words, ("question", q)) for p, words, q in asks]
        rest *= 0.85
    weights = {c: 1.0 + (c is s.actor) + 2.0 * (c is s.addressed) for c in s.cast}
    total = sum(weights.values())
    for c, w in weights.items():
        refs = references(s, c, pronoun=1.5, name=1.0, describe=0.6 if s.config.describe else 0.0)
        options += [option(rest * w / total * p, ref, ("actor", c)) for p, ref in refs]
    return options


def references(s: State, c: Character, pronoun: float = 0.0, name: float = 0.0, describe: float = 0.0,
               case: dict[str, str] = SUBJECT) -> list[tuple[float, list[str]]]:
    """Ways to refer to `c`, with probabilities: by name; by pronoun, if `c` is the latest of
    its gender mentioned; as 'the girl with the kite', if `c` holds something."""
    held = s.held(c)
    ways = [(name, [c.name])]
    if s.last.get(c.gender) is c:
        ways.append((pronoun, [case[c.gender]]))
    ways += [(describe / len(held), ["the", c.gender, "with the", t.noun]) for t in held]
    total = sum(w for w, _ in ways)
    return [(w / total, words) for w, words in ways if w > 0]


def other(s: State, actor: Character, others: list[Character]) -> Choice:
    """One of `others` the actor turns to: by name, or by pronoun when that cannot mean the
    actor."""
    options = []
    for b in others:
        pronoun = 0.4 if b.gender != actor.gender else 0.0
        options += [option(p / len(others), ref, b) for p, ref in references(s, b, pronoun, 0.6, case=OBJECT)]
    return (yield from choose(options))


def questions(s: State) -> list[tuple[float, list[str], tuple]]:
    """What the story may ask: where a thing or character is, who has a thing, how many coins
    someone has."""
    kinds = [
        [(["where is the", t.noun, "?"], ("where", t)) for t in s.things],
        [(["who has the", t.noun, "?"], ("who", t)) for t in s.things],
        [(["where is", c.name, "?"], ("where", c)) for c in s.cast],
    ]
    if s.countable:
        kinds.append([(["how many", s.countable, "does", c.name, "have ?"], ("many", c)) for c in s.cast])
    kinds = [k for k in kinds if k]
    return [(1 / len(kinds) / len(k), words, q) for k in kinds for words, q in k]


def answer(s: State, question: tuple) -> Program:
    """The true answer: 'it is in the barn .', 'leo has it .', 'nobody has it .', 'she is in
    the barn .', 'he has five coins .'"""
    ask, about = question
    s.at("answer")
    if isinstance(about, Character):
        s.mention(about)
        if ask == "where":
            yield from say(f"{SUBJECT[about.gender]} is in the {about.place}")
        else:
            yield from say(f"{SUBJECT[about.gender]} has {NUMBER[about.coins] if about.coins else 'no'} {s.countable}")
    elif ask == "where":
        yield from say(f"it is in the {about.where()}")
    elif about.holder:
        s.mention(about.holder)
        yield from say(f"{about.holder.name} has it")
    else:
        yield from say("nobody has it")
    yield from say(".")


def facts(s: State, c: Character) -> dict[str, bool]:
    return {
        "girl": c.gender == "girl",
        "company": bool(s.others_here(c)),
        "holding": bool(s.held(c)),
        "bright": c.place in BRIGHT_PLACES,
        "marked": c.name in MARKED_NAMES,
        "early": s.sentences < 8,
    }


def event_probs(s: State, c: Character, temperament: int) -> dict[str, float]:
    """What `c` may do next, given its temperament: weights the temperament and the laws set,
    normalised over what is possible; an answer, when `c` was just spoken to, half the time."""
    held, visible, here = s.held(c), s.visible(c), s.others_here(c)
    possible = {
        "go": True,
        "take": any(t.holder is None for t in visible),
        "drop": bool(held),
        "give": bool(held) and bool(here),
        "look": bool(visible),
        "use": bool(visible),
        "find": c.coins <= 18,
        "lose": c.coins >= 2,
        "say": bool(here),
        "see": len(s.things) <= 8,
        "act": True,
    }
    weights = TEMPERAMENT_EVENT_WEIGHT[temperament] if s.config.temperament else BASE_EVENT_WEIGHT
    w = {e: weights[e] for e, ok in possible.items() if ok and s.config.has(e)}
    if s.config.laws and "laws" not in s.ignore:  # the laws share going and acting's weight
        holds = facts(s, c)
        tilt = {"go": w["go"], "act": w["act"]}
        for law in LAWS:
            tilt[law.favoured(holds)] *= LAW_BOOST
        pair, tilted = w["go"] + w["act"], tilt["go"] + tilt["act"]
        w["go"], w["act"] = pair * tilt["go"] / tilted, pair * tilt["act"] / tilted
    answering = 0.5 if s.config.dialogue and s.addressed is c else 0.0
    total = sum(w.values())
    return {"answer": answering} | {e: (1 - answering) * x / total for e, x in w.items()}


def event(s: State, c: Character) -> Program:
    """What `c` does, said by its verb; a transitive verb says which visible thing it may be
    used on."""
    s.at("event", "temperament", "laws", "kinds")
    visible = s.visible(c)
    options = []
    for t, b in c.temperament.support():
        for e, p in event_probs(s, c, t).items():
            if e == "use":
                options += [option(b * p / len(visible), [uses(s.kind(x))], (e, t, x)) for x in visible]
            elif e == "act":
                options.append(option(b * p, [ACTS[t] if s.config.temperament else PLAIN_ACTS], (e, t, None)))
            else:
                options.append(option(b * p, [VERB[e]], (e, t, None)))
    said = yield from choose(options)
    c.temperament.update(said, at=1)
    e = said.value(0)
    listed, s.listed = s.listed, []
    speaker = s.actor
    s.actor, s.addressed, s.counted = c, None, None
    match e:
        case "go":
            yield from say("to the")
            s.at("place", "theme")
            c.place = yield from place(s, c, new=1.0)
            s.at("end")
            yield from say(".")
        case "take":
            s.at("object")
            options = [option(0.4 if listed else 1.0, ["the", x.noun, "."], x) for x in visible if x.holder is None]
            options += [option(0.6, ["the", lx.ORDINALS[i], "one ."], x) for i, x in enumerate(listed)
                        if x.holder is None and x.where() == c.place]
            took = (yield from choose(options)).value()
            took.holder = c  # type: ignore[union-attr]
        case "drop":
            s.at("object")
            dropped = (yield from choose([option(1.0, ["the", x.noun, "."], x) for x in s.held(c)])).value()
            dropped.holder, dropped.place = None, c.place  # type: ignore[union-attr]
        case "give":
            s.at("object")
            given = (yield from choose([option(1.0, ["the", x.noun, "to"], x) for x in s.held(c)])).value()
            s.at("recipient")
            to = yield from other(s, c, s.others_here(c))
            s.mention(to.value())  # type: ignore[arg-type]
            given.holder = to.value()  # type: ignore[union-attr]
            s.at("end")
            yield from say(".")
        case "look":
            yield from say("at the")
            s.at("object")
            seen = (yield from choose([option(1.0, [x.noun, "."], x) for x in visible])).value()
            yield from say("it is")
            s.at("adjective", "kinds", "theme")
            first = yield from adjective(s, s.kind(seen))  # type: ignore[arg-type]
            s.at("adjective")
            yield from say("and")
            s.at("adjective", "kinds", "theme")
            yield from adjective(s, s.kind(seen), first)  # type: ignore[arg-type]
            s.at("end")
            yield from say(".")
        case "use":
            s.at("object", "kinds")
            used = said.posterior(2)
            yield from choose([option(p, ["the", x.noun, "."], x) for x, p in used.items()])  # type: ignore[union-attr]
        case "find" | "lose":
            s.at("count")
            if e == "find":
                amounts = [n for n in range(2, 6) if c.coins + n <= 20]
            else:
                amounts = [n for n in range(2, min(5, c.coins) + 1) if c.coins - n != 1]
            counted = [COUNTABLES if s.countable is None else one(s.countable)]
            said = yield from choose([option(1.0, [NUMBER[n], *counted, "."], n) for n in amounts])
            s.countable = said.words[1]
            c.coins += said.value() if e == "find" else -said.value()  # type: ignore[operator]
            s.counted = c
        case "say" | "answer":
            if e == "say":
                yield from say("to")
                s.at("recipient")
                to = (yield from other(s, c, s.others_here(c))).value()
                s.mention(to)  # type: ignore[arg-type]
            else:
                to = speaker
            s.at("quote")
            yield from say(', "')
            yield from utterance(s, c, to)  # type: ignore[arg-type]
            s.at("quote")
            yield from say('. "')
            s.addressed = to  # type: ignore[assignment]
        case "see":
            s.at("count")
            k = (yield from choose([option(1.0, [NUMBER[k], "things :"], k) for k in (2, 3, 4)])).value()
            for i in range(k):  # type: ignore[call-overload]
                s.at("list", "theme")
                said = yield from choose(
                    option(b, ["a", s.unused(s.themed("noun", z))], (None, z)) for z, b in s.theme.support()
                )
                s.theme.update(said, at=1)
                s.things.append(Thing(said.words[-1], c.place))
                s.listed.append(s.things[-1])
                s.at("list")
                yield from say("." if i == k - 1 else "and" if i == k - 2 else ",")
        case "act":
            s.at("adverb", "temperament")
            t_options = []
            for t, b in c.temperament.support():
                adverbs = ADVERBS[t] if s.config.temperament else PLAIN_ADVERBS
                t_options += [option(b * 0.3, ["."], (None, t)), option(b * 0.7, [adverbs, "."], (None, t))]
            c.temperament.update((yield from choose(t_options)), at=1)


def adjective(s: State, kind: int | None, other_than: str = "") -> Generator[Dist, str, str]:
    """An adjective that fits a thing of `kind`, favouring the theme's."""
    said = yield from choose(option(b, [adjectives(kind, z if s.config.theme else None).without([other_than])], (None, z))
                             for z, b in s.theme.support())
    s.theme.update(said, at=1)
    return said.words[0]


def utterance(s: State, speaker: Character, to: Character) -> Program:
    """Something true the speaker says to `to`: 'i have the kite', 'you have the cup', 'the
    ball is in the barn', 'we are in the barn', 'i am happy', 'i have five coins'."""
    kinds: list[list[Option]] = [
        [option(1.0, ["i have the", x.noun], ("have", None)) for x in s.held(speaker)],
        [option(1.0, ["you have the", x.noun], ("have", None)) for x in s.held(to)],
        [option(1.0, ["the", x.noun, "is in the", x.place], ("lies", None)) for x in s.things if x.holder is None],
        [option(1.0, ["we are in the", speaker.place], ("here", None))],
        [option(b, ["i am", adjectives(None, z if s.config.theme else None)], ("am", z)) for z, b in s.theme.support()],
    ]
    if s.countable:
        kinds.append([option(1.0, ["i have", NUMBER[speaker.coins] if speaker.coins else "no", s.countable], ("have", None))])
    kinds = [k for k in kinds if k]
    said = yield from choose(
        Option(o.weight / len(kinds) / (1 if o.value[0] == "am" else len(k)), o.words, o.value)  # type: ignore[index]
        for k in kinds for o in k
    )
    if said.value(0) == "am":
        s.theme.update(said, at=1)


def sample(program: Program, n_words: int, rng: random.Random) -> list[str]:
    """The first `n_words` words `program` produces, drawn with `rng`."""
    words: list[str] = []
    dist = next(program)
    while True:
        u = rng.random()
        for k, (p, part) in enumerate(dist):  # parts may be one Words object several times
            if u < p or k == len(dist) - 1:
                break
            u -= p
        words.append(part.sample(rng))
        if len(words) == n_words:
            return words
        dist = program.send(words[-1])
