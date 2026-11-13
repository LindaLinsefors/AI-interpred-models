# Readable code of an accurate program for the harder generator's 6-layer model, written
# without a decomposition (joint KL 0.054 on held-out development text).
# Packed numeric tables and long word lists are elided: each b85-encoded table is replaced
# by the name ELIDED_PACKED_TABLE, and each long word list by `...` with a preview.
# The file parses but does not run. 94% of the original 4.85 MB were these tables.
ELIDED_PACKED_TABLE = None

"""Every block's write, from what a reader of these stories knows at each word.

The text is a synthetic story language: rows of packed stories, each from one of 16 "worlds"
(its own pool of names, places and objects), told in ~25 sentence templates: an introduction
("once there was a boy named N and a girl named N ."), placements ("N is in the L .", "there
is a O in the L ."), events (goes to / takes / drops / gives / looks at ("it is A and A .") /
sees K things / finds and loses K coins ("now N has K coins .") / intransitive verb with an
adverb / other transitive verbs), questions with their answers ("where is N ? she is in the
L .", "where is the O ? it is in the L .", "who has the O ? N has it .", "how many coins does
N have ? she has K coins ."), and dialogue ("N says to N , \" i am A / we are in the L / the O
is in the L / i have the O . \"", "N answers , ...").

The model computes, and this program reproduces, one shared reading of the story:
  1. Story (below) keeps the world state symbolically: the cast and their genders, where each
     person and object is, who holds what, how many coins each person has, the last list of
     things seen, which kinds of events the story and each person have used, the question
     being answered, the speaker and the addressee, and who "he"/"she" refer to.
  2. World infers which of the 16 worlds the story is set in from the distinct words seen.
  3. Featurizer names the concepts at each word (the CONCEPT FIELDS listed below): the word
     and its two predecessors, its grammatical slot, what the slot expects next (an open
     choice, where the world's prior applies, or a closed one among the story's own people,
     places and objects), the word the
     world state fixes next (a question's answer, the next character to be placed, the coin
     count, the held object after "the girl with the"), the words it allows (objects at the
     subject's place after "takes the", people at the same place after "says to", ...), the
     world, the subject's role and preconditions, and the story's characters, places, objects.
  4. Every concept value has one fitted signature (SIGNATURES, 384 numbers; the previous
     words and a few shortcut fields get theirs from smaller tables through a projection).
     The weighted sum of the active signatures is the reader's state at this word.
  5. A mixing layer (MIX_IN/MIX_OUT, 1024 units) adds to the state what a plain sum cannot
     express: effects of combinations of concepts. It is a fitted approximation, not a
     recovered circuit. Its strongest units, read off on held-out rows: at the verb slot,
     units that combine the subject's role (kinds of events it has done) with its
     preconditions and promote that role's verbs (finders: finds/plays/shivers/cheers;
     after seeing things: likes/helps/kicks; nothing possible: is/goes); in the
     introduction, units that cross the world with the gender ("a boy named" in worlds 4,
     13, 14 -> jack tom evan noah; "a girl named" -> lucy zoe ava emma); and many units at
     "N is in" that write which character is being placed (the next word there is always
     "the", so this is context the blocks carry forward, not a prediction).
  6. READOUT turns the state into all twelve blocks' writes (OFFSET + READOUT . state).
     Everything numeric was fitted on 20k training rows so that each block's write matches
     the model's and the sum of the twelve writes gives the model's next-word prediction
     (most of what the blocks write is that prediction and the context behind it).

Long tables: CLASSES (word -> word class), GENDER (name -> boy/girl), WORLD_* (how strongly
each word points to each world), CONCEPTS (the named concept values), SIGNATURES, MIX_* and
WRITE_MAP (fitted numbers, 5-8 bit with scales), OFFSET (each block's mean write).

CONCEPT FIELDS (Featurizer.step returns them):
d holds categorical values (strings) and multi-valued ones (lists, or (value, weight) pairs):
  w0, w1, w2   current / previous / two-back word ('' at row start)
  c0           class of current word (NAME/OBJ/LOC/ADJ/NUM/PLUR/VI/VT/ADV/ORD/PRON or the word)
  slot         grammar state after w0 (story3.Story.slot), e.g. '|S goes to the',
               'A.whereN|S is in the', 'END |S takes the O .', 'Q|we are in the', 'open'
  slot3        backoff: ctx + last 3 abstracted items
  ans          the next word the world state fixes (list of 0 or 1-2 words)
  cand         candidate next words the world state allows (a set; may be empty)
  evt          kinds of event sentences told so far in this story (multi-valued)
  world        posterior over the 16 story worlds, as (world, probability) pairs
  wx           the same, crossed with `expect` (the kind of word the slot expects next)
  expect       START/NAME/LOC/OBJ/ADJ/VERB/ADV/OTHER; NAMEc/LOCc/OBJc when the word must be one
               the story already holds
  addr         at a verb slot: whether the subject was just spoken to (yes@VERB/no@VERB); at a
               sentence start: who was last spoken to
  list         how many sentences ago someone saw a list of things (0-3, 4 = longer), with the
               expected word kind ('the first one' and the listed things stay likely for a while)
  clast        at a verb slot: the kind of the subject's own latest event (people tend to repeat
               or follow on: goes->goes, finds->finds, sees->takes, drops->takes)
  recent       asked where an object is: the place read most recently (another shortcut)
  nc           how many candidates the world state allows, with the expected word kind
  prov         how a question's answer is known (object held / free / never seen), '/part' when
               the story began before the row did
  count        while placing objects: how many are placed (at most 3); in the introduction: how
               many characters are named (at most 4)
  cevt         at a verb slot: the kinds of events the subject has done before (its role), and
               which intransitive verbs it used
  psubj        at a sentence start: the previous sentence's subject (who may act again)
  others       at a sentence start: the other characters
  naive        where the asked-about object was first seen (the model half-believes it is still there)
  evx, prex    evt and pre crossed with expect, at sentence starts and verb slots only
  castset      the story's characters; places: its locations; objset: its objects (multi-valued)
  holds        whether the subject holds any object ('yes'/'no')
  pre          the subject's preconditions for events, 4 letters: holds an object (H/h), someone
               else is at its place (N/n), a free object is at its place (O/o), has things (C/c)
  subj         resolved subject name of the current sentence ('' if none)
  ref          name the current word refers to (for names/he/she/him/her), else ''
  sg           gender of subj ('boy'/'girl'/'')
  about        the object the current sentence is about (last looked-at object in 'it is A and A')
  phase        'none' (no cast yet), 'locs' (cast locations being stated), 'objs' (objects being
               placed), 'story' (events)
  pos          position bucket: '0','1','2','3','4-7','8-15','16+'
"""
import base64
import lzma

import numpy as np

PREDICTS = tuple('lm/L%d.%s' % (i, k) for i in range(6) for k in ('attn', 'mlp'))
READS = ()

# ---------------------------------------------------------------- word knowledge
_BY_CLASS = ...  # elided literal, 20,260 chars; starts: {'NAME': ['aaron', 'abby', 'abe', 'abel', 'ada', 'adam', 'agnes', 'alan', 'albert', 'alex', 'alice', 'alma', '
CLASSES = {w: c for c, ws in _BY_CLASS.items() for w in ws}
_BOYS = ...  # elided literal, 1,324 chars; starts: ['aaron', 'abe', 'abel', 'adam', 'alan', 'albert', 'alex', 'arthur', 'barry', 'ben', 'benny', 'bob', 'boris', 
GENDER = {n: ('boy' if n in _BOYS else 'girl') for n in _BY_CLASS['NAME']}
WORLD_WORDS = ...  # elided literal, 19,122 chars; starts: ('pot', 'panics', 'fancy', 'awake', 'necklace', 'stretches', 'raisin', 'alex', 'factory', 'cabin', 'frolics', 
# log P(word appears in a story | world) * 8, rounded: [16 worlds, words]
WORLD_AFFINITY = np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='|i1').reshape((16, 1983))  # packed table elided: 17,038 chars
WORLD_PRIOR = np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='|i1').reshape((16,))  # packed table elided: 98 chars

# ---------------------------------------------------------------- the story reader
NUMW = ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen',
        'eighteen', 'nineteen', 'twenty']
NUMV = {w: i for i, w in enumerate(NUMW)}
ORD = {'first': 0, 'second': 1, 'third': 2, 'fourth': 3}
ABBR = {'NAME': 'N', 'OBJ': 'O', 'LOC': 'L', 'ADJ': 'A', 'NUM': 'K', 'PLUR': 'P', 'VI': 'V',
        'VT': 'T', 'ADV': 'D', 'ORD': 'R', 'PRON': 'S'}
PRON_OF = {'boy': 'he', 'girl': 'she'}
OBJ_PRON_OF = {'boy': 'him', 'girl': 'her'}


class World:
    """Which of the 16 story worlds we are in. Each world has its own cast, places and objects
    (its vocabulary); `affinity[word]` holds log P(word appears in a story | world) for the
    words that tell worlds apart. The posterior sums the affinities of the distinct words seen
    since the story began."""

    def __init__(self, words, affinity, prior):
        import numpy as np
        self.np = np
        self.affinity = {w: affinity[:, j].astype(np.float64) / 8 for j, w in enumerate(words)}
        self.prior = prior.astype(np.float64) / 8
        self.reset()

    def reset(self):
        self.score = self.prior.copy()
        self.seen = set()
        self._post = None

    def feed(self, w):
        a = self.affinity.get(w)
        if a is not None and w not in self.seen:
            self.seen.add(w)
            self.score = self.score + a
            self._post = None

    def posterior(self):
        if self._post is None:
            e = self.np.exp(self.score - self.score.max())
            self._post = e / e.sum()
        return self._post


def numword(k):
    return NUMW[k] if 0 <= k < len(NUMW) else 'no'


class Story:
    def __init__(self, cls, gender=None):
        self.cls = cls
        self.gtab = gender or {}  # every name's fixed gender ('boy'/'girl')
        self.reset()
        self.ctx = 'EOS'

    def reset(self):
        self.gender = {}      # name -> 'boy'/'girl'
        self.cast = []        # names in order of introduction
        self.loc = {}         # name -> location
        self.objloc = {}      # obj -> location
        self.objloc0 = {}     # obj -> the location it was first seen in
        self.holder = {}      # obj -> name or None
        self.objs = []        # objects in order of mention
        self.locs = []        # locations in order of mention
        self.adjs = []        # adjectives said of objects ("it is A and A")
        self.cadjs = []       # adjectives characters say of themselves ("i am A")
        self.count = {}       # name -> how many of the story's plural things
        self.plural = None    # the story's plural noun (coins, apples, ...)
        self.lastlist = []    # objects of the last "sees" list
        self.since_list = 9   # sentences since that list
        self.last_loc = None  # the place word read most recently
        self.last_of = {}     # gender -> last mentioned name of that gender
        self.sent = []        # words of the current sentence or utterance
        self.abs = []         # its abstracted form
        self.inquote = False
        self.speaker = None
        self.addressee = None
        self.last_looked = None
        self.subj = None      # resolved subject of the current sentence
        self.pending = None   # the question the current sentence answers
        self.question = None  # the question just asked
        self.ctx = ''         # '', 'Q' (inside quotes), 'A.<question kind>'
        self.started = False  # has any event sentence (not intro / 'there is') finished
        self.prev_subj = None # subject of the previous sentence
        self.events = set()   # kinds of event sentences told so far in this story
        self.acts = {}        # name -> kinds of events that person has done (their role)
        self.last_act = {}    # name -> the kind of that person's latest event
        self.nthere = 0       # objects placed by "there is a O in the L" (at most 3)
        self.by_ord = {}      # obj -> the list it was taken from as "the second one" etc.
        self.whole = False    # whether this story's beginning was read

    def c(self, w):
        return self.cls.get(w, w)

    def resolve(self, w):
        if w in ('he', 'him'):
            return self.last_of.get('boy')
        if w in ('she', 'her'):
            return self.last_of.get('girl')
        if self.c(w) == 'NAME':
            return w
        return None

    def mention(self, name):
        if name is not None and name in self.gender:
            self.last_of[self.gender[name]] = name

    def add_obj(self, o):
        if o not in self.objs:
            self.objs.append(o)
        self.holder.setdefault(o, None)

    def put(self, o, l):
        self.objloc[o] = l
        self.objloc0.setdefault(o, l)

    def colocate(self, name, o):
        """Someone acting on an object is where the object is."""
        if name is None or o is None:
            return
        if name in self.loc and o not in self.objloc:
            self.put(o, self.loc[name])
        elif o in self.objloc and name not in self.loc:
            self.loc[name] = self.objloc[o]

    def meet(self, a, b):
        """Two people talking or handing things over are in the same place."""
        if a is None or b is None:
            return
        if a in self.loc and b not in self.loc:
            self.loc[b] = self.loc[a]
        elif b in self.loc and a not in self.loc:
            self.loc[a] = self.loc[b]

    def add_loc(self, l):
        if l not in self.locs:
            self.locs.append(l)

    # ------------------------------------------------------------------ reading
    def feed(self, w):
        """Consume word w; update the world state and the slot."""
        if w == '[EOS]':
            self.reset()
            self.ctx, self.abs = 'EOS', []
            self.whole = True
            return
        if w == 'once' and not self.cast:
            self.whole = True
        if w == '"':
            if self.inquote:          # closing quote
                self.inquote = False
                self.ctx, self.sent, self.abs = 'close', [], []
            else:                     # opening quote: the clause before it names the speaker
                self.open_quote(self.sent)
                self.inquote = True
                self.ctx, self.sent, self.abs = 'open', [], []
            return
        s = self.sent
        if not s:                     # a sentence begins
            self.prev_subj = self.subj
            if self.inquote:
                self.ctx = 'Q'
            elif self.question is not None:
                self.pending, self.question = self.question, None
                self.ctx = 'A.' + self.pending[0]
            else:
                self.pending = None
                self.ctx = ''
            self.abs = []
        s.append(w)
        if self.c(w) == 'LOC':
            self.last_loc = w
        if w not in self.gender and w in self.gtab:   # a name we join mid-story
            self.gender[w] = self.gtab[w]
            self.cast.append(w)
        a = ABBR.get(self.c(w), w)
        n = len(s)
        if n == 1 and a in ('N', 'S'):
            a = 'S'
            self.subj = self.resolve(w)
        elif n == 1:
            self.subj = None
        if s[0] == 'the' and n == 5 and s[1] in ('boy', 'girl') and s[2] == 'with':
            self.subj = self.holder.get(w)
            self.abs = ['S(' + s[1] + ')']
            a = None
        elif s[0] == 'now' and n == 2:
            self.subj = self.resolve(w)
            a = 'S'
        elif a == 'K' and 'sees' in s:
            a = w                       # keep the list length
        if a is not None:
            self.abs.append(a)
        if w in ('.', '?'):
            self.since_list += 1
            self.finish(s)
            self.ctx = ('Qend ' if self.inquote else 'END ') + self.ctx
            self.sent = []
            return

    def slot(self):
        """The grammar state after the current word: where in which sentence we are."""
        if self.ctx in ('EOS', 'open', 'close'):
            return self.ctx
        a = self.abs
        if ':' in a:                  # inside a "sees K things : a O , a O and a O" list
            i = a.index(':')
            items = a[i + 1:]
            k = sum(1 for x in items if x == 'O')
            a = a[:i + 1] + ['#%d' % k] + items[-1:]
        return (self.ctx + '|' + ' '.join(a[-7:])).strip()

    def open_quote(self, s):
        name, i = self.subject_of(s)
        self.mention(name)
        rest = s[i:]
        if rest:
            self.events.add(rest[0])
            if name is not None:
                self.acts.setdefault(name, set()).add(rest[0])
                self.last_act[name] = rest[0]
        if rest and rest[0] == 'says' and len(rest) >= 3:
            to = self.resolve(rest[2])
            self.speaker, self.addressee = name, to
            self.meet(name, to)
            self.mention(to)
        elif rest and rest[0] == 'answers':
            self.speaker, self.addressee = name, self.speaker
            self.meet(name, self.addressee)

    def subject_of(self, s):
        if not s:
            return None, 0
        if s[0] == 'the' and len(s) >= 5 and s[1] in ('boy', 'girl') and s[2] == 'with':
            return self.holder.get(s[4]), 5
        if s[0] in ('he', 'she') or self.c(s[0]) == 'NAME':
            return self.resolve(s[0]), 1
        if s[0] == 'now' and len(s) > 1:
            return self.resolve(s[1]), 2
        return None, 0

    def finish(self, s):
        if self.inquote:
            sp, ad = self.speaker, self.addressee
            if s[:2] == ['i', 'am'] and len(s) >= 3 and s[2] not in self.cadjs:
                self.cadjs.append(s[2])
            elif s[:4] == ['we', 'are', 'in', 'the'] and len(s) >= 5:
                self.add_loc(s[4])
                for x in (sp, ad):
                    if x is not None:
                        self.loc[x] = s[4]
            elif s[:1] == ['the'] and s[2:5] == ['is', 'in', 'the'] and len(s) >= 6:
                self.add_obj(s[1])
                self.put(s[1], s[5])
            elif s[:3] in (['i', 'have', 'the'], ['you', 'have', 'the']) and len(s) >= 4:
                self.add_obj(s[3])
                self.holder[s[3]] = sp if s[0] == 'i' else ad
            elif s[:2] == ['i', 'have'] and len(s) >= 4 and s[2] in NUMV and sp is not None:
                self.count[sp] = NUMV[s[2]]
                self.plural = s[3]
            return
        if s[-1] == '?':
            self.started = True
            self.events.add(s[0])
            if s[0] == 'where' and len(s) >= 4 and s[2] != 'the':
                self.question = ('whereN', s[2])
            elif s[0] == 'where' and len(s) >= 5:
                self.question = ('whereO', s[3])
            elif s[0] == 'who' and len(s) >= 5:
                self.question = ('who', s[3])
            elif s[0] == 'how' and len(s) >= 6:
                self.question = ('howmany', s[4])
            if self.question and self.question[1] in self.gender:
                self.mention(self.question[1])
            return
        if s[0] == 'once' or (s[0] == 'there' and len(s) > 1 and s[1] == 'was'):
            for i, w in enumerate(s):
                if w == 'named' and i + 1 < len(s):
                    n, g = s[i + 1], s[i - 1]
                    self.gender[n] = g
                    if n not in self.cast:
                        self.cast.append(n)
                    self.last_of[g] = n
            return
        if s[0] == 'there' and len(s) >= 7 and s[1] == 'is':
            o, l = s[3], s[6]
            self.nthere += 1
            self.add_obj(o)
            self.put(o, l)
            self.add_loc(l)
            return
        if s[0] in ('where', 'who', 'how', 'now'):
            self.events.add(s[0])
        if s[0] in ('nobody',):
            return
        p = self.pending
        if p is not None and p[0] == 'whereO' and len(s) >= 5 and s[:4] == ['it', 'is', 'in', 'the']:
            self.put(p[1], s[4])
            self.add_loc(s[4])
        if p is not None and p[0] == 'who' and len(s) >= 2:
            self.holder[p[1]] = self.resolve(s[0]) if s[0] != 'nobody' else None
        if s[0] == 'it':
            if len(s) >= 5 and s[2] != 'in':
                for x in (s[2], s[4]):
                    if x not in self.adjs:
                        self.adjs.append(x)
            return
        name, i = self.subject_of(s)
        self.mention(name)
        rest = s[i:]
        if not rest:
            return
        v = rest[0]
        cv = self.c(v)
        if v != 'is':
            self.started = True
            self.events.add(v if cv not in ('VI', 'VT') else cv)
            if name is not None and s[0] != 'now':
                a = self.acts.setdefault(name, set())
                a.add(v if cv not in ('VI', 'VT') else cv)
                if cv == 'VI':
                    a.add(v)
                self.last_act[name] = v if cv not in ('VI', 'VT') else cv
        if v in ('is', 'goes') and len(rest) >= 4:
            self.add_loc(rest[3])
            if name is not None:
                self.loc[name] = rest[3]
                for o, h in self.holder.items():
                    if h == name:
                        self.put(o, rest[3])
        elif v == 'takes' and len(rest) >= 3:
            o = rest[2]
            if o in ORD:
                k = ORD[o]
                o = self.lastlist[k] if k < len(self.lastlist) else None
                if o is not None:
                    self.by_ord[o] = list(self.lastlist)
            if o is not None:
                self.add_obj(o)
                self.colocate(name, o)
                self.holder[o] = name
                if name in self.loc:
                    self.put(o, self.loc[name])
        elif v == 'drops' and len(rest) >= 3:
            self.add_obj(rest[2])
            self.colocate(name, rest[2])
            self.holder[rest[2]] = None
        elif v == 'gives' and len(rest) >= 5:
            o = rest[2]
            to = self.resolve(rest[4])
            self.add_obj(o)
            self.meet(name, to)
            self.colocate(name, o)
            self.holder[o] = to
            self.mention(to)
            if to in self.loc:
                self.put(o, self.loc[to])
        elif v == 'looks' and len(rest) >= 4:
            self.last_looked = rest[3]
            self.add_obj(rest[3])
            self.colocate(name, rest[3])
        elif v in ('finds', 'loses') and len(rest) >= 3:
            k = NUMV.get(rest[1], 0)
            self.plural = rest[2]
            if name is not None:
                self.count[name] = self.count.get(name, 0) + (k if v == 'finds' else -k)
        elif v == 'has' and len(rest) >= 3 and rest[1] != 'it':
            self.plural = rest[2]
            if name is not None:
                self.count[name] = NUMV.get(rest[1], 0)
        elif v == 'sees':
            self.lastlist = [rest[j + 1] for j in range(len(rest) - 1) if rest[j] == 'a']
            self.since_list = 0
            for o in self.lastlist:
                self.add_obj(o)
                if name in self.loc and o not in self.objloc:
                    self.put(o, self.loc[name])
        elif len(rest) >= 3 and rest[1] == 'the':
            self.add_obj(rest[2])
            self.colocate(name, rest[2])

    # ------------------------------------------------------------ predictions
    def defn(self, a):
        self.definite = bool(a)
        return a

    def naive(self):
        """A shortcut answer the model also leans on: where an object was first seen."""
        s = self.sent
        p = self.pending
        if p is not None and p[0] == 'whereO' and s == ['it', 'is', 'in', 'the']:
            l = self.objloc0.get(p[1])
            return [l] if l else []
        if self.inquote and len(s) == 5 and s[0] == 'the' and s[2:5] == ['is', 'in', 'the']:
            l = self.objloc0.get(s[1])
            return [l] if l else []
        # an object taken as "the second one": the model is unsure which item of the list it was
        a = self.answer() if s and s[-1] == 'the' else []
        out = []
        if len(s) >= 2 and s[-2:] == ['takes', 'the'] and self.since_list <= 3:
            # things just seen look takeable even after the seer has moved on
            out = [o for o in self.lastlist if self.holder.get(o) is None and o not in a]
        for o in a:
            for x in self.by_ord.get(o, ()):
                if x not in a and x not in out:
                    out.append(x)
        return out

    def recent_place(self):
        """Asked where an object is, the model also leans to the place read most recently."""
        s = self.sent
        p = self.pending
        if (p is not None and p[0] == 'whereO' and s == ['it', 'is', 'in', 'the']) or (
                self.inquote and len(s) == 5 and s[0] == 'the' and s[2:5] == ['is', 'in', 'the']):
            return self.last_loc or ''
        return ''

    def provenance(self):
        """How the answer to the pending question is known: from events read, or by default."""
        s = self.sent
        q = self.question
        if not s and q is not None and q[0] == 'who':
            o = q[1]
            if self.holder.get(o) is not None:
                return 'who:held'
            return 'who:free' if o in self.objs else 'who:unseen'
        p = self.pending
        if p is not None and p[0] == 'whereO' and s == ['it', 'is', 'in', 'the']:
            return 'whereO:' + ('held' if self.holder.get(p[1]) else 'free' if p[1] in self.objs else 'unseen')
        return ''

    def answer2(self):
        """(definite, candidates): the word the world state fixes next, or a set it allows."""
        self.definite = False
        a = self.answer()
        return (a, []) if self.definite else ([], a)

    def answer(self):
        """World-state candidates for the next word: a list of words, possibly empty."""
        s = self.sent
        n = len(s)
        q = self.question
        if n == 0 and q is not None and not self.inquote:
            # a question was just asked: its answer begins here
            kind, arg = q
            if kind in ('whereN', 'howmany'):
                g = self.gender.get(arg)
                return self.defn([PRON_OF[g]] if g else [])
            if kind == 'whereO':
                return self.defn(['it'])
            if kind == 'who':
                h = self.holder.get(arg)
                return self.defn([h if h else 'nobody'])
            return []
        if n == 0 and self.ctx == 'END ' and self.cast and not self.started:
            # the introduction: each character's place in the order they were named
            for x in self.cast:
                if x not in self.loc:
                    return self.defn([x])
            if self.abs[:2] == ['S', 'is']:
                return self.defn(['there'])
        if n == 0:
            if self.ctx in ('END ', 'EOS', 'close', 'END A.whereN', 'END A.whereO',
                            'END A.who', 'END A.howmany') and self.cast:
                return list(self.cast)
            return []
        last = s[-1]
        if self.inquote:
            sp = self.speaker
            if s == ['we', 'are', 'in', 'the']:
                return self.defn([self.loc[sp]] if sp in self.loc else [])
            if n == 5 and s[0] == 'the' and s[2:5] == ['is', 'in', 'the']:
                return self.defn([self.objloc[s[1]]] if s[1] in self.objloc else [])
            if s == ['i', 'have', 'the']:
                return [o for o, h in self.holder.items() if h is not None and h == sp]
            if s == ['you', 'have', 'the']:
                return [o for o, h in self.holder.items() if h is not None and h == self.addressee]
            if s == ['i', 'have']:
                return self.defn([numword(self.count.get(sp, 0))])
            if n == 3 and s[:2] == ['i', 'have'] and last in NUMV:
                return self.defn([self.plural] if self.plural else [])
            if s == ['i', 'am']:
                return list(self.cadjs)
            if s == ['the']:
                return list(self.objs)
            return []
        p = self.pending
        if p is not None:
            kind, arg = p
            if kind == 'whereN':
                if n == 4:
                    return self.defn([self.loc[arg]] if arg in self.loc else [])
            elif kind == 'whereO':
                if n == 4:
                    return self.defn([self.objloc[arg]] if arg in self.objloc else [])
            elif kind == 'who':
                if n == 0:
                    pass
            elif kind == 'howmany':
                if n == 2:
                    return self.defn([numword(self.count.get(arg, 0))])
                if n == 3:
                    return self.defn([self.plural] if self.plural else [])
            return []
        if s == ['how', 'many']:
            return self.defn([self.plural] if self.plural else [])
        if s == ['where', 'is'] or (n == 4 and s[0] == 'how' and last == 'does'):
            return list(self.cast)
        if s == ['now'] and self.prev_subj in self.gender:
            return [self.prev_subj, PRON_OF[self.gender[self.prev_subj]]]
        if n >= 3 and last == 'the' and self.c(s[-2]) == 'VT':
            here = self.loc.get(self.subj)
            return [o for o in self.objs if here is None or self.objloc.get(o) in (None, here)]
        if s[0] == 'now' and n == 3:
            return self.defn([numword(self.count.get(self.subj, 0))])
        if s[0] == 'now' and n == 4:
            return self.defn([self.plural] if self.plural else [])
        if last in NUMV and n >= 2 and s[-2] in ('finds', 'loses'):
            return self.defn([self.plural] if self.plural else [])
        if s[0] == 'the' and n == 4 and s[1] in ('boy', 'girl') and s[2] == 'with':
            g = s[1]
            return [o for o, h in self.holder.items() if h is not None and self.gender.get(h) == g]
        if s[0] in ('where', 'who') and last == 'the':
            return list(self.objs)
        if s[0] == 'there' and n == 6 and s[1] == 'is':
            return list(self.locs)
        if last == 'the' and n >= 2 and s[-2] == 'takes':
            here = self.loc.get(self.subj)
            return [o for o in self.objs if self.holder.get(o) is None
                    and (here is None or self.objloc.get(o) in (None, here))]
        if last == 'the' and n >= 2 and s[-2] in ('drops', 'gives'):
            return [o for o, h in self.holder.items() if h is not None and h == self.subj]
        if last == 'the' and n >= 3 and s[-2] == 'at' and s[-3] == 'looks':
            here = self.loc.get(self.subj)
            return [o for o in self.objs if here is None or self.objloc.get(o) in (None, here)]
        if last == 'to' and n >= 2 and s[-2] == 'says':
            here = self.loc.get(self.subj)
            near = [x for x in self.cast if x != self.subj and (here is None or self.loc.get(x) == here)]
            return near + sorted({OBJ_PRON_OF[self.gender[x]] for x in near if x in self.gender})
        if last == 'to' and n >= 4 and s[-4] == 'gives':
            return [x for x in self.cast if x != self.subj]
        if last == 'the' and n >= 3 and s[-2] == 'to' and s[-3] == 'goes':
            return [l for l in self.locs if l != self.loc.get(self.subj)]
        if last == 'the' and n >= 3 and s[-3] == 'is' and s[-2] == 'in':
            return list(self.locs)
        if s[0] == 'it' and n in (2, 4):
            return list(self.adjs)
        if n == 1 and s[0] in ('he', 'she') or (n == 1 and self.c(s[0]) == 'NAME'):
            return []
        return []


# ---------------------------------------------------------------- the concepts
def pos_bucket(t):
    if t < 4:
        return str(t)
    if t < 8:
        return '4-7'
    if t < 16:
        return '8-15'
    return '16+'


def expect(slot):
    """The kind of word the grammar expects next at this slot; a trailing 'c' marks a closed
    choice (the word must be one the story already holds: a person, place or object in it)."""
    head, _, body = slot.partition('|')
    items = body.split()
    last = items[-1] if items else ''
    prev = items[-2] if len(items) > 1 else ''
    if slot in ('EOS', 'close') or head.startswith('END') or head.startswith('Qend'):
        return 'START'
    if head.startswith('A.'):
        return 'OTHER'              # an answer sentence: its words are fixed
    if (last == 'to' and prev == 'says') or (last == 'to' and 'gives' in items):
        return 'NAMEc'
    if last == 'named':
        return 'NAME'
    if 'named' in items or items == ['the']:
        return 'OTHER'              # 'a boy/girl', 'the boy/girl'
    if last == 'the' and prev == 'in' and (items[:2] == ['there', 'is'] or head == 'Q'):
        return 'LOCc'
    if last == 'the' and prev in ('in', 'to'):
        return 'LOC'
    if last == 'the':
        return 'OBJc'
    if last == 'a' and ('is' in items or ':' in items or ',' in items or 'and' in items):
        return 'OBJ'
    if last in ('is', 'and', 'am') and (items[:1] == ['it'] or 'am' in items):
        return 'ADJ'
    if last.startswith('S'):
        return 'VERB'
    if last == 'V':
        return 'ADV'
    return 'OTHER'


def gender_table():
    return GENDER


def world_table():
    return WORLD_WORDS, WORLD_AFFINITY, WORLD_PRIOR


class Featurizer:
    def __init__(self, cls):
        self.cls = cls
        self.st = Story(cls, gender_table())
        self.world = World(*world_table())
        self.w1 = ''
        self.w2 = ''
        self.t = 0

    def step(self, w):
        st = self.st
        ref = ''
        if w in ('he', 'she', 'him', 'her') or self.cls.get(w) == 'NAME':
            ref = st.resolve(w) or ''
        st.feed(w)
        if w == '[EOS]':
            self.world.reset()
        else:
            self.world.feed(w)
        slot = st.slot()
        head, _, body = slot.partition('|')
        slot3 = head + '|' + ' '.join(body.split()[-3:]) if body else slot
        ans, cand = st.answer2()
        post = self.world.posterior()
        ex = expect(slot)
        subj = st.subj
        holds = any(h == subj for h in st.holder.values() if h)
        sl = st.loc.get(subj)
        near = sl is not None and any(st.loc.get(x) == sl for x in st.cast if x != subj)
        here = sl is not None and any(st.objloc.get(o) == sl and st.holder.get(o) is None
                                      for o in st.objs)
        pre = ''.join(['H' if holds else 'h', 'N' if near else 'n', 'O' if here else 'o',
                       'C' if st.count.get(subj, 0) > 0 else 'c'])
        about = ''
        sent = st.sent
        if sent[:1] == ['it'] and not (len(sent) > 2 and sent[2] == 'in'):
            about = st.last_looked or ''
        else:
            for x in reversed(sent):
                if self.cls.get(x) == 'OBJ':
                    about = x
                    break
        if not st.cast:
            phase = 'none'
        elif any(n not in st.loc for n in st.cast):
            phase = 'locs'
        elif not st.started:
            phase = 'objs'
        else:
            phase = 'story'
        d = {'w0': w, 'w1': self.w1, 'w2': self.w2, 'c0': self.cls.get(w, w),
             'slot': slot, 'slot3': slot3, 'ans': ans, 'cand': cand,
             'evt': sorted(st.events),
             'world': [(str(k), float(p)) for k, p in enumerate(post) if p > 0.01],
             'wx': [(str(k) + '@' + ex, float(p)) for k, p in enumerate(post) if p > 0.01]
             if ex != 'OTHER' else [],
             'expect': ex,
             'naive': st.naive(),
             'addr': ('yes@' if subj is not None and subj == st.addressee else 'no@') + ex
             if ex == 'VERB' else (st.addressee or '') if ex == 'START' and st.addressee else '',
             'list': 'L%d@%s' % (min(st.since_list, 4), ex) if ex in ('OBJc', 'VERB', 'START') else '',
             'recent': st.recent_place(),
             'clast': st.last_act.get(subj, 'none') + '@VERB' if ex == 'VERB' else '',
             'nc': '%d@%s' % (min(len(cand), 4), ex) if cand else '',
             'prov': st.provenance() + ('' if st.whole else '/part'),
             'count': ('there%d' % st.nthere if phase == 'objs' else
                       'cast%d' % sent.count('named') if sent[:1] == ['once'] else ''),
             'cevt': sorted(st.acts.get(subj, ())) if ex == 'VERB' else [],
             'psubj': (subj or '') if ex == 'START' else '',
             'others': [x for x in st.cast if x != subj] if ex == 'START' else [],
             'evx': [e + '@' + ex for e in sorted(st.events)] if ex in ('START', 'VERB') else [],
             'prex': pre + '@' + ex if ex in ('START', 'VERB') else '',
             'castset': list(st.cast), 'places': list(st.locs), 'objset': list(st.objs),
             'holds': 'yes' if holds else 'no',
             'pre': pre,
             'subj': st.subj or '', 'sg': st.gender.get(st.subj, ''), 'ref': ref,
             'about': about, 'phase': phase, 'pos': pos_bucket(self.t)}
        self.w2, self.w1 = self.w1, w
        self.t += 1
        return d



# ---------------------------------------------------------------- fitted writes
FIELDS = ('w0', 'w1', 'w2', 'c0', 'slot', 'slot3', 'subj', 'sg', 'ref', 'about', 'phase', 'pos', 'holds', 'pre', 'expect', 'prex', 'psubj', 'count', 'nc', 'prov', 'addr', 'list', 'recent', 'clast')      # one value each
MULTI_FIELDS = ('ans', 'cand', 'naive', 'others', 'evt', 'cevt', 'castset', 'places', 'objset', 'world', 'wx', 'evx')      # several values, weighted
UNIT_WEIGHT = ('evt', 'evx', 'cevt')      # their values count 1 each (others 1/n)
MAX_TERMS = 80
CONCEPTS = ...  # elided literal, 192,004 chars; starts: ('BIAS', 'about=', 'about=acorn', 'about=alligator', 'about=almond', 'about=anchor', 'about=ant', 'about=apple
# each concept value's signature (384 numbers); the word fields listed in SMALL_TABLES get theirs
# from a smaller per-word table through a projection (those roles need fewer distinctions)
SIGNATURES = np.zeros((13443, 384), np.float32)
SIGNATURES[np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<i4').reshape((7838,))] = np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='|i1').reshape((7838, 384)).astype(np.float32) * np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<f4').reshape((7838,))[:, None]  # packed table elided: 2,086,624 chars
SMALL_TABLES = ('w1', 'w2', 'naive', 'recent', 'addr', 'psubj')
SIGNATURES[np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<i4').reshape((2137,))] = (np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='|i1').reshape((2137, 64)).astype(np.float32) * np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<f4').reshape((2137,))[:, None]) @ (np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='|i1').reshape((64, 384)).astype(np.float32) * np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<f4').reshape((64,))[:, None])   # w1  # packed table elided: 162,115 chars
SIGNATURES[np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<i4').reshape((2137,))] = (np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='|i1').reshape((2137, 64)).astype(np.float32) * np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<f4').reshape((2137,))[:, None]) @ (np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='|i1').reshape((64, 384)).astype(np.float32) * np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<f4').reshape((64,))[:, None])   # w2  # packed table elided: 162,820 chars
SIGNATURES[np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<i4').reshape((565,))] = (np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='|i1').reshape((565, 48)).astype(np.float32) * np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<f4').reshape((565,))[:, None]) @ (np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='|i1').reshape((48, 384)).astype(np.float32) * np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<f4').reshape((48,))[:, None])   # naive  # packed table elided: 50,080 chars
SIGNATURES[np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<i4').reshape((158,))] = (np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='|i1').reshape((158, 32)).astype(np.float32) * np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<f4').reshape((158,))[:, None]) @ (np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='|i1').reshape((32, 384)).astype(np.float32) * np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<f4').reshape((32,))[:, None])   # recent  # packed table elided: 20,710 chars
SIGNATURES[np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<i4').reshape((305,))] = (np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='|i1').reshape((305, 32)).astype(np.float32) * np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<f4').reshape((305,))[:, None]) @ (np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='|i1').reshape((32, 384)).astype(np.float32) * np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<f4').reshape((32,))[:, None])   # addr  # packed table elided: 25,145 chars
SIGNATURES[np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<i4').reshape((303,))] = (np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='|i1').reshape((303, 32)).astype(np.float32) * np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<f4').reshape((303,))[:, None]) @ (np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='|i1').reshape((32, 384)).astype(np.float32) * np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<f4').reshape((32,))[:, None])   # psubj  # packed table elided: 25,290 chars
# the mixing layer: 1024 units that respond to combinations of concepts
MIX_IN = np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='|i1').reshape((1024, 384)).astype(np.float32) * np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<f4').reshape((1024,))[:, None]  # packed table elided: 335,991 chars
MIX_IN_BIAS = np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<f4').reshape((1024,))  # packed table elided: 4,943 chars
MIX_OUT = np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='|i1').reshape((384, 1024)).astype(np.float32) * np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<f4').reshape((384,))[:, None]  # packed table elided: 309,781 chars
MIX_OUT_BIAS = np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<f4').reshape((384,))  # packed table elided: 1,968 chars
WRITE_MAP = (np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='|i1').reshape((384, 4608)).astype(np.float32).reshape(384, 12, 384) * np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<f4').reshape((12,))[None, :, None]).reshape(384, 4608)  # packed table elided: 1,347,236 chars
OFFSET = np.frombuffer(lzma.decompress(base64.b85decode(ELIDED_PACKED_TABLE)),dtype='<f4').reshape((4608,))  # packed table elided: 21,588 chars
READOUT = np.ascontiguousarray(WRITE_MAP.T)      # [12 blocks * 384, signature]
CONCEPT_INDEX = {name: i for i, name in enumerate(CONCEPTS)}


def active_concepts(facts):
    """The concept values present at this word, with their weights."""
    terms = [('BIAS', 1.0)] + [(f + '=' + facts[f], 1.0) for f in FIELDS]
    for f in MULTI_FIELDS:
        values = facts[f]
        if values:
            if isinstance(values[0], tuple):          # (value, probability) pairs
                terms += [(f + '=' + v, w) for v, w in values]
            else:
                w = 1.0 if f in UNIT_WEIGHT else 1.0 / len(values)
                terms += [(f + '=' + v, w) for v in values]
    terms = terms[:MAX_TERMS]
    index = [CONCEPT_INDEX.get(k, CONCEPT_INDEX.get(k.split('=', 1)[0] + '=<rare>', 0)) for k, _ in terms]
    return index, np.array([w for _, w in terms], np.float32)


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def step(carry, prefix, upstream):
    if carry is None:
        carry = Featurizer(CLASSES)
    facts = carry.step(prefix.token_strings[prefix.position].strip())
    index, weights = active_concepts(facts)
    # (einsum keeps every product on one thread: numpy's BLAS would spread these small
    # matrix-vector products over all cores and only burn CPU time)
    state = np.einsum('ji,j->i', SIGNATURES[index], weights)          # the reader's state here
    hidden = gelu(np.einsum('ij,j->i', MIX_IN, state) + MIX_IN_BIAS)
    state = state + np.einsum('ij,j->i', MIX_OUT, hidden) + MIX_OUT_BIAS
    writes = (OFFSET + np.einsum('ij,j->i', READOUT, state)).reshape(12, 384)
    return carry, {subject: writes[i] for i, subject in enumerate(PREDICTS)}
