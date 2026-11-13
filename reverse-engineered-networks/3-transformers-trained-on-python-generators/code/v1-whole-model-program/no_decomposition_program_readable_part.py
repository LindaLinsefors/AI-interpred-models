# READABLE PART of the program that replaces all four layer outputs of the 2-layer model trained
# on the simple generator (the attention and MLP outputs of layers 0 and 1), written by an agent
# team that saw the trained model and its training text, but not the generator and no
# decomposition.
#
# Score: joint KL 0.011 nats per token from the network (name positions 0.074, list counts 0.000,
# everything else 0.000), over 2.09M held-out positions.
#
# Note: the word lists below (FEMALE_NAMES, SINGULAR_NOUNS, PLACES, ...) match the generator's
# lexicon word for word and in the same order. The tokenizer assigns token ids class by class
# (see code/generator-v1/corpus.py), so these lists can be read off the vocabulary order.
#
# Omitted: after this ~230-line parser, the original 82 MB file lists ~20.8k feature keys and
# ~16M fitted float32 coefficients (ridge regression, plus a KL refit of the layer-1 MLP part)
# that map the features linearly onto the four 192-dimensional layer outputs. This file parses
# but does not run on its own.
"""A shared streaming grammar account of all four residual writers in the 2-layer model.

The vocabulary describes generated stories, not arbitrary prose. The concepts below
are: word classes and lexical deviations, clause/noun-phrase roles, person/number,
quotation, story age, list length/item number, and the characters introduced by
'named'. All four writes read these concepts ONCE, through the same additive table.

What the blocks do, at this level of approximation:
 * L0 attention retrieves local grammar cues and the last character/pronoun.
 * L0 MLP gives a strongly word-dependent, context-refined grammatical representation.
 * L1 attention retrieves the story cast, phrase governors, quotation/list context.
 * L1 MLP translates that state into the next-word constraints: subject versus object,
   agreement, possessives, dialogue versus narration, numbered-list continuation, and
   reuse of the story's character names. Story termination depends on story age.

The key nonlocal operation is COPY: average the eligible cast's name vectors ONLY
where a name is grammatically permitted (sentence/quote end, transitive object,
'both', and the second name after 'both X and'). Exclude X in that last case.
Two shared name-vector tables transfer this operation to every slot: each eligible
name contributes A[name]/n + B[name]/n**2, a first-order mixture and a cardinality
correction. A small slot-by-count baseline handles grammar, not name identity.
Background name memory is much weaker. The transformer
actually gives the header names unequal attention; this program approximates it
rather than claiming that uniform averaging is its exact internal algorithm.

Other long tables have explicit meanings: current/preceding/predicate word refinements,
class-by-grammatical-role interactions, person/number and quotation, list counters,
positions, and a decaying mixture of recent punctuation/gender/pronoun cues. No table
is indexed by an entire prefix, row, or uninterpreted latent activation.

Coefficients are jointly fitted residual vectors (4 blocks x 192 coordinates), not
copied model weights. Linear ridge fitting used training rows 10000:12048; a second
fit adjusted only L1 MLP's SAME concept coefficients for next-token KL while penalizing
its residual-write error. There is no learned hidden network in this program.
The coefficients are losslessly packed float32 for storage. Their KEYS are listed in
plain text, grouped by meaning. No runtime model, files, upstream values, or future
text are used.
"""
import numpy as np
import base64
import zlib
"""Shared concepts: vocabulary classes, grammatical roles, agreement, quotation,
list counters, story cast, and recency of structural/lexical cues.
This is a streaming parser; EOS resets story state but not row history."""


FEMALE_NAMES = frozenset('mia lily emma ava zoe ella ruby lucy grace chloe anna sara nora ivy rosa maya luna hazel iris jade kate lena alice clara daisy eva fiona gina hana isla julia kira laura mila nina olive paula rita sofia tara uma vera wendy yara zara amy bella cora dina elsa faye gwen holly ines jenny leah molly nell opal piper quinn rachel sadie tess ursula willa xena yasmin zelda abby beth cindy dora erin flora greta heidi irene mabel nadia olga pam stella tina una wren maria emily hannah sophie megan june fern pearl ruth edith'.split())
MALE_NAMES = frozenset('leo max sam ben tom jack finn noah liam owen eli jake luke adam alex carl dan evan fred gus hugo ian joel kai lars milo nate oscar paul ray seth theo umar vince will xavi yusuf zack abe bob cody dean eric frank glen hank ivan jim kurt lou mark ned otto pete rex stan ted vic walt zeke aaron brian colin david ethan felix george henry isaac james kevin louis martin nick oliver peter quentin robert simon tim victor wesley arthur bruno cyrus dylan edgar oswald harry igor jonas karl leon marco nolan omar perry rudy silas'.split())
SINGULAR_NOUNS = frozenset('dog cat bird frog duck horse cow pig goat rabbit mouse fox bear wolf lion tiger owl bee ant snail turtle snake whale shark crab puppy kitten monkey camel zebra donkey chicken goose hen lamb pony squirrel butterfly spider beetle worm parrot penguin hamster dolphin seal otter ball box book toy hat cup bag kite car boat bike doll drum bell key lamp chair table bed door window cake apple pie cookie sandwich egg flower tree leaf stone rock shell star cloud shoe sock coat dress ring coin map pen pencil brush cap spoon fork knife plate bowl basket bucket rope stick feather button candle clock mirror picture letter gift balloon blanket pillow umbrella wagon truck ladder hammer shovel puzzle robot rocket crayon marble ribbon scarf glove boot jar bottle pot pan teapot whistle flag tent sled trumpet violin guitar piano bench fence friend teacher farmer baker doctor king queen prince princess wizard giant knight pirate sailor child baby man woman dragon fairy ghost monster troll elf witch clown dwarf carrot banana orange grape pear peach berry lemon potato tomato muffin pizza noodle cherry melon onion nut seed biscuit pancake cupcake donut candy pumpkin acorn mushroom crown sword shield wand broom jewel treasure necklace bracelet wheel engine'.split())
PLURAL_NOUNS = frozenset('dogs cats birds frogs ducks horses cows pigs goats rabbits mice foxes bears wolves lions tigers owls bees ants snails turtles snakes whales sharks crabs puppies kittens monkeys camels zebras donkeys chickens geese hens lambs ponies squirrels butterflies spiders beetles worms parrots penguins hamsters dolphins seals otters balls boxes books toys hats cups bags kites cars boats bikes dolls drums bells keys lamps chairs tables beds doors windows cakes apples pies cookies sandwiches eggs flowers trees leaves stones rocks shells stars clouds shoes socks coats dresses rings coins maps pens pencils brushes caps spoons forks knives plates bowls baskets buckets ropes sticks feathers buttons candles clocks mirrors pictures letters gifts balloons blankets pillows umbrellas wagons trucks ladders hammers shovels puzzles robots rockets crayons marbles ribbons scarves gloves boots jars bottles pots pans teapots whistles flags tents sleds trumpets violins guitars pianos benches fences friends teachers farmers bakers doctors kings queens princes princesses wizards giants knights pirates sailors children babies men women dragons fairies ghosts monsters trolls elves witches clowns dwarves carrots bananas oranges grapes pears peaches berries lemons potatoes tomatoes muffins pizzas noodles cherries melons onions nuts seeds biscuits pancakes cupcakes donuts candies pumpkins acorns mushrooms crowns swords shields wands brooms jewels treasures necklaces bracelets wheels engines'.split())
PLACES = frozenset('park forest garden house school castle village city farm beach cave field kitchen room shop street bridge tower island mountain meadow jungle desert ocean valley barn library market palace nest yard zoo museum station hospital bakery lake pond river hill swamp orchard attic cellar playground classroom harbor lighthouse tunnel'.split())
ADJECTIVES = frozenset('big small red blue green yellow happy sad old new tall short little tiny huge soft hard warm cold hot wet dry fast slow loud quiet bright dark shiny dirty clean funny silly brave kind shy clever lazy busy sleepy hungry angry scared gentle strong weak pretty ugly fancy plain round square flat long thin fat heavy sweet sour salty fresh rotten broken golden silver purple pink white black brown gray striped spotted fluffy furry smooth rough sharp dull empty full rich poor young wild tame magic secret special strange odd lucky curious proud calm cheerful grumpy friendly lonely noisy sticky crunchy juicy tasty spicy icy sunny rainy windy snowy foggy muddy sandy rocky grassy leafy wooden tidy messy neat cozy chilly dusty crisp creamy bumpy wiggly jolly merry mighty gloomy eager polite rude wise foolish honest careful careless playful bouncy sparkly glowing hidden lost famous gigantic enormous ancient modern quick nimble clumsy graceful fierce timid bold cute lovely nasty awful wonderful great fine perfect terrible soggy frozen melted painted colorful pale deep shallow narrow wide tight loose sleek shaggy hairy spiky slimy crooked straight curly fuzzy grand humble mean nice cool keen sly swift stern vast'.split())
INTRANSITIVE_BASE = frozenset('run jump walk sleep sing dance laugh cry smile play swim sit stand wait rest shout hide fly fall hop skip spin wave nod yawn sneeze giggle wander rush hurry stop grow shine glow float sink roll slide crawl march skate bounce shiver blink frown sigh cheer clap dream wake travel return listen think bow kneel wobble tumble stumble dash race leap paddle drift howl bark purr chirp buzz roar quack squeak snore scream grin pout wiggle sway rise'.split())
INTRANSITIVE_S = frozenset('runs jumps walks sleeps sings dances laughs cries smiles plays swims sits stands waits rests shouts hides flies falls hops skips spins waves nods yawns sneezes giggles wanders rushes hurries stops grows shines glows floats sinks rolls slides crawls marches skates bounces shivers blinks frowns sighs cheers claps dreams wakes travels returns listens thinks bows kneels wobbles tumbles stumbles dashes races leaps paddles drifts howls barks purrs chirps buzzes roars quacks squeaks snores screams grins pouts wiggles sways rises'.split())
TRANSITIVE_BASE = frozenset('see like love find hold take want need help hug kick push pull carry throw catch eat drink make build paint draw read write open close wash fix break chase follow meet visit watch touch keep lose bring buy sell cook bake cut fill drop lift share give feed pet tickle kiss call greet thank miss remember forget hear smell taste grab squeeze shake borrow collect count fold hang pack pick plant pour protect save scare search steal tie trade trust wrap admire ask bump chew comb cover crush dig discover enjoy explore guard hit hunt inspect join lick mend mix notice obey pinch poke polish praise rescue rub scrub sniff stack stir stroke sweep teach tease toss warn weigh'.split())
TRANSITIVE_S = frozenset('sees likes loves finds holds takes wants needs helps hugs kicks pushes pulls carries throws catches eats drinks makes builds paints draws reads writes opens closes washes fixes breaks chases follows meets visits watches touches keeps loses brings buys sells cooks bakes cuts fills drops lifts shares gives feeds pets tickles kisses calls greets thanks misses remembers forgets hears smells tastes grabs squeezes shakes borrows collects counts folds hangs packs picks plants pours protects saves scares searches steals ties trades trusts wraps admires asks bumps chews combs covers crushes digs discovers enjoys explores guards hits hunts inspects joins licks mends mixes notices obeys pinches pokes polishes praises rescues rubs scrubs sniffs stacks stirs strokes sweeps teaches teases tosses warns weighs'.split())
ADVERBS = frozenset('quickly slowly happily sadly loudly quietly softly gently carefully suddenly often always again today together alone away outside everywhere badly bravely kindly proudly calmly eagerly politely wildly sleepily angrily cheerfully silently noisily gracefully clumsily lazily busily nervously patiently sweetly playfully'.split())
PREPOSITIONS = frozenset('in at near behind under beside inside across'.split())

WORD_CLASS = {}
for group, label in [(FEMALE_NAMES,'name_f'), (MALE_NAMES,'name_m'),
 (SINGULAR_NOUNS,'noun_s'), (PLURAL_NOUNS,'noun_p'), (PLACES,'place'),
 (ADJECTIVES,'adj'), (INTRANSITIVE_BASE,'vi'), (INTRANSITIVE_S,'vi_s'),
 (TRANSITIVE_BASE,'vt'), (TRANSITIVE_S,'vt_s'), (ADVERBS,'adv'), (PREPOSITIONS,'prep')]:
    for word in group: WORD_CLASS[word] = label
NUMBERS = dict(zip('two three four five six seven eight nine'.split(),range(2,10)))
for word in NUMBERS: WORD_CLASS[word]='number'
for word in '1 2 3 4 5 6 7 8 9'.split(): WORD_CLASS[word]='digit'
NAMES = FEMALE_NAMES | MALE_NAMES
NOUNS = SINGULAR_NOUNS | PLURAL_NOUNS
DETERMINERS = frozenset('a the some many his her their my your our'.split())
PRONOUNS = {'he':'male', 'she':'female', 'they':'plural', 'i':'first', 'you':'second', 'we':'plural'}
VERBS = frozenset(['vi','vi_s','vt','vt_s'])


def new_state():
    return dict(previous='START', previous2='START', previous_word='', phase='subject',
      np_role='subject', number='unknown', subject='unknown', quote=False, header=False,
      cast=[], joint=False, joint_names=0, adj_count=0, list_n=0, list_i=0,
      last_name='', last_pronoun='', last_verb='', last_cue='START', last_cue_position=0,
      last_boundary=0, story_start=0, quote_start=0, sentence_count=0,
      recent_names={}, cue_history={}, class_counts={}, last_seen={}, body_sentences=0, last_character_gender='unknown')


def text_features(state, word, position):
    if state is None: state=new_state()
    s=state
    c=WORD_CLASS.get(word,word)
    previous=s['previous']; previous2=s['previous2']; pw=s['previous_word']
    before=s['phase']; quote=s['quote']; header=s['header']
    if word in ('once','[EOS]'):
        s['body_sentences']=0; s['last_character_gender']='unknown'
    if c in ('name_m','name_f') or word in ('he','she'):
        s['last_character_gender']='female' if c=='name_f' or word=='she' else 'male'
    if (word in ('.','!') and not quote and not header) or (word=='"' and quote):
        s['body_sentences']+=1
    if word=='[EOS]':
        # The language model sees EOS as an ordinary token; retain row-history pools.
        s.update(phase='subject', np_role='subject', number='unknown', subject='unknown',
                 quote=False,header=False,cast=[],joint=False,joint_names=0,
                 adj_count=0,list_n=0,list_i=0,story_start=position+1,sentence_count=0)
    if word=='once':
        s['header']=True; s['story_start']=position; s['cast']=[]
    header=s['header']
    role=before
    if header: role='header'
    elif c in ('noun_s','noun_p','name_m','name_f','adj') or word in DETERMINERS:
        if before in ('subject','object','location','list'): role=before
        elif before in ('predicate','predicate_done'): role='predicate'
        elif word in DETERMINERS: role='object'
    if word in DETERMINERS:
        if not header:
            s['np_role']=role
            s['phase']=role
        if word=='a': s['number']='singular'
        elif word in ('some','many'): s['number']='plural'
        else: s['number']='unknown'
        s['adj_count']=0
    if word in PRONOUNS:
        s['subject']=PRONOUNS[word]; s['phase']='verb'; s['last_pronoun']=word
        s['joint']=False; s['joint_names']=0
    if word=='both':
        s['joint']=True; s['joint_names']=0; s['subject']='plural'; s['phase']='subject'
    if c in ('name_f','name_m'):
        if header and word not in s['cast']: s['cast'].append(word)
        if before=='subject' or s['joint'] or header:
            if not s['joint']: s['subject']='female' if c=='name_f' else 'male'
            else:
                s['joint_names']+=1
            s['phase']='verb'
        else: s['phase']='after_object'
        s['last_name']=word; s['recent_names'][word]=position
    if c in ('noun_s','noun_p'):
        if role=='subject': s['subject']='singular' if c=='noun_s' else 'plural'; s['phase']='verb'
        elif role=='location': s['phase']='after_location'
        elif role=='list': s['phase']='after_item'
        else: s['phase']='after_object'
    if c=='place': s['phase']='after_location'; role='location'
    if c=='adj':
        s['adj_count']+=1
        if role=='predicate': s['phase']='predicate_done'
    elif word not in DETERMINERS: s['adj_count']=0
    if c in VERBS or word in ('is','are','am','say','says'):
        s['last_verb']=word
        if c in ('vt','vt_s'): s['phase']='object'
        elif word in ('is','are','am'): s['phase']='predicate'
        else: s['phase']='after_verb'
        s['joint']=False
    if c=='prep': s['phase']='location'; s['np_role']='location'
    if word=='made': s['phase']='list_intro'; s['list_n']=0; s['list_i']=0
    if word in NUMBERS: s['list_n']=NUMBERS[word]
    if word==':' and s['list_n']: s['phase']='list'
    if c=='digit': s['list_i']=int(word); s['phase']='list'; s['np_role']='list'
    if word==',':
        if s['list_i']: s['phase']='list'
        else: s['phase']='quote_intro'
    if word=='and':
        if header: s['phase']='subject'
        elif s['joint'] and s['joint_names']==1: s['phase']='subject'
        else: s['phase']='verb'
    if word=='"':
        s['quote']=not s['quote']
        if s['quote']: s['quote_start']=position
        s['phase']='subject'; s['subject']='unknown'; s['joint']=False
        role='quote_close' if quote else 'quote_open'
    if word in ('.','!'):
        s['header']=False; s['phase']='subject'; s['joint']=False
        s['list_n']=0; s['list_i']=0; s['sentence_count']+=1
    # After-token semantic contexts, calculated once and shared by every write.
    q=int(s['quote']); hd=int(header)
    subject=s['subject']; number=s['number']; adj=min(s['adj_count'],2)
    li=s['list_i']; ln=s['list_n']
    f=[(('bias',),1.), (('word',word),1.), (('role',c,role,q,hd),1.),
       (('agreement',c,role,subject),1.), (('previous',c,previous),1.),
       (('previous2',c,previous2),1.), (('local_syntax',c,previous,role),1.),
       (('word_role',word,role),1.), (('quote',c,q),1.),
       (('noun_number',c,role,number),1.), (('adjectives',c,role,adj),1.),
       (('header',c,hd,len(s['cast'])),1.),
       (('row_position',c,min(position//32,15)),1.),
       (('sentence_position',c,min(position-s['last_boundary'],12)),1.),
       (('cue',c,s['last_cue'],min(position-s['last_cue_position'],8)),1.)]
    f.extend([(('story_progress',c,role,q,min(s['body_sentences'],6)),1.),
              (('antecedent_gender',c,q,s['last_character_gender']),1.)])
    # The repeated list template carries both the target length and item counter.
    if ln or li or before in ('list','after_item','list_intro'):
        f.extend([(('list',c,ln,li),1.), (('list_end',c,role,li==ln,li>0),1.)])
    # A story's cast is a sparse name-valued memory. Names share one lookup across all blocks.
    memory_gate = c if c in ('.','!','"',',','vt','vt_s','name_m','name_f','both','and') else 'background'
    if s['cast']:
        n=len(s['cast'])
        for name in s['cast']:
            f.append((('cast',memory_gate,name),1./n))
        f.append((('cast_gender',c,sum(x in FEMALE_NAMES for x in s['cast']),n),1.))
    if s['last_name']:
        f.append((('last_name',memory_gate,s['last_name']),1.))
    if s['last_pronoun']:
        f.append((('last_pronoun',c,s['last_pronoun']),1.))
    # Most lexical carryover is short-range: previous word and last predicate.
    if pw: f.append((('preceding_word',c,pw),1.))
    if s['last_verb'] and (c in ('adv','prep','adj') or word=='and'):
        f.append((('predicate_word',c,s['last_verb']),1.))
    # A decaying distribution of structural cues distinguishes early, mixed, and long contexts.
    for cue, value in list(s['cue_history'].items()):
        value*=0.96
        if value<0.001: del s['cue_history'][cue]
        else: s['cue_history'][cue]=value
    cue=c
    if c in ('name_m','name_f') or word in PRONOUNS or word in ('.','!','"','[EOS]','and','girl','boy'):
        s['cue_history'][cue]=s['cue_history'].get(cue,0.)+1.
        total=sum(s['cue_history'].values())
    else: total=sum(s['cue_history'].values())
    if total:
        for cue,value in s['cue_history'].items(): f.append((('cue_pool',c,cue),value/total))
    if word in ('.','!','"','[EOS]'):
        s['last_boundary']=position
    if c in ('vt','vt_s','vi','vi_s') or word in ('.','!','"','[EOS]','and','is','are','am','girl','boy'):
        s['last_cue']=c; s['last_cue_position']=position
    # Lexical identity is copied only at grammatical slots that permit a story character.
    copy_slot = None
    if word in ('.','!') and not q: copy_slot='sentence_boundary'
    elif word=='"' and quote: copy_slot='closing_quote'
    elif c in ('vt','vt_s') and not q: copy_slot='object_name'
    elif word=='both': copy_slot='first_joint_name'
    elif word=='and' and s['joint'] and s['joint_names']==1: copy_slot='second_joint_name'
    if copy_slot and s['cast']:
        available = s['cast']
        if copy_slot=='second_joint_name':
            available=[name for name in available if name != s['last_name']]
        n=len(available)
        f.append((('copy_base',copy_slot,n),1.))
        for name in available:
            f.append((('copy_identity',name),1./max(n,1)))
            f.append((('copy_identity2',name),1./max(n,1)**2))
    s['previous2']=previous; s['previous']=c; s['previous_word']=word
    return s,f

PREDICTS = ('L0.attn', 'L0.mlp', 'L1.attn', 'L1.mlp')
READS = ()

# ... KEYS (~20.8k feature keys), packed coefficients and the predict function omitted ...
