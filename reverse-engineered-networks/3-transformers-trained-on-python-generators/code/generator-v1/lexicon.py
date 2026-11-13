"""The generator's words, one class each. Every word belongs to exactly one class, so a word
alone says which choice the generator made."""

FEMALE_NAMES = """
mia lily emma ava zoe ella ruby lucy grace chloe anna sara nora ivy rosa maya luna hazel iris
jade kate lena alice clara daisy eva fiona gina hana isla julia kira laura mila nina olive
paula rita sofia tara uma vera wendy yara zara amy bella cora dina elsa faye gwen holly ines
jenny leah molly nell opal piper quinn rachel sadie tess ursula willa xena yasmin zelda abby
beth cindy dora erin flora greta heidi irene mabel nadia olga pam stella tina una wren maria
emily hannah sophie megan june fern pearl ruth edith
""".split()

MALE_NAMES = """
leo max sam ben tom jack finn noah liam owen eli jake luke adam alex carl dan evan fred gus
hugo ian joel kai lars milo nate oscar paul ray seth theo umar vince will xavi yusuf zack abe
bob cody dean eric frank glen hank ivan jim kurt lou mark ned otto pete rex stan ted vic walt
zeke aaron brian colin david ethan felix george henry isaac james kevin louis martin nick
oliver peter quentin robert simon tim victor wesley arthur bruno cyrus dylan edgar oswald
harry igor jonas karl leon marco nolan omar perry rudy silas
""".split()

# Singular nouns; plurals follow `plural`, with the exceptions in IRREGULAR_PLURALS.
NOUNS = """
dog cat bird frog duck horse cow pig goat rabbit mouse fox bear wolf lion tiger owl bee ant
snail turtle snake whale shark crab puppy kitten monkey camel zebra donkey chicken goose hen
lamb pony squirrel butterfly spider beetle worm parrot penguin hamster dolphin seal otter
ball box book toy hat cup bag kite car boat bike doll drum bell key lamp chair table bed door
window cake apple pie cookie sandwich egg flower tree leaf stone rock shell star cloud shoe
sock coat dress ring coin map pen pencil brush cap spoon fork knife plate bowl basket bucket
rope stick feather button candle clock mirror picture letter gift balloon blanket pillow
umbrella wagon truck ladder hammer shovel puzzle robot rocket crayon marble ribbon scarf glove
boot jar bottle pot pan teapot whistle flag tent sled trumpet violin guitar piano bench fence
friend teacher farmer baker doctor king queen prince princess wizard giant knight pirate
sailor child baby man woman dragon fairy ghost monster troll elf witch clown dwarf
carrot banana orange grape pear peach berry lemon potato tomato muffin pizza noodle cherry
melon onion nut seed biscuit pancake cupcake donut candy pumpkin acorn mushroom
crown sword shield wand broom jewel treasure necklace bracelet wheel engine
""".split()

IRREGULAR_PLURALS = {
    "mouse": "mice", "goose": "geese", "child": "children", "man": "men", "woman": "women",
    "wolf": "wolves", "leaf": "leaves", "knife": "knives", "scarf": "scarves", "elf": "elves",
    "dwarf": "dwarves", "potato": "potatoes", "tomato": "tomatoes",
}

PLACES = """
park forest garden house school castle village city farm beach cave field kitchen room shop
street bridge tower island mountain meadow jungle desert ocean valley barn library market
palace nest yard zoo museum station hospital bakery lake pond river hill swamp orchard attic
cellar playground classroom harbor lighthouse tunnel
""".split()

ADJECTIVES = """
big small red blue green yellow happy sad old new tall short little tiny huge soft hard warm
cold hot wet dry fast slow loud quiet bright dark shiny dirty clean funny silly brave kind shy
clever lazy busy sleepy hungry angry scared gentle strong weak pretty ugly fancy plain round
square flat long thin fat heavy sweet sour salty fresh rotten broken golden silver purple pink
white black brown gray striped spotted fluffy furry smooth rough sharp dull empty full rich
poor young wild tame magic secret special strange odd lucky curious proud calm cheerful grumpy
friendly lonely noisy sticky crunchy juicy tasty spicy icy sunny rainy windy snowy foggy muddy
sandy rocky grassy leafy wooden tidy messy neat cozy chilly dusty crisp creamy bumpy wiggly
jolly merry mighty gloomy eager polite rude wise foolish honest careful careless playful
bouncy sparkly glowing hidden lost famous gigantic enormous ancient modern quick nimble clumsy
graceful fierce timid bold cute lovely nasty awful wonderful great fine perfect terrible
soggy frozen melted painted colorful pale deep shallow narrow wide tight loose sleek shaggy
hairy spiky slimy crooked straight curly fuzzy grand humble mean nice cool keen sly swift
stern vast
""".split()

# Base forms; the third-person singular follows `third_person`.
INTRANSITIVE_VERBS = """
run jump walk sleep sing dance laugh cry smile play swim sit stand wait rest shout hide fly
fall hop skip spin wave nod yawn sneeze giggle wander rush hurry stop grow shine glow float
sink roll slide crawl march skate bounce shiver blink frown sigh cheer clap dream wake travel
return listen think bow kneel wobble tumble stumble dash race leap paddle drift howl bark purr
chirp buzz roar quack squeak snore scream grin pout wiggle sway rise
""".split()

TRANSITIVE_VERBS = """
see like love find hold take want need help hug kick push pull carry throw catch eat drink
make build paint draw read write open close wash fix break chase follow meet visit watch
touch keep lose bring buy sell cook bake cut fill drop lift share give feed pet tickle kiss
call greet thank miss remember forget hear smell taste grab squeeze shake borrow collect count
fold hang pack pick plant pour protect save scare search steal tie trade trust wrap admire ask
bump chew comb cover crush dig discover enjoy explore guard hit hunt inspect join lick mend mix
notice obey pinch poke polish praise rescue rub scrub sniff stack stir stroke sweep teach tease
toss warn weigh
""".split()

ADVERBS = """
quickly slowly happily sadly loudly quietly softly gently carefully suddenly often always
again today together alone away outside everywhere badly bravely kindly proudly calmly
eagerly politely wildly sleepily angrily cheerfully silently noisily gracefully clumsily
lazily busily nervously patiently sweetly playfully
""".split()

PREPOSITIONS = "in at near behind under beside inside across".split()

NUMBER_WORDS = "two three four five six seven eight nine".split()
DIGITS = [str(k) for k in range(1, 10)]


def plural(noun: str) -> str:
    if noun in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[noun]
    if noun.endswith(("s", "sh", "ch", "x", "z")):
        return noun + "es"
    if noun.endswith("y") and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def third_person(verb: str) -> str:
    if verb.endswith(("s", "sh", "ch", "x", "z", "o")):
        return verb + "es"
    if verb.endswith("y") and verb[-2] not in "aeiou":
        return verb[:-1] + "ies"
    return verb + "s"
