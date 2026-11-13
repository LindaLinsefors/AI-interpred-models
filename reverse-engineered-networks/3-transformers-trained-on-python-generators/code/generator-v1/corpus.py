"""The generator's vocabulary, its word-level tokenizer and its datasets.

Every row is cut from a fresh `stories.stream()`, so it starts at a story's first word and
its prefix fixes the generator's state: `stories.replay` on a row gives the exact next-word
distributions. Rows are `seq_len` words, stories joined by [EOS], the last one cut off.

Run from this directory with the param_decomp environment:
  python corpus.py tokenizer <dir>
  python corpus.py generate <out_dir> --tokenizer_dir <dir> --n_rows N --seed S
  python corpus.py finish <out_dir> <dataset_dir> --producer_revision $(git rev-parse HEAD)
"""

import random
from multiprocessing import Pool
from pathlib import Path

import fire
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

import lexicon as lx
import stories as st

UNK = "[UNK]"
FUNCTION_WORDS = """
once there was a girl boy named and . , " : ! the some many he she they both his her their
i you we my your our is are am says say made list of things
""".split()
CONTENT_CLASSES = [
    lx.FEMALE_NAMES,
    lx.MALE_NAMES,
    lx.NOUNS,
    [lx.plural(n) for n in lx.NOUNS],
    lx.PLACES,
    lx.ADJECTIVES,
    lx.INTRANSITIVE_VERBS,
    [lx.third_person(v) for v in lx.INTRANSITIVE_VERBS],
    lx.TRANSITIVE_VERBS,
    [lx.third_person(v) for v in lx.TRANSITIVE_VERBS],
    lx.ADVERBS,
    lx.PREPOSITIONS,
    lx.NUMBER_WORDS,
    lx.DIGITS,
]
VOCAB = [UNK, st.EOS, *FUNCTION_WORDS, *(w for words in CONTENT_CLASSES for w in words)]
assert len(set(VOCAB)) == len(VOCAB), "a word in two classes"
TOKEN_ID = {w: i for i, w in enumerate(VOCAB)}
PREPROCESSING_NAME = "task2394-genstories-v1"


def tokenizer(out_dir: str) -> None:
    """A word-level tokenizer over `VOCAB`: words split on spaces, decoded as WordPiece
    decodes (punctuation joined to the word before), which explain2's vocabulary reads."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel(TOKEN_ID, unk_token=UNK))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.decoder = decoders.WordPiece()
    fast = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token=UNK, eos_token=st.EOS)
    fast.save_pretrained(out_dir)
    text = " ".join(st.sample(st.stream(), 200, random.Random(0)))
    assert fast.decode(fast(text)["input_ids"]).replace(" ", "") == text.replace(" ", "")


def rows(n_rows: int, seq_len: int, seed: int) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """`n_rows` rows of token ids, each from a fresh stream, and each token's story within
    its row (a story's ids run through the [EOS] that ends it)."""
    rng = random.Random(seed)
    ids = np.empty((n_rows, seq_len), np.int32)
    for r in range(n_rows):
        ids[r] = [TOKEN_ID[w] for w in st.sample(st.stream(), seq_len, rng)]
    ends = ids == TOKEN_ID[st.EOS]
    documents = (np.cumsum(ends, axis=1) - ends).astype(np.int32)
    return ids, documents


def _write_shard(args: tuple[Path, int, int, int, int]) -> None:
    out, shard, n_rows, seq_len, seed = args
    ids, documents = rows(n_rows, seq_len, seed)
    table = {
        "input_ids": pa.FixedSizeListArray.from_arrays(pa.array(ids.reshape(-1)), seq_len),
        "document_ids": pa.FixedSizeListArray.from_arrays(pa.array(documents.reshape(-1)), seq_len),
    }
    pq.write_table(pa.table(table), out / f"shard_{shard:05d}.parquet", row_group_size=8192)


def generate(
    out_dir: str, *, tokenizer_dir: str, n_rows: int, seed: int, seq_len: int = 513, shards: int = 64
) -> None:
    """Write `n_rows` rows as `shards` parquet shards plus `meta.json` under `out_dir`."""
    from param_decomp_formats.dataset_store import DatasetMeta, write_dataset_meta

    out = Path(out_dir)
    assert out.is_absolute() and Path(tokenizer_dir).is_absolute(), (out_dir, tokenizer_dir)
    out.mkdir(parents=True)
    per_shard = -(-n_rows // shards)
    jobs = [
        (out, k, min(per_shard, n_rows - k * per_shard), seq_len, seed * 1_000_003 + k)
        for k in range(shards)
        if k * per_shard < n_rows
    ]
    with Pool() as pool:
        pool.map(_write_shard, jobs)
    write_dataset_meta(
        out,
        DatasetMeta(
            format_version=2,
            seq_len=seq_len,
            tokenizer_name=tokenizer_dir,
            preprocessing_name=PREPROCESSING_NAME,
            end_of_document_token_id=TOKEN_ID[st.EOS],
        ),
    )


def finish(tokenized_dir: str, dataset_dir: str, *, producer_revision: str, seed: int = 0) -> None:
    """Publish `dataset_dir` as the rows of `tokenized_dir` shuffled across it."""
    from param_decomp.infra.row_shuffle import shuffle_rows

    out = Path(dataset_dir)
    shuffle_rows(
        Path(tokenized_dir),
        out,
        out.parent / f".{out.name}.staging",
        seed,
        rows_per_shard=1_000_000,
        buckets_per_task=1,
        workers=8,
        producer_revision=producer_revision,
    )


if __name__ == "__main__":
    fire.Fire({"tokenizer": tokenizer, "generate": generate, "finish": finish})
