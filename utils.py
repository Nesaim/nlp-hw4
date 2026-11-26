import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def custom_transform(example):
    import re

    neighbors = {
        "q": "wa", "w": "qes", "e": "wrd", "r": "etf",
        "t": "ryg", "y": "tuh", "u": "yij", "i": "uok",
        "o": "ipl", "p": "ol",
        "a": "qsz", "s": "wadx", "d": "esfc", "f": "rdgv",
        "g": "tfhb", "h": "ygjn", "j": "uhkm",
        "k": "ijlm", "l": "ok",
        "z": "asx", "x": "zsdc", "c": "xdfv",
        "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk"
    }

    def introduce_typo(match):
        word = match.group(0)

        if len(word) < 4 or random.random() >= 0.35:
            return word

        position = random.randrange(len(word))
        original = word[position]
        replacement = random.choice(neighbors[original.lower()])

        if original.isupper():
            replacement = replacement.upper()

        return word[:position] + replacement + word[position + 1:]

    example["text"] = re.sub(
        r"\b[A-Za-z]+\b", introduce_typo, example["text"]
    )
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.

