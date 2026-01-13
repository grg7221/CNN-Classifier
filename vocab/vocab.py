from datasets import load_dataset
import re
from collections import Counter
import json

ds = load_dataset("stanfordnlp/imdb")['train']

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

# Считаем самые частые слова
counter = Counter()

for text in ds['text']:
    tokens = clean_text(text)
    counter.update(tokens)

most_common = counter.most_common(25000 - 2)

word2idx = {
    "<PAD>": 0,
    "<UNK>": 1
}

for i, (word, _) in enumerate(most_common, start=2):
    word2idx[word] = i

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(word2idx, f, ensure_ascii=False, indent=2)