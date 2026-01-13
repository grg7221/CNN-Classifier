import re
import json

class Tokenizer:
    def __init__(self, max_len):
        with open("vocab/vocab.json", "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)
        self.max_len = max_len

    def encode(self, text, make_pads=True):
        text = text.lower()
        text = re.sub(r"<br\s*/?>", " ", text)
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = text.split()

        tokens = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in text]

        if make_pads:
            tokens = tokens[:self.max_len]

            if len(tokens) < self.max_len:
                tokens = tokens + [self.word2idx["<PAD>"]] * (self.max_len - len(tokens))

        return tokens
    
    def decode(self, tokens):
        pass