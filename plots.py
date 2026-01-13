import matplotlib.pyplot as plt
from datasets import load_dataset
from dataset.custom_tk import Tokenizer

tk = Tokenizer(500)
def tokenize(batch):
    return {
        "tokens": [tk.encode(x, False) for x in batch['text']]
    }

ds_train = load_dataset("stanfordnlp/imdb")['train']
ds_train_tok = ds_train.map(tokenize, batched=True)

text_len = []
for tokens in ds_train_tok['tokens']:
    text_len.append(len(tokens))

plt.figure(figsize=(10,6))
plt.hist(text_len, 100)

plt.xlabel("Length")
plt.ylabel("Freq")

plt.savefig(f'plots/text_lengths.png', dpi=150)

plt.show()