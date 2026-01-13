from datasets import load_dataset
from custom_tk import Tokenizer
import torch

ds_train = load_dataset("stanfordnlp/imdb")['train']
ds_test = load_dataset("stanfordnlp/imdb")['test']

tk = Tokenizer(500)

def tokenize(batch):
    return {
        "tokens": [tk.encode(x) for x in batch['text']]
    }

def main():
    ds_train_tok = ds_train.map(tokenize, batched=True)
    #ds_train_tok.save_to_disk('dataset/IMDB/train')
    ds_test_tok = ds_test.map(tokenize, batched=True)
    #ds_test_tok.save_to_disk('dataset/IMDB/test')

    Y_test = torch.tensor(ds_test_tok['label'], dtype=torch.long)
    Y_train = torch.tensor(ds_train_tok['label'], dtype=torch.long)

    X_test = torch.tensor(ds_test_tok['tokens'], dtype=torch.long)
    X_train = torch.tensor(ds_train_tok['tokens'], dtype=torch.long)

    assert X_train.shape == (25000, 500), X_train.shape
    assert Y_train.shape == (25000,), Y_train.shape
    assert X_test.shape == (25000, 500), X_test.shape
    assert Y_test.shape == (25000,), Y_test.shape

    torch.save((X_test, Y_test), "dataset/IMDB/test/test.pt")
    torch.save((X_train, Y_train), "dataset/IMDB/train/train.pt")

if __name__ == "__main__":
    main()