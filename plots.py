import matplotlib.pyplot as plt
from datasets import load_dataset
from dataset.custom_tk import Tokenizer
from pandas import read_csv
import plotly.express as px

def dataset_text_length():
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

def train_plots(batch_size, embed_dim, num_filters):
    df = read_csv(f'metrics/b{batch_size}_d{embed_dim}_f{num_filters}.csv')

    def line_plot(df, y_cols, title, y_label, file_name):
        d = df.melt(id_vars="epoch", value_vars=y_cols, var_name="metric", value_name="value")
        fig = px.line(d, x="epoch", y="value", color="metric", markers=True, title=title, labels={"epoch": "Epoch", "value": y_label, "metric": ""})
        fig.update_layout(template="plotly_white")
        fig.write_image(f'plots/{file_name}_b{batch_size}_d{embed_dim}_f{num_filters}.png', width=900, height=500, scale=2)
        #fig.show()

    line_plot(
        df,
        y_cols=["train_loss", "val_loss"],
        title="Loss: train vs validation",
        y_label="Cross-entropy loss",
        file_name='Loss'
    )

    line_plot(
        df,
        y_cols=["accuracy", "F1"],
        title="Accuracy and F1",
        y_label="Score",
        file_name='Acc+F1'
    )

    line_plot(
        df,
        y_cols=["precision", "recall"],
        title="Precision and Recall",
        y_label="Score",
        file_name='Prec+Rec'
    )