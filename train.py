from CNN import CNN
from torch.optim import Adam
from plots import train_plots
import torch
import torch.nn.functional as F
import csv
#from tqdm import trange

torch.manual_seed(2506)

batch_size = 512
vocab_size = 25000
embed_dim = 128
num_filters = 200
kernel_sizes = [3, 4, 5]
epochs = 15
val_split = 0.1
lr = 1e-3 * (batch_size / 512)

# создаем csv для текущего набора гиперпараметров
with open(f'metrics/b{batch_size}_d{embed_dim}_f{num_filters}.csv', mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'val_loss', 'accuracy', 'precision', 'recall', 'F1'])

data = torch.load('dataset/IMDB/train/train.pt', map_location='cuda')
X = data[0].long()
Y = data[1].long()
N = len(X) # количество документов

# шафлим
perm = torch.randperm(N, device='cuda')
X = X[perm]
Y = Y[perm]

# разделяем на train и val
split = int(N*val_split)
N -= split # количество док-ов в train split
X_train = X[split:]
Y_train = Y[split:]

X_val = X[:split]
Y_val = Y[:split]

# инициализируем модель и оптимизатор
model = CNN(
    vocab_size=vocab_size,
    embed_dim=embed_dim,          # C
    num_classes=2,          # output dim
    kernel_sizes=kernel_sizes,
    num_filters=num_filters
).cuda().train()
optimizer = Adam(model.parameters(), lr)

print(f"Total params: {sum(p.numel() for p in model.parameters())}")

for e in range(epochs):
    # шафлим каждую эпоху
    perm = torch.randperm(N, device='cuda')
    X_shuffle = X_train[perm]
    Y_shuffle = Y_train[perm]

    total_loss = 0

    # train split
    model.train()
    for i in range(0, N, batch_size):
        xb = X_shuffle[i:i+batch_size]
        yb = Y_shuffle[i:i+batch_size]

        logits = model(xb)
        loss = F.cross_entropy(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # val split
    model.eval()
    with torch.no_grad():
        TP = FP = FN = TN = 0
        for i in range(0, len(X_val), batch_size):
            logits = model(X_val[i:i+batch_size])
            val_loss = F.cross_entropy(logits, Y_val[i:i+batch_size])
            preds = logits.argmax(dim=1)
            y = Y_val[i:i+batch_size]

            TP += ((preds == 1) & (y == 1)).sum().item()
            FP += ((preds == 1) & (y == 0)).sum().item()
            FN += ((preds == 0) & (y == 1)).sum().item()
            TN += ((preds == 0) & (y == 0)).sum().item()

    # метрики
    accuracy = (TP + TN) / (TP + TN + FP + FN) # доля правильных ответов
    precision = TP / (TP + FP + 1e-8) # сколько реально positive из найденных
    recall = TP / (TP + FN + 1e-8) # сколько positive найдено из всех
    F1 = 2*(precision*recall)/(precision+recall)
    print(f"Epoch {e+1}: total_loss={total_loss:.4f}, last loss={loss.item():.4f}")
    print(f"Val split: val_loss={val_loss.item():.4f} acc={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}, F1={F1:.4f}")

    with open(f'metrics/b{batch_size}_d{embed_dim}_f{num_filters}.csv', mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([e, loss.item(), val_loss.item(), accuracy, precision, recall, F1])

torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
}, f"checkpoints/b{batch_size}_d{embed_dim}_f{num_filters}.pt")

train_plots(batch_size, embed_dim, num_filters)