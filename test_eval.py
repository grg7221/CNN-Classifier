import torch
from CNN import CNN
import torch.nn.functional as F
import csv

batch_size = 512
vocab_size = 25000
embed_dim = 128
num_filters = 100
kernel_sizes = [3, 4, 5]

data = torch.load('dataset/IMDB/test/test.pt', map_location='cuda')

X = data[0].long()
Y = data[1].long()
N = len(X)

pt = torch.load('checkpoints/b512_d128_f100.pt')

with open(f'metrics/test_split.csv', mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['avg_loss', 'accuracy', 'precision', 'recall', 'F1'])

model = CNN(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_classes=2,
    kernel_sizes=kernel_sizes,
    num_filters=num_filters
).cuda()
model.load_state_dict(pt['model'])
model.eval()

with torch.no_grad():
    TP = FP = FN = TN = 0
    total_loss = 0

    for i in range(0, N, batch_size):
        xb = X[i:i+batch_size]
        yb = Y[i:i+batch_size]

        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        total_loss += loss.item()

        preds = logits.argmax(dim=1)

        TP += ((preds == 1) & (yb == 1)).sum().item()
        FP += ((preds == 1) & (yb == 0)).sum().item()
        FN += ((preds == 0) & (yb == 1)).sum().item()
        TN += ((preds == 0) & (yb == 0)).sum().item()

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    F1 = 2*(precision*recall)/(precision+recall+1e-8)
    avg_loss = total_loss / (N/batch_size)

    with open(f'metrics/test_split.csv', mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([avg_loss, accuracy, precision, recall, F1])