import torch
import torch.nn.functional as F

from models.gcn import GCN
from data.load_dataset import load_dataset

from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = load_dataset()
data = data.to(device)

model = GCN(
    input_dim=data.num_features,
    hidden_dim=64,
    output_dim=2
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


for epoch in range(100):

    model.train()

    out = model(data.x, data.edge_index)
    weights = torch.tensor([1.0, 9.0]).to(device)

    loss = F.cross_entropy(
    out[data.train_mask],
    data.y[data.train_mask],
    weight=weights
)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Epoch:", epoch, "Loss:", loss.item())




model.eval()

with torch.no_grad():

    out = model(data.x, data.edge_index)

    pred = out.argmax(dim=1)

    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()

    acc = int(correct) / int(data.test_mask.sum())

    print("Test Accuracy:", acc)
  

print(classification_report(
    data.y[data.test_mask].cpu(),
    pred[data.test_mask].cpu()
))

torch.save(model.state_dict(), "models/gcn_weighted_baseline.pth")