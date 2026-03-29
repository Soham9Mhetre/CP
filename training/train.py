import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report

from data.load_dataset import load_dataset
from models.adversarial_injection import inject_adversarial_edges
from models.gat_encoder import FraudGAT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =====================
# Load Dataset
# =====================

data = load_dataset()

# Inject adversarial nodes
data = inject_adversarial_edges(data)

data = data.to(device)

# Debug prints (optional)
print("Total nodes:", data.x.shape[0])
print("Total edges:", data.edge_index.shape[1])


# =====================
# Model
# =====================

model = FraudGAT(
    input_dim=data.num_features,
    hidden_dim=64,
    output_dim=2
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


# =====================
# Class Weights
# =====================

class_weights = torch.tensor([1.0, 9.0]).to(device)


# =====================
# Training Loop
# =====================
print("Unique labels:", torch.unique(data.y))
model.train()

for epoch in range(250):

    optimizer.zero_grad()

    out = model(data.x, data.edge_index, data.time_steps)

    # Ignore fake nodes
    mask = (data.y == 0) | (data.y == 1)

    loss = F.cross_entropy(out[mask], data.y[mask], weight=class_weights)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} Loss: {loss.item()}")


# =====================
# Evaluation
# =====================

model.eval()

with torch.no_grad():

    out = model(data.x, data.edge_index, data.time_steps)

    pred = out.argmax(dim=1)

    #  Ignore fake nodes
    mask = mask = (data.test_mask) & ((data.y == 0) | (data.y == 1))

    correct = (pred[mask] == data.y[mask]).sum()

    acc = int(correct) / int(mask.sum())

    print("Test Accuracy:", acc)

    print(classification_report(
        data.y[mask].cpu(),
        pred[mask].cpu()
    ))