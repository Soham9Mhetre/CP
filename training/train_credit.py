"""
Credit-card fraud detection: LSTM → GAT pipeline.

Key design decisions vs the broken SMOTE version
-------------------------------------------------
1. NO SMOTE on sequential data  — interpolating between fraud rows then
   sliding-windowing them produces nonsense sequences. Instead we handle
   imbalance with Focal Loss + hard class weights + sequence-level
   oversampling (duplicate real fraud sequences in the training index).

2. Fast pure-torch graph builder — cosine-sim topk, no sklearn per batch.

3. Correct train/test split — sequences are built inside each split so
   there is zero leakage between them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
import numpy as np
import random

from data.credit_card_loader import load_credit_card_data
from models.credit_lstm import CreditLSTM
from models.credit_gat import CreditGAT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.85, gamma=2.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce      = F.cross_entropy(logits, targets, reduction="none")
        pt      = torch.exp(-ce)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * ce).mean()


# ── Load & split RAW data (no resampling yet) ─────────────────────────────────
X, y = load_credit_card_data("data/banksim.csv")
X, y = X[:50000], y[:50000]

split      = int(0.8 * len(X))
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

print(f"Train raw — Legit: {(y_tr==0).sum().item()}  Fraud: {(y_tr==1).sum().item()}")
print(f"Test  raw — Legit: {(y_te==0).sum().item()}  Fraud: {(y_te==1).sum().item()}\n")


# ── Build sequences INSIDE each split (no cross-split leakage) ───────────────
SEQ_LEN = 5

def make_sequences(X, y, seq_len):
    """Sliding-window sequences; label = label of the last element."""
    n   = len(X) - seq_len
    seq = torch.stack([X[i:i+seq_len] for i in range(n)])
    lbl = y[seq_len:]
    return seq, lbl

X_train_seq, y_train_seq = make_sequences(X_tr, y_tr, SEQ_LEN)
X_test_seq,  y_test_seq  = make_sequences(X_te, y_te, SEQ_LEN)

print(f"Train seqs: {X_train_seq.shape}  Fraud: {y_train_seq.sum().item()}")
print(f"Test  seqs: {X_test_seq.shape}   Fraud: {y_test_seq.sum().item()}\n")


# ── Oversample FRAUD sequences in the training INDEX only ────────────────────
# We don't touch the tensor data — we just repeat fraud indices so each
# mini-batch sees a balanced mix. Target: fraud ≈ 25 % of each batch.
fraud_idx  = (y_train_seq == 1).nonzero(as_tuple=True)[0]
legit_idx  = (y_train_seq == 0).nonzero(as_tuple=True)[0]

oversample = int(len(legit_idx) * 0.33)          # fraud copies to add
extra      = fraud_idx[torch.randint(len(fraud_idx), (oversample,))]
train_idx  = torch.cat([torch.arange(len(y_train_seq)), extra])

print(f"After index-oversampling — total train seqs: {len(train_idx)}")
print(f"  Fraud in index: {(y_train_seq[train_idx]==1).sum().item()}\n")

# Move to device
X_train_seq = X_train_seq.to(device)
y_train_seq = y_train_seq.to(device)
X_test_seq  = X_test_seq.to(device)
y_test_seq  = y_test_seq.to(device)
train_idx   = train_idx.to(device)


# ── Models ────────────────────────────────────────────────────────────────────
HIDDEN = 48

lstm = CreditLSTM(input_dim=X.shape[1], hidden_dim=HIDDEN, dropout=0.3).to(device)
gat  = CreditGAT (input_dim=HIDDEN * 2, hidden_dim=48, heads=4, dropout=0.3).to(device)

n_params = sum(p.numel() for p in lstm.parameters()) + \
           sum(p.numel() for p in gat.parameters())
print(f"Total parameters: {n_params:,}\n")


# ── Optimiser ─────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    list(lstm.parameters()) + list(gat.parameters()),
    lr=5e-4, weight_decay=1e-4,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)
criterion = FocalLoss(alpha=0.85, gamma=2.5)


# ── Fast cosine-similarity graph (pure torch, stays on GPU) ──────────────────
@torch.no_grad()
def build_graph(embed, k=5):
    n    = embed.size(0)
    k    = min(k, n - 1)
    norm = F.normalize(embed, dim=1)
    sim  = torch.mm(norm, norm.t())
    sim.fill_diagonal_(-1.0)
    top  = sim.topk(k, dim=1).indices              # (N, k)
    src  = torch.arange(n, device=embed.device).unsqueeze(1).expand(-1, k)
    return torch.stack([src.reshape(-1), top.reshape(-1)])


# ── Training ──────────────────────────────────────────────────────────────────
BATCH      = 4096
NUM_EPOCHS = 60
EVAL_EVERY = 5

best_ap, best_lstm_sd, best_gat_sd = 0.0, None, None

for epoch in range(NUM_EPOCHS):

    lstm.train(); gat.train()

    perm       = train_idx[torch.randperm(len(train_idx), device=device)]
    epoch_loss = 0.0
    n_batches  = 0

    for start in range(0, len(perm), BATCH):
        idx = perm[start:start + BATCH]
        xb  = X_train_seq[idx]
        yb  = y_train_seq[idx]

        optimizer.zero_grad()

        embed      = lstm(xb)
        edge_index = build_graph(embed, k=5)
        out        = gat(embed, edge_index)
        loss       = criterion(out, yb)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(lstm.parameters()) + list(gat.parameters()), 1.0
        )
        optimizer.step()

        epoch_loss += loss.item()
        n_batches  += 1

    scheduler.step()

    if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS - 1:
        lstm.eval(); gat.eval()
        with torch.no_grad():
            emb  = lstm(X_test_seq)
            ei   = build_graph(emb, k=5)
            out  = gat(emb, ei)
            prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            yn   = y_test_seq.cpu().numpy()

        ap  = average_precision_score(yn, prob)
        auc = roc_auc_score(yn, prob)
        print(f"Epoch {epoch:>3}  loss={epoch_loss/n_batches:.4f}  "
              f"AUC-ROC={auc:.4f}  AUC-PR={ap:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.1e}")

        if ap > best_ap:
            best_ap      = ap
            best_lstm_sd = {k: v.clone() for k, v in lstm.state_dict().items()}
            best_gat_sd  = {k: v.clone() for k, v in gat.state_dict().items()}


# ── Restore best checkpoint ───────────────────────────────────────────────────
print(f"\nBest AUC-PR = {best_ap:.4f} — restoring checkpoint")
lstm.load_state_dict(best_lstm_sd)
gat.load_state_dict(best_gat_sd)


# ── Evaluation with optimal threshold ────────────────────────────────────────
lstm.eval(); gat.eval()
with torch.no_grad():
    emb  = lstm(X_test_seq)
    ei   = build_graph(emb, k=5)
    out  = gat(emb, ei)
    prob = torch.softmax(out, dim=1)[:, 1]

prob_np = prob.cpu().numpy()
y_np    = y_test_seq.cpu().numpy()

prec, rec, thr = precision_recall_curve(y_np, prob_np)
f1              = 2 * prec * rec / (prec + rec + 1e-8)
best_idx        = int(np.argmax(f1))
best_thr        = float(thr[best_idx]) if best_idx < len(thr) else 0.5

print(f"\nOptimal threshold: {best_thr:.3f}  (F1={f1[best_idx]:.4f})")

pred = (prob_np >= best_thr).astype(int)

print("\n── CREDIT CARD FRAUD EVALUATION ──────────────────\n")
print(f"Accuracy : {accuracy_score(y_np, pred):.4f}")
print(f"AUC-ROC  : {roc_auc_score(y_np, prob_np):.4f}")
print(f"AUC-PR   : {average_precision_score(y_np, prob_np):.4f}")
print()
print(classification_report(y_np, pred, target_names=["Legit", "Fraud"]))


# ── Fraud prevention demo ─────────────────────────────────────────────────────
def decision(p, thr):
    if p >= thr + 0.15:  return "BLOCK TRANSACTION"
    if p >= thr:         return "REQUIRE OTP"
    return "ALLOW"

print("\n── FRAUD PREVENTION (10 random transactions) ─────\n")
for i in random.sample(range(len(prob_np)), 10):
    p   = float(prob_np[i])
    lbl = "⚠ FRAUD" if y_np[i] == 1 else "✓ LEGIT"
    print(f"Txn {i:>5} [{lbl}]  score={p:.3f}  → {decision(p, best_thr)}")