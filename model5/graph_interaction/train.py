"""Training loop for Model 5 (Step 6).

Transductive setup: the full graph is used for message passing every step,
but the loss is only computed on training-node labels. Class-weighted BCE
(pos_weight ~= 19 for the ~5% fraud rate), Adam (lr=1e-3), early stopping on
validation PR-AUC. After training, scores are calibrated (Platt scaling) on
the validation split so Model 5 sits near 0 for normal profiles, the same
way the noisy-OR fusion needs every other model to behave — otherwise it
would distort the fusion the way M2-A's uncalibrated baseline did.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from model5.graph_interaction.fusion_head import GraphInteractionModel


@dataclass
class TrainResult:
    model: nn.Module
    history: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    calibrator: LogisticRegression | None = None


def _pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if y_true.sum() == 0:
        return 0.0
    return float(average_precision_score(y_true, scores))


def train_model(
    data,
    seq: torch.Tensor | None,
    seq_pad_mask: torch.Tensor | None,
    use_sequence: bool,
    in_dim: int,
    use_skip: bool = True,
    use_gat: bool = True,
    pos_weight: float = 19.0,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    max_epochs: int = 150,
    patience: int = 15,
    seed: int = 42,
    device: str = "cpu",
) -> TrainResult:
    torch.manual_seed(seed)
    model = GraphInteractionModel(in_dim=in_dim, use_sequence=use_sequence, use_skip=use_skip, use_gat=use_gat).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.float().to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    seq_d = seq.to(device) if seq is not None else None
    seq_pad_d = seq_pad_mask.to(device) if seq_pad_mask is not None else None

    best_val_pr_auc = -1.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index, seq_d, seq_pad_d)
        loss = loss_fn(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index, seq_d, seq_pad_d)
            val_scores = torch.sigmoid(logits[val_mask]).cpu().numpy()
            val_y = y[val_mask].cpu().numpy()
            val_pr_auc = _pr_auc(val_y, val_scores)

        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_pr_auc": val_pr_auc})

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Platt scaling: fit a 1-D logistic regression (raw logit -> calibrated
    # probability) on the validation split only.
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, seq_d, seq_pad_d)
        val_logits = logits[val_mask].cpu().numpy().reshape(-1, 1)
        val_y = y[val_mask].cpu().numpy().astype(int)
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(val_logits, val_y)

    return TrainResult(model=model, history=history, best_epoch=best_epoch, calibrator=calibrator)


def calibrated_scores(model: nn.Module, calibrator: LogisticRegression, data, seq, seq_pad_mask, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device),
                        seq.to(device) if seq is not None else None,
                        seq_pad_mask.to(device) if seq_pad_mask is not None else None)
    raw = logits.cpu().numpy().reshape(-1, 1)
    return calibrator.predict_proba(raw)[:, 1]
