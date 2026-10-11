"""Vòng lặp huấn luyện: Adam lr 1e-3, BCEWithLogitsLoss, early stopping patience 2."""
import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, default_collate

from . import config as C


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _collate(batch):
    x, l, y = default_collate(batch)
    return x[:, : int(l.max())], l, y      # cắt phần pad thừa trong batch cho nhanh


def make_loader(ids, lengths, labels, shuffle, seed=0, batch_size=C.BATCH_SIZE):
    ds = TensorDataset(torch.as_tensor(ids), torch.as_tensor(lengths),
                       torch.as_tensor(labels, dtype=torch.float32))
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen,
                      collate_fn=_collate)


@torch.no_grad()
def predict(model, loader, device):
    """Trả về (loss trung bình, xác suất dương, nhãn thật)."""
    model.eval()
    crit = nn.BCEWithLogitsLoss(reduction="sum")
    total, probs, labels = 0.0, [], []
    for x, l, y in loader:
        x, l, y = x.to(device), l.to(device), y.to(device)
        logits = model(x, l)
        total += crit(logits, y).item()
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.cpu().numpy())
    probs, labels = np.concatenate(probs), np.concatenate(labels)
    return total / len(labels), probs, labels


def train_model(model, train_loader, val_loader, device, max_epochs=C.MAX_EPOCHS,
                patience=C.PATIENCE, verbose=True):
    """Huấn luyện với early stopping theo validation loss; khôi phục trọng số tốt nhất.

    Learning rate hằng số 1e-3 (không dùng scheduler) cho mọi điều kiện.
    Trả về (history, best_epoch, thời gian huấn luyện giây).
    """
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=C.LR)
    crit = nn.BCEWithLogitsLoss()
    best, best_state, best_epoch, bad = float("inf"), None, 0, 0
    history, t0 = [], time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        run_loss, n = 0.0, 0
        for x, l, y in train_loader:
            x, l, y = x.to(device), l.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x, l), y)
            loss.backward()
            nn.utils.clip_grad_norm_(params, C.GRAD_CLIP)
            opt.step()
            run_loss += loss.item() * len(y)
            n += len(y)
        val_loss, probs, labels = predict(model, val_loader, device)
        val_acc = float(((probs > 0.5) == (labels > 0.5)).mean())
        history.append({"epoch": epoch, "train_loss": run_loss / n,
                        "val_loss": val_loss, "val_acc": val_acc})
        if verbose:
            print(f"    epoch {epoch:2d} | train {run_loss / n:.4f} | "
                  f"val {val_loss:.4f} | val acc {val_acc:.4f}", flush=True)
        if val_loss < best:
            best, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return history, best_epoch, time.time() - t0
