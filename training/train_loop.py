import time
import copy
import os
from typing import Optional

import torch


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None,
    num_epochs: int = 10,
    save_path: Optional[str] = None,
    grad_clip: Optional[float] = None,
):
    """Train a PyTorch model with a standard train/validation loop.

    Returns the best model (weights) found on validation accuracy.

    Args:
        model: torch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data (can be None to skip validation).
        criterion: loss function.
        optimizer: optimizer.
        scheduler: optional LR scheduler (stepped once per epoch).
        device: torch.device or None to auto-select.
        num_epochs: number of epochs.
        save_path: path to save best checkpoint (optional).
        grad_clip: max norm to clip gradients (optional).
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    since = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        running_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            preds = outputs.argmax(dim=1)
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels).item()
            running_samples += batch_size

        epoch_loss = running_loss / max(1, running_samples)
        epoch_acc = running_corrects / max(1, running_samples)

        # --- Validation phase ---
        val_loss = None
        val_acc = None
        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0
            val_running_corrects = 0
            val_running_samples = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)
                    bs = inputs.size(0)
                    val_running_loss += loss.item() * bs
                    val_running_corrects += torch.sum(preds == labels).item()
                    val_running_samples += bs

            val_loss = val_running_loss / max(1, val_running_samples)
            val_acc = val_running_corrects / max(1, val_running_samples)

        # Step scheduler once per epoch (if provided)
        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                # Some schedulers expect metrics; ignore here for simplicity
                pass

        epoch_time = time.time() - epoch_start

        # Print short summary for this epoch
        if val_loader is not None:
            print(
                f"Epoch {epoch}/{num_epochs} - "
                f"train_loss: {epoch_loss:.4f} acc: {epoch_acc:.4f} | "
                f"val_loss: {val_loss:.4f} acc: {val_acc:.4f} - {epoch_time:.1f}s"
            )
        else:
            print(
                f"Epoch {epoch}/{num_epochs} - "
                f"train_loss: {epoch_loss:.4f} acc: {epoch_acc:.4f} - {epoch_time:.1f}s"
            )

        # Checkpoint best
        if val_acc is not None and val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            if save_path is not None:
                _save_checkpoint(save_path, model, optimizer, epoch, best_acc)

    total_time = time.time() - since
    print(f"Training complete in {total_time//60:.0f}m {total_time%60:.0f}s. Best val acc: {best_acc:.4f}")

    # load best weights
    model.load_state_dict(best_model_wts)
    return model


def _save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_acc: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_acc": best_acc,
    }
    torch.save(state, path)
