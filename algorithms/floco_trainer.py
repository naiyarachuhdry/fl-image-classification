"""FLOCo Client Trainer matching baseline implementation."""

import copy
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer


def _regularize_pers_model(model: nn.Module, reg_params: list, lamda: float) -> None:
    """Regularize personalized model gradients towards the global model parameters."""
    for pers_p, global_p in zip(model.parameters(), reg_params):
        if pers_p.requires_grad and pers_p.grad is not None:
            pers_p.grad.data += lamda * (pers_p.data - global_p.data)


class FlocoTrainer(BaseTrainer):
    """Client-side trainer for FLOCo."""

    def __init__(self, pers_lamda: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.pers_lamda = float(pers_lamda)

    def compute_loss(self, model: nn.Module, outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        return criterion(outputs, labels)

    def train(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        epochs: int,
        lr: float,
        device: torch.device,
        subregion_parameters: Optional[Tuple[np.ndarray, float]] = None,
        pers_model: Optional[nn.Module] = None,
    ) -> float:
        """Local training on simplex/subregion."""
        model.to(device)
        model.train()
        if hasattr(model, "subregion_parameters"):
            model.subregion_parameters = subregion_parameters

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.5,
            weight_decay=5e-4,
        )

        reg_params = copy.deepcopy(list(model.parameters())) if self.pers_lamda > 0 else None

        total_loss = 0.0
        num_batches = 0

        for _ in range(epochs):
            for batch in trainloader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
                if images.size(0) <= 1:
                    continue

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()

                if reg_params is not None:
                    _regularize_pers_model(model, reg_params, self.pers_lamda)

                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        subregion_parameters: Optional[Tuple[np.ndarray, float]] = None,
    ) -> Tuple[float, float]:
        """Local evaluation on validation set."""
        model.to(device)
        model.eval()
        if hasattr(model, "subregion_parameters"):
            model.subregion_parameters = subregion_parameters

        criterion = nn.CrossEntropyLoss().to(device)
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += images.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy