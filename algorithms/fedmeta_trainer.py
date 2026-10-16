"""FedMeta: Federated Meta-Learning Client Trainers.

Supports:
- FedAvg-Meta: Standard local SGD on support set + 1-step meta-testing adaptation.
- FedMeta-MAML: Inner-loop adaptation on support set + outer-loop meta-gradients on query set.
- FedMeta-Meta-SGD: Inner-loop adaptation with learnable vector step size alpha + meta-gradients.
"""

from copy import deepcopy
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .base_trainer import BaseTrainer


def split_support_query(dataloader: DataLoader, support_ratio: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """Split a DataLoader's dataset into support and query DataLoaders.
    
    Uses drop_last=True to prevent 1-sample batches that crash BatchNorm in training mode.
    """
    dataset = dataloader.dataset
    total_len = len(dataset)
    sup_len = max(1, int(total_len * support_ratio))
    qry_len = total_len - sup_len

    indices = list(range(total_len))
    batch_size = dataloader.batch_size or 32

    # Drop remainder batches only if we have more than 1 batch to prevent single-sample BatchNorm crash
    drop_sup = sup_len > batch_size
    drop_qry = qry_len > batch_size

    sup_loader = DataLoader(
        Subset(dataset, indices[:sup_len]),
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_sup,
    )
    qry_loader = DataLoader(
        Subset(dataset, indices[sup_len:]),
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_qry,
    )
    return sup_loader, qry_loader


class FedMetaBaseTrainer(BaseTrainer):
    """Base class for FedMeta trainers with support/query splitting and meta-testing."""

    def __init__(
        self,
        alpha: float = 0.01,
        gradient_step: int = 5,
        support_ratio: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha = float(alpha)
        self.gradient_step = int(gradient_step)
        self.support_ratio = float(support_ratio)
        self.last_gradients: Optional[Dict[str, torch.Tensor]] = None

    def compute_loss(self, model, outputs, labels, criterion) -> torch.Tensor:
        return criterion(outputs, labels)

    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """Return the outer-loop meta-gradients computed during train()."""
        return self.last_gradients or {}


class FedAvgMetaTrainer(FedMetaBaseTrainer):
    """FedAvg-Meta Client Trainer."""

    def train(self, model, trainloader, epochs, lr, device) -> float:
        """Standard local SGD training on support set."""
        sup_loader, _ = split_support_query(trainloader, self.support_ratio)

        model.to(device)
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        total_loss = 0.0
        num_batches = 0
        for _ in range(epochs):
            for batch in sup_loader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)

                # Skip single-sample batch to protect BatchNorm
                if images.size(0) <= 1:
                    continue

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def evaluate(self, model, valloader, device) -> Tuple[float, float]:
        """Few-shot meta-testing: 1-step Adam adaptation on support, test on query."""
        sup_loader, qry_loader = split_support_query(valloader, self.support_ratio)
        test_net = deepcopy(model).to(device)
        test_net.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(test_net.parameters(), lr=self.alpha, weight_decay=1e-3)

        for batch in sup_loader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            if images.size(0) <= 1:
                continue

            optimizer.zero_grad()
            loss = criterion(test_net(images), labels)
            loss.backward()
            optimizer.step()

        # Evaluate adapted model on query set
        test_net.eval()
        correct, total, eval_loss = 0, 0, 0.0
        with torch.no_grad():
            for batch in qry_loader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
                outputs = test_net(images)
                eval_loss += criterion(outputs, labels).item() * labels.size(0)
                preds = torch.max(outputs, 1)[1]
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        return eval_loss / max(total, 1), correct / max(total, 1)


class FedMetaMamlTrainer(FedMetaBaseTrainer):
    """FedMeta-MAML Client Trainer."""

    def train(self, model, trainloader, epochs, lr, device) -> float:
        sup_loader, qry_loader = split_support_query(trainloader, self.support_ratio)

        model.to(device)
        train_net = deepcopy(model).to(device)
        train_net.train()
        criterion = nn.CrossEntropyLoss()

        step_size = self.alpha

        # 1. Inner loop adaptation on support loader
        for _ in range(self.gradient_step):
            for batch in sup_loader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)

                # Skip single-sample batch to protect BatchNorm
                if images.size(0) <= 1:
                    continue

                loss = criterion(train_net(images), labels)
                grads = torch.autograd.grad(
                    loss,
                    list(train_net.parameters()),
                    create_graph=True,
                    retain_graph=True,
                )

                for param, g in zip(train_net.parameters(), grads):
                    param.data = param.data - step_size * g

                for param in train_net.parameters():
                    if param.grad is not None:
                        param.grad.zero_()

        # 2. Outer loop evaluation on query loader
        loss_sum = torch.tensor(0.0, device=device)
        total_samples = 0
        for batch in qry_loader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            if images.size(0) <= 1:
                continue

            loss = criterion(train_net(images), labels)
            loss_sum = loss_sum + loss * labels.size(0)
            total_samples += labels.size(0)

        loss_avg = loss_sum / max(total_samples, 1)

        # 3. Outer loop meta-gradients
        meta_grads = torch.autograd.grad(loss_avg, list(train_net.parameters()))

        param_names = [k for k, v in model.named_parameters() if v.requires_grad]
        self.last_gradients = {
            name: g.detach().cpu()
            for name, g in zip(param_names, meta_grads)
        }

        for name, buf in model.named_buffers():
            self.last_gradients[name] = buf.detach().cpu()

        return float(loss_avg.detach().item())

    def evaluate(self, model, valloader, device) -> Tuple[float, float]:
        """Meta-testing adaptation on support, test on query."""
        sup_loader, qry_loader = split_support_query(valloader, self.support_ratio)
        test_net = deepcopy(model).to(device)
        test_net.train()
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.gradient_step):
            for batch in sup_loader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)

                if images.size(0) <= 1:
                    continue

                loss = criterion(test_net(images), labels)
                grads = torch.autograd.grad(
                    loss,
                    list(test_net.parameters()),
                    create_graph=True,
                    retain_graph=True,
                )
                for param, g in zip(test_net.parameters(), grads):
                    param.data = param.data - self.alpha * g

                for param in test_net.parameters():
                    if param.grad is not None:
                        param.grad.zero_()

        test_net.eval()
        correct, total, eval_loss = 0, 0, 0.0
        with torch.no_grad():
            for batch in qry_loader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
                outputs = test_net(images)
                eval_loss += criterion(outputs, labels).item() * labels.size(0)
                preds = torch.max(outputs, 1)[1]
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        return eval_loss / max(total, 1), correct / max(total, 1)


class FedMetaMetaSgdTrainer(FedMetaMamlTrainer):
    """FedMeta-Meta-SGD Client Trainer."""

    def __init__(self, alpha_state: Optional[Dict[str, torch.Tensor]] = None, **kwargs):
        super().__init__(**kwargs)
        self.alpha_state = alpha_state

    def train(self, model, trainloader, epochs, lr, device) -> float:
        sup_loader, qry_loader = split_support_query(trainloader, self.support_ratio)

        model.to(device)
        train_net = deepcopy(model).to(device)
        train_net.train()
        criterion = nn.CrossEntropyLoss()

        param_names = [k for k, v in model.named_parameters() if v.requires_grad]
        alphas: List[torch.Tensor] = []
        for name, param in zip(param_names, train_net.parameters()):
            if self.alpha_state and name in self.alpha_state:
                alphas.append(self.alpha_state[name].to(device))
            else:
                alphas.append(torch.full_like(param, self.alpha, device=device))

        # 1. Inner loop adaptation with learnable alpha vectors
        for _ in range(self.gradient_step):
            for batch in sup_loader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)

                if images.size(0) <= 1:
                    continue

                loss = criterion(train_net(images), labels)
                grads = torch.autograd.grad(
                    loss,
                    list(train_net.parameters()),
                    create_graph=True,
                    retain_graph=True,
                )

                for param, g, a in zip(train_net.parameters(), grads, alphas):
                    param.data = param.data - a * g

                for param in train_net.parameters():
                    if param.grad is not None:
                        param.grad.zero_()

        # 2. Outer loop evaluation on query loader
        loss_sum = torch.tensor(0.0, device=device)
        total_samples = 0
        for batch in qry_loader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            if images.size(0) <= 1:
                continue

            loss = criterion(train_net(images), labels)
            loss_sum = loss_sum + loss * labels.size(0)
            total_samples += labels.size(0)

        loss_avg = loss_sum / max(total_samples, 1)

        # 3. Outer loop meta-gradients
        meta_grads = torch.autograd.grad(loss_avg, list(train_net.parameters()))

        self.last_gradients = {}
        for name, g in zip(param_names, meta_grads):
            self.last_gradients[name] = g.detach().cpu()
            self.last_gradients[f"alpha.{name}"] = g.detach().cpu()

        for name, buf in model.named_buffers():
            self.last_gradients[name] = buf.detach().cpu()

        return float(loss_avg.detach().item())