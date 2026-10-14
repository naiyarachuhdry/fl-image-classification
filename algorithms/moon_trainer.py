"""MOON client-side local training."""

from __future__ import annotations

import copy
from typing import Mapping, Optional
import torch
import torch.nn.functional as F

from utils.moon_helper import MoonFeatureForward
from .base_trainer import BaseTrainer


class MoonTrainer(BaseTrainer):
    """Model-Contrastive Federated Learning trainer."""

    def __init__(
        self,
        mu: float = 1.0,
        temperature: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if temperature <= 0:
            raise ValueError("MOON temperature must be greater than zero.")
        self.mu = mu
        self.temperature = temperature

    def compute_loss(self, model, outputs, labels, criterion):
        return criterion(outputs, labels)

    def train(
        self,
        model: torch.nn.Module,
        trainloader,
        epochs: int,
        lr: float,
        device: torch.device,
        previous_state_dict: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> float:
        """Train with classification loss plus the MOON contrastive loss."""
        global_model = copy.deepcopy(model)
        previous_model = copy.deepcopy(model)

        if previous_state_dict is not None:
            previous_model.load_state_dict(previous_state_dict, strict=True)

        model.to(device)
        global_model.to(device)
        previous_model.to(device)

        model.train()
        global_model.eval()
        previous_model.eval()

        for parameter in global_model.parameters():
            parameter.requires_grad_(False)

        for parameter in previous_model.parameters():
            parameter.requires_grad_(False)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-5,
        )
        criterion = torch.nn.CrossEntropyLoss().to(device)

        local_forward = MoonFeatureForward(model)
        global_forward = MoonFeatureForward(global_model)
        previous_forward = MoonFeatureForward(previous_model)

        total_loss = 0.0
        num_batches = 0

        try:
            for _ in range(epochs):
                for batch in trainloader:
                    images = batch["img"].to(device)
                    labels = batch["label"].long().to(device)

                    optimizer.zero_grad()

                    _, local_projection, logits = local_forward.forward(images)

                    with torch.no_grad():
                        _, global_projection, _ = global_forward.forward(images)
                        _, previous_projection, _ = previous_forward.forward(images)

                    pos_sim = F.cosine_similarity(local_projection, global_projection, dim=-1)
                    neg_sim = F.cosine_similarity(local_projection, previous_projection, dim=-1)

                    contrastive_logits = torch.stack((pos_sim, neg_sim), dim=1) / self.temperature
                    contrastive_targets = torch.zeros(images.size(0), dtype=torch.long, device=device)

                    loss_ce = criterion(logits, labels)
                    loss_con = criterion(contrastive_logits, contrastive_targets)

                    loss = loss_ce + self.mu * loss_con
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1
        finally:
            local_forward.close()
            global_forward.close()
            previous_forward.close()

        return total_loss / max(num_batches, 1)