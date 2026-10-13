import torch
import torch.nn as nn
from .base_trainer import BaseTrainer

class FedRepTrainer(BaseTrainer):
    def __init__(self, head_module_name="fc3", head_epochs=1, **kwargs):
        super().__init__(**kwargs)
        self.head_module_name = head_module_name
        self.head_epochs = head_epochs
        self.last_metrics = {}

    def get_metrics(self):
        return self.last_metrics

    def compute_loss(self, model, outputs, labels, criterion):
        return criterion(outputs, labels)

    def train(self, model, trainloader, local_epochs, lr, device):
        model.to(device)

        modules = dict(model.named_modules())
        if self.head_module_name not in modules:
            raise ValueError(
                f"Head module '{self.head_module_name}' "
                "was not found in the model."
            )

        head = modules[self.head_module_name]

        head_params = list(head.parameters())
        head_param_ids = {id(param) for param in head_params}
        rep_params = [
            param
            for param in model.parameters()
            if id(param) not in head_param_ids
        ]

        if not head_params or not rep_params:
            raise ValueError(
                "FedRep requires non-empty head and representation parameters."
            )

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_batches = 0

        def run_phase(params, epochs, train_head):
            nonlocal total_loss, total_batches

            for param in model.parameters():
                param.requires_grad_(False)

            for param in params:
                param.requires_grad_(True)

            optimizer = torch.optim.SGD(params, lr=lr)

            for _ in range(epochs):
                # Freeze the inactive part's training-mode behavior too.
                model.eval()
                if train_head:
                    head.train()
                else:
                    model.train()
                    head.eval()

                for batch in trainloader:
                    images = batch["img"].to(device)
                    labels = batch["label"].to(device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total_batches += 1

        # Phase 1: train the local head, freeze the representation.
        run_phase(head_params, self.head_epochs, train_head=True)

        # Phase 2: train the shared representation, freeze the local head.
        run_phase(rep_params, local_epochs, train_head=False)

        # Restore normal parameter behavior for subsequent operations.
        for param in model.parameters():
            param.requires_grad_(True)

        model.train()

        average_loss = total_loss / max(total_batches, 1)
        self.last_metrics = {"train_loss": average_loss}
        return average_loss
