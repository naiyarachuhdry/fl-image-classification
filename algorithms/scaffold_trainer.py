import torch
from .base_trainer import BaseTrainer

class ScaffoldTrainer(BaseTrainer):
    def __init__(self, server_cv=None, client_cv=None, momentum=0.0, weight_decay=0.0, **kwargs,):
        super().__init__(**kwargs)
        self.server_cv = server_cv
        self.client_cv = client_cv
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.model_delta = None
        self.client_cv_delta = None
        self.client_cv_new = None

    def compute_loss(self, model, outputs, labels, criterion):
        return criterion(outputs, labels)

    def train(self, model, trainloader, epochs, lr, device):
        if self.server_cv is None or self.client_cv is None:
            raise ValueError("SCAFFOLD requires server_cv and client_cv.")

        if lr <= 0:
            raise ValueError("Learning rate must be positive.")

        model.to(device)

        params = [
            p for p in model.parameters()
            if p.requires_grad
        ]

        if (
            len(params) != len(self.server_cv)
            or len(params) != len(self.client_cv)
        ):
            raise ValueError(
                "Control variates must match the trainable model parameters."
            )

        server_cv = [v.to(device) for v in self.server_cv]
        client_cv = [v.to(device) for v in self.client_cv]
        initial_params = [p.detach().clone() for p in params]

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            params,
            lr=lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        model.train()
        total_loss = 0.0
        num_batches = 0

        for _ in range(epochs):
            for batch in trainloader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # SCAFFOLD control-variate correction
                with torch.no_grad():
                    for param, server_c, client_c in zip(
                        params, server_cv, client_cv
                    ):
                        param.add_(
                            server_c - client_c,
                            alpha=-lr,
                        )

                total_loss += loss.item()
                num_batches += 1

        if num_batches == 0:
            raise ValueError("The training loader produced no batches.")

        # Local model update: y_i - x
        self.model_delta = [
            (param.detach() - initial).clone()
            for param, initial in zip(params, initial_params)
        ]

        # Updated client control variate
        self.client_cv_new = [
            (
                old_c
                - server_c
                + (initial - param.detach()) / (lr * num_batches)
            ).detach().clone()
            for old_c, server_c, initial, param in zip(
                client_cv, server_cv, initial_params, params
            )
        ]

        self.client_cv_delta = [
            (new_c - old_c).detach().clone()
            for new_c, old_c in zip(
                self.client_cv_new, client_cv
            )
        ]

        return total_loss / num_batches

    def get_updates(self):
        if self.model_delta is None or self.client_cv_delta is None:
            raise RuntimeError("Call train() before get_updates().")

        return self.model_delta, self.client_cv_delta

    def get_new_client_cv(self):
        if self.client_cv_new is None:
            raise RuntimeError(
                "Call train() before get_new_client_cv()."
            )

        return self.client_cv_new