import torch
import torch.nn.functional as F

from .base_trainer import BaseTrainer


class FedMlbTrainer(BaseTrainer):
    """FedMLB local training with intermediate-branch supervision."""

    def __init__(
        self,
        global_model=None,
        lambda_1=1.0,
        lambda_2=1.0,
        temperature=1.0,
        weight_decay=1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.global_model = global_model
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.temperature = temperature
        self.weight_decay = weight_decay

    def compute_loss(self, model, outputs, labels, criterion):
        raise RuntimeError("FedMLB requires its custom train() method.")

    def train(self, model, trainloader, epochs, lr, device, global_model=None):
        teacher = global_model or self.global_model

        if teacher is None:
            raise ValueError(
                "FedMLB requires the received global model as its teacher."
            )

        model.to(device)
        model.train()

        teacher.to(device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=self.weight_decay
        )

        total_loss = 0.0
        num_batches = 0

        for _ in range(epochs):
            for batch in trainloader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()

                # Local model: extract intermediate features and final logits
                local_features = model(images, return_features=True)
                local_logits = local_features[-1]

                # 1. Local classification cross-entropy loss
                classification_loss = criterion(local_logits, labels)

                # Precompute student log-probabilities once for all 5 branches
                local_log_probs = F.log_softmax(
                    local_logits / self.temperature, dim=1
                )

                branch_ce_losses = []
                branch_kd_losses = []

                # Five intermediate feature levels: out0 ... out4
                for branch in range(5):
                    local_feature = local_features[branch]

                    # Teacher suffix predicts from the local feature at level (branch + 1)
                    teacher_logits = teacher(local_feature, level=branch + 1)

                    branch_ce_losses.append(criterion(teacher_logits, labels))

                    teacher_probs = F.softmax(
                        teacher_logits.detach() / self.temperature, dim=1
                    )

                    branch_kd_losses.append(
                        (self.temperature ** 2)
                        * F.kl_div(
                            local_log_probs,
                            teacher_probs,
                            reduction="batchmean",
                        )
                    )

                branch_ce = torch.stack(branch_ce_losses).mean()
                branch_kd = torch.stack(branch_kd_losses).mean()

                loss = (
                    classification_loss
                    + self.lambda_1 * branch_ce
                    + self.lambda_2 * branch_kd
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                # Clean release of intermediate tensors (prevents CPU memory accumulation)
                del local_features, teacher_logits, branch_ce_losses, branch_kd_losses, loss

        return total_loss / max(num_batches, 1)