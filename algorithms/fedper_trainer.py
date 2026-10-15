from .base_trainer import BaseTrainer

class FedPerTrainer(BaseTrainer):
    """Train the full model locally; the client keeps its head private."""

    def compute_loss(self, model, outputs, labels, criterion):
        return criterion(outputs, labels)
