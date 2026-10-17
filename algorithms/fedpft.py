"""FedPFT server strategy (Flower message API).

Clients send per-class GMMs once; the server samples synthetic features,
trains a linear classifier head on them and returns the head's weights.
"""
from collections.abc import Iterable
from logging import INFO
import torch
from flwr.app import ArrayRecord, Message, MetricRecord
from flwr.common.logger import log
from flwr.serverapp.strategy import FedAvg
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from utils.fedpft_helper import build_synthetic_dataset, unpack


def train_head(features, labels, feature_dim, num_classes, epochs, lr,
               batch_size, device, seed=0):
    """Train a linear head on (synthetic) features.

    Returns (trained head, accuracy of the last epoch).
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=gen)

    head = nn.Linear(feature_dim, num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    acc = 0.0
    for epoch in range(epochs):
        correct, total = 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = head(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            correct += (out.argmax(dim=1) == yb).sum().item()
            total += yb.shape[0]
        acc = correct / max(total, 1)
        log(INFO, "FedPFT head: epoch %d/%d, synthetic accuracy %.4f",
            epoch + 1, epochs, acc)
    return head, acc


class FedPFT(FedAvg):
    """Server side of FedPFT (https://arxiv.org/abs/2402.01862)."""

    def __init__(self, *args, feature_dim, num_classes, cov_type="spherical",
                 seed=0, head_epochs=50, head_lr=1e-3, head_batch_size=256,
                 device=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.cov_type = cov_type
        self.seed = seed
        self.head_epochs = head_epochs
        self.head_lr = head_lr
        self.head_batch_size = head_batch_size
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        """Turn the clients' GMMs into a trained classifier head."""
        replies = list(replies)
        valid = [m for m in replies if not m.has_error()]
        log(INFO, "aggregate_train: %d results, %d failures",
            len(valid), len(replies) - len(valid))
        if not valid:
            return None, None

        params = []
        for msg in valid:
            arrays = msg.content["arrays"].to_numpy_ndarrays()
            params.extend(unpack(arrays))
        if not params:
            return None, None

        features, labels = build_synthetic_dataset(
            params, cov_type=self.cov_type, seed=self.seed
        )
        log(INFO, "FedPFT: %d GMMs -> %d synthetic features of dim %d",
            len(params), features.shape[0], features.shape[1])

        head, acc = train_head(
            features, labels, self.feature_dim, self.num_classes,
            self.head_epochs, self.head_lr, self.head_batch_size,
            self.device, self.seed,
        )
        state = {f"fc.{k}": v for k, v in head.state_dict().items()}
        metrics = MetricRecord({
            "num-gmms": len(params),
            "synthetic-samples": int(features.shape[0]),
            "synthetic-accuracy": float(acc),
        })
        return ArrayRecord(state), metrics