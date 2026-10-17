"""FedPFT client side: extract features once, fit per-class GMMs, return them."""
import numpy as np
import torch
import torch.nn as nn

from utils.fedpft_helper import fit_class_gmms, pack

from .base_trainer import BaseTrainer


class FedPFTTrainer(BaseTrainer):
    """No SGD on the client: the extractor is frozen, only GMMs are fitted."""

    def __init__(self, n_mixtures=1, cov_type="spherical", seed=0,
                 tol=1e-12, max_iter=1000, **kwargs):
        super().__init__(**kwargs)
        self.n_mixtures = n_mixtures
        self.cov_type = cov_type
        self.seed = seed
        self.tol = tol
        self.max_iter = max_iter
        self.gmm_arrays = []

    def compute_loss(self, model, outputs, labels, criterion):
        raise NotImplementedError("FedPFT clients do not train with a loss.")

    def train(self, model, trainloader, epochs, lr, device):
        """Extract features, fit one GMM per class and keep them in `gmm_arrays`.

        Same signature as BaseTrainer.train; `epochs` and `lr` are not used.
        """
        features, labels = self._extract_features(model, trainloader, device)
        gmms = fit_class_gmms(
            features, labels,
            n_mixtures=self.n_mixtures, cov_type=self.cov_type,
            seed=self.seed, tol=self.tol, max_iter=self.max_iter,
        )
        self.gmm_arrays = pack(gmms)
        return 0.0  # no loss: nothing is trained locally

    def get_metrics(self):
        return {"num-gmms": len(self.gmm_arrays) // 5}

    @torch.no_grad()
    def _extract_features(self, model, trainloader, device):
        head, model.fc = model.fc, nn.Identity()  # without the classifier, output = features
        model.to(device)
        model.eval()
        feats, labels = [], []
        for batch in trainloader:
            feats.append(model(batch["img"].to(device)).cpu().numpy())
            labels.append(batch["label"].numpy())
        model.fc = head
        return np.concatenate(feats), np.concatenate(labels)