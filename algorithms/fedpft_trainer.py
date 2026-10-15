"""FedPFT client side: extract features once, fit per-class GMMs, return them."""
import numpy as np
import torch

from utils.fedpft_helper import fit_class_gmms, pack

from .base_trainer import BaseTrainer


class FedPFTTrainer(BaseTrainer):
    """No SGD on the client: the extractor is frozen, only GMMs are fitted."""

    def __init__(self, n_mixtures=1, cov_type="spherical", seed=0,
                 tol=1e-12, max_iter=10000, **kwargs):
        super().__init__(**kwargs)
        self.n_mixtures = n_mixtures
        self.cov_type = cov_type
        self.seed = seed
        self.tol = tol
        self.max_iter = max_iter

    def compute_loss(self, model, outputs, labels, criterion):
        raise NotImplementedError("FedPFT clients do not train with a loss.")

    def extract_and_fit(self, extractor, trainloader, device):
        """Return (flat list of GMM arrays, number of local samples)."""
        extractor.to(device)
        extractor.eval()
        feats, labs = [], []
        with torch.no_grad():
            for batch in trainloader:
                x = batch["img"].to(device)
                feats.append(extractor(x).cpu().numpy())
                labs.append(batch["label"].cpu().numpy())
        features = np.concatenate(feats)
        labels = np.concatenate(labs)

        gmms = fit_class_gmms(
            features, labels,
            n_mixtures=self.n_mixtures, cov_type=self.cov_type,
            seed=self.seed, tol=self.tol, max_iter=self.max_iter,
        )
        return pack(gmms), len(labels)