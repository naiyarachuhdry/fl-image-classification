"""Generic MOON helpers for any model ending in a Linear classifier."""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoonProjectionHead(nn.Module):
    """Projection MLP used for MOON's contrastive objective."""

    def __init__(self, input_dim: int, projection_dim: int):
        super().__init__()
        self.l1 = nn.Linear(input_dim, input_dim)
        self.l2 = nn.Linear(input_dim, projection_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.relu(self.l1(features))
        return self.l2(features)


def _find_last_classifier(model: nn.Module) -> Tuple[str, nn.Linear]:
    """Find the final Linear layer of the model."""
    linear_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and not name.startswith("moon_projector.")
    ]
    if not linear_layers:
        raise ValueError("MOON requires a model containing at least one nn.Linear layer.")
    return linear_layers[-1]


def prepare_moon_model(model: nn.Module, projection_dim: int) -> nn.Module:
    """Attach the MOON projection head to an existing model.
    The projection head becomes part of model.state_dict(), so Flower sends,
    aggregates, and restores it automatically.
    """
    if hasattr(model, "moon_projector"):
        return model

    classifier_name, classifier = _find_last_classifier(model)
    model.add_module(
        "moon_projector",
        MoonProjectionHead(
            input_dim=classifier.in_features,
            projection_dim=projection_dim,
        ),
    )
    model._moon_classifier_name = classifier_name
    return model


class MoonFeatureForward:
    """Captures the features passed into the final classifier during forward()."""

    def __init__(self, model: nn.Module):
        if not hasattr(model, "moon_projector"):
            raise ValueError(
                "Call prepare_moon_model(model, projection_dim) before creating MoonFeatureForward."
            )
        self.model = model
        self.features = None
        classifier_name = model._moon_classifier_name
        modules = dict(model.named_modules())
        classifier = modules[classifier_name]
        self.hook = classifier.register_forward_pre_hook(self._capture_features)

    def _capture_features(self, _module, inputs) -> None:
        features = inputs[0]
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        self.features = features

    def forward(self, images: torch.Tensor):
        """Return representation, projected representation, and logits."""
        self.features = None
        logits = self.model(images)
        if self.features is None:
            raise RuntimeError("MOON could not capture classifier input features.")
        projection = self.model.moon_projector(self.features)
        return self.features, projection, logits

    def close(self) -> None:
        """Remove the registered PyTorch hook."""
        self.hook.remove()