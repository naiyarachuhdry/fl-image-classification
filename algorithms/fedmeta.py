"""FedMeta Server Strategy (FedMeta-MAML & FedMeta-Meta-SGD).

Aggregates client meta-gradients and updates global parameters (and learnable alpha
for Meta-SGD) using a server-side Adam optimizer matching the Flower FedMeta baseline.
"""

from typing import Dict, Optional
import torch
from flwr.app import Array, ArrayRecord
from flwr.serverapp.strategy import FedAvg

from models import create_model


class FedMeta(FedAvg):
    """Server strategy for FedMeta (MAML and Meta-SGD)."""

    def __init__(
        self,
        algo: str = "fedmeta_maml",
        model_name: str = "resnet18_mlb",
        beta: float = 0.001,
        alpha_init: float = 0.01,
        weight_decay: float = 0.001,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.algo = algo
        self.model_name = model_name
        self.beta = float(beta)
        self.alpha_init = float(alpha_init)
        self.weight_decay = float(weight_decay)

        # Server-side model instance to hold Adam optimizer states across rounds
        self.server_model = create_model(model_name)
        self.alpha_dict: Optional[Dict[str, torch.Tensor]] = None

        if self.algo == "fedmeta_meta_sgd":
            self.alpha_dict = {
                name: torch.full_like(param, self.alpha_init)
                for name, param in self.server_model.named_parameters()
                if param.requires_grad
            }
            trainable_params = list(self.server_model.parameters()) + list(self.alpha_dict.values())
        else:
            trainable_params = list(self.server_model.parameters())

        self.server_optimizer = torch.optim.Adam(
            trainable_params,
            lr=self.beta,
            weight_decay=self.weight_decay,
        )

    def configure_train(self, server_round, arrays, config, grid):
        """Send global model (and alpha vectors if Meta-SGD) to clients."""
        self.current_arrays = arrays

        if self.algo == "fedmeta_meta_sgd" and self.alpha_dict is not None:
            combined = dict(arrays)
            for k, v in self.alpha_dict.items():
                combined[f"alpha.{k}"] = Array(v.cpu().numpy())
            arrays = ArrayRecord(combined)

        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        """Aggregate client meta-gradients and update global parameters via Adam."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True, validate=False)
        if not valid_replies:
            return None, None

        # Collect sample count weights n_i
        weights = [float(msg.content["metrics"].get(self.weighted_by_key, 1.0)) for msg in valid_replies]
        total_w = sum(weights) or 1.0

        # Collect client meta-gradient dictionaries
        client_state_dicts = [msg.content["arrays"].to_torch_state_dict() for msg in valid_replies]

        # 1. Weighted gradient aggregation across clients
        first_dict = client_state_dicts[0]
        aggregated_grads: Dict[str, torch.Tensor] = {}

        for key in first_dict.keys():
            if first_dict[key].is_floating_point():
                aggregated_grads[key] = sum(
                    (w / total_w) * client_dict[key].float()
                    for w, client_dict in zip(weights, client_state_dicts)
                )
            else:
                aggregated_grads[key] = first_dict[key]

        # 2. Update server model parameters using Adam
        self.server_optimizer.zero_grad()

        for name, param in self.server_model.named_parameters():
            if param.requires_grad and name in aggregated_grads:
                param.grad = aggregated_grads[name].to(param.device, param.dtype)

        # 3. For Meta-SGD, also update alpha step-sizes using Adam
        if self.algo == "fedmeta_meta_sgd" and self.alpha_dict is not None:
            for name, alpha_tensor in self.alpha_dict.items():
                grad_key = f"alpha.{name}"
                if grad_key in aggregated_grads:
                    alpha_tensor.grad = aggregated_grads[grad_key].to(alpha_tensor.device, alpha_tensor.dtype)

        self.server_optimizer.step()
        self.server_optimizer.zero_grad()

        # Save updated parameters back into current_arrays
        new_state = self.server_model.state_dict()
        self.current_arrays = ArrayRecord({k: Array(v.cpu().numpy()) for k, v in new_state.items()})

        metrics = self.train_metrics_aggr_fn([msg.content for msg in valid_replies], self.weighted_by_key)
        return self.current_arrays, metrics