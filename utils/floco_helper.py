"""FLOCo: Federated Learning Optimization with Continuous Simplex Classifiers.

Based on: 'FLOCo: Federated Learning of Continuous Subspaces'
Provides Simplex Linear layer, Simplex Model wrapper, and client PCA projection.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sklearn import decomposition
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def _initialize_weight(init_weight: torch.Tensor, seed: int) -> torch.nn.Parameter:
    """Initialize a weight tensor with Xavier normal initialization."""
    weight = torch.nn.Parameter(torch.zeros_like(init_weight))
    torch.manual_seed(seed)
    torch.nn.init.xavier_normal_(weight)
    return weight


class SimplexLinear(torch.nn.Linear):
    """A fully-connected continuous simplex classifier layer.
    
    Contains `endpoints` parameter matrices. The forward pass computes:
        W = sum_{k=0}^{K-1} alpha_k * W_k
    """

    def __init__(self, endpoints: int, seed: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.endpoints = int(endpoints or 3)
        self.alphas = [1.0 / self.endpoints for _ in range(self.endpoints)]
        self._weights = torch.nn.ParameterList(
            [_initialize_weight(self.weight, seed + i) for i in range(self.endpoints)]
        )

    @property
    def weight(self) -> torch.nn.Parameter:
        """Return the weighted sum of the simplex endpoint weights."""
        return sum(alpha * w for alpha, w in zip(self.alphas, self._weights))


def _sample_l1_ball(center: np.ndarray, radius: float) -> np.ndarray:
    """Sample uniformly from the L1 ball around center."""
    u = np.random.uniform(-1, 1, len(center))
    norm_u = np.sum(np.abs(u))
    if norm_u > 0:
        u = np.sign(u) * (np.abs(u) / norm_u)
    return center + np.random.uniform(0, radius) * u


class SimplexModel(nn.Module):
    """Wrapper that turns any base PyTorch model into a FLOCo simplex model."""

    def __init__(self, base_model: nn.Module, endpoints: int = 3, seed: int = 0):
        super().__init__()
        self.base_model = base_model
        self.endpoints = int(endpoints or 3)
        self.seed = seed
        self.subregion_parameters: Optional[Tuple[np.ndarray, float]] = None

        # Find and replace the final linear classification layer
        parent, attr_name, last_linear = self._find_last_linear(base_model)
        self.simplex_layer = SimplexLinear(
            endpoints=self.endpoints,
            seed=seed,
            in_features=last_linear.in_features,
            out_features=last_linear.out_features,
            bias=(last_linear.bias is not None),
        )
        setattr(parent, attr_name, self.simplex_layer)

    @staticmethod
    def _find_last_linear(model: nn.Module) -> Tuple[nn.Module, str, nn.Linear]:
        """Find the final linear classification layer across any architecture."""
        last_parent = model
        last_attr = None
        last_module = None

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if "." in name:
                    parent_name, attr = name.rsplit(".", 1)
                    last_parent = model.get_submodule(parent_name)
                    last_attr = attr
                else:
                    last_parent = model
                    last_attr = name
                last_module = module

        if last_module is None:
            raise ValueError("Model does not contain any nn.Linear layer for classification.")
        return last_parent, last_attr, last_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with stochastic alpha sampling."""
        if self.subregion_parameters is None:  # Before projection
            if self.training:
                # Sample uniformly on the unit simplex (Dirichlet distribution)
                sample = np.random.exponential(scale=1.0, size=self.endpoints)
                self.simplex_layer.alphas = tuple(sample / sample.sum())
            else:
                # Simplex center for evaluation
                self.simplex_layer.alphas = tuple([1.0 / self.endpoints for _ in range(self.endpoints)])
        else:  # After projection
            center, radius = self.subregion_parameters
            if self.training:
                # Sample within client's specific L1 subregion
                self.simplex_layer.alphas = tuple(_sample_l1_ball(center, radius))
            else:
                # Subregion center for evaluation
                self.simplex_layer.alphas = tuple(center)

        return self.base_model(x)


def convert_to_floco(model: nn.Module, endpoints: int = 3, seed: int = 0) -> SimplexModel:
    """Convenience converter matching project helper pattern (fedpara, moon)."""
    if isinstance(model, SimplexModel):
        return model
    return SimplexModel(base_model=model, endpoints=endpoints, seed=seed)


# ============================================================================
# PCA Projection and Simplex Optimization (Server-Side)
# ============================================================================

def project_clients(client_gradients: Dict[int, np.ndarray], endpoints: int) -> np.ndarray:
    """Optimize client projection onto a simplex of dimension (endpoints - 1)."""
    sorted_keys = sorted(client_gradients.keys())
    client_stats = np.array([client_gradients[k] for k in sorted_keys])

    if HAS_SKLEARN:
        kappas = decomposition.PCA(n_components=endpoints).fit_transform(client_stats)
    else:
        # Exact mathematical SVD fallback if scikit-learn is not installed
        mean = np.mean(client_stats, axis=0)
        centered = client_stats - mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        kappas = np.dot(centered, vt[:endpoints].T)

    lowest_log_energy = np.inf
    best_beta = None

    for z in np.linspace(1e-4, 1, 1000):
        betas = _project_client_onto_simplex(kappas, z=z)
        row_sums = betas.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        betas /= row_sums
        log_energy = _riesz_s_energy(betas)
        if log_energy not in [-np.inf, np.inf] and log_energy < lowest_log_energy:
            lowest_log_energy = log_energy
            best_beta = betas

    if best_beta is None:
        best_beta = np.ones((len(client_stats), endpoints)) / endpoints

    return best_beta


def _project_client_onto_simplex(kappas: np.ndarray, z: float) -> np.ndarray:
    """Project points onto the simplex using sorting and water-filling."""
    sorted_kappas = np.sort(kappas, axis=1)[:, ::-1]
    z_vec = np.ones(len(kappas)) * z
    cssv = np.cumsum(sorted_kappas, axis=1) - z_vec[:, np.newaxis]
    ind = np.arange(kappas.shape[1]) + 1
    cond = sorted_kappas - cssv / ind > 0
    nonzero = np.count_nonzero(cond, axis=1)
    normalized_kappas = cssv[np.arange(len(kappas)), nonzero - 1] / nonzero
    betas = np.maximum(kappas - normalized_kappas[:, np.newaxis], 0)
    return betas


def _riesz_s_energy(simplex_points: np.ndarray) -> float:
    """Compute Riesz s-energy of client projections."""
    diff = simplex_points[:, None] - simplex_points[None, :]
    distance = np.sqrt((diff**2).sum(axis=2))
    np.fill_diagonal(distance, np.inf)
    epsilon = 1e-4
    distance[distance < epsilon] = epsilon
    mutual_dist = distance[np.triu_indices(len(simplex_points), 1)]
    mutual_dist[np.argwhere(mutual_dist == 0).flatten()] = epsilon
    energies = 1 / (mutual_dist**2)
    valid_energies = energies[~np.isnan(energies)]
    if len(valid_energies) == 0:
        return np.inf
    energy = valid_energies.sum()
    log_energy = -np.log(len(mutual_dist)) + np.log(max(energy, 1e-12))
    return float(log_energy)