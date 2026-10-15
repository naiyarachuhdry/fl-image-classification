from flwr.serverapp.strategy import FedAvg, FedProx, FedAvgM, FedAdam, FedAdagrad, FedYogi, FedMedian
from .fednova import FedNova
from .fedpft import FedPFT

STRATEGY_REGISTRY = {
    "fedavg": FedAvg,
    "fedprox": FedProx,
    "fedavgm": FedAvgM,
    "fedadam": FedAdam,
    "fedadagrad": FedAdagrad,
    "fedyogi": FedYogi,
    "fedmedian": FedMedian,
    "fednova": FedNova,
    "fedbn": FedAvg,
    "fedpara": FedAvg,
    "fedrep": FedAvg,
    "moon": FedAvg,
    "fedpft": FedPFT,
}


def get_strategy(algorithm: str, **kwargs):
    try:
        strategy_cls = STRATEGY_REGISTRY[algorithm]
    except KeyError:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Available: {list(STRATEGY_REGISTRY)}"
        )
    return strategy_cls(**kwargs)