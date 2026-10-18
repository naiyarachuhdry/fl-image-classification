from flwr.serverapp.strategy import FedAvg, FedProx, FedAvgM, FedAdam, FedAdagrad, FedYogi, FedMedian
from .fednova import FedNova
from .fedmeta import FedMeta
from .floco import Floco 
from .scaffold import Scaffold

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
    "fedmlb": FedAvg,
    "fedper": FedAvg,
    "floco": Floco,
    "scaffold": Scaffold,
    # --- FedMeta Algorithms ---
    "fedavg_meta": FedAvg,      
    "fedmeta_maml": FedMeta,    
    "fedmeta_meta_sgd": FedMeta,
}

def get_strategy(algorithm: str, **kwargs):
    try:
        strategy_cls = STRATEGY_REGISTRY[algorithm]
    except KeyError:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Available: {list(STRATEGY_REGISTRY)}"
        )
    return strategy_cls(**kwargs)