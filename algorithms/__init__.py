from .fedavg_trainer import FedAvgTrainer
from .fedprox_trainer import FedProxTrainer
from .fedavgm_trainer import FedAvgMTrainer
from .fedadam_trainer import FedAdamTrainer
from .fedadagrad_trainer import FedAdagradTrainer
from .fedyogi_trainer import FedYogiTrainer
from .fedmedian_trainer import FedMedianTrainer
from .fednova_trainer import FedNovaTrainer
from .fedbn_trainer import FedBNTrainer
from .fedpara_trainer import FedParaTrainer
from .fedrep_trainer import FedRepTrainer
from .moon_trainer import MoonTrainer
from .fedmlb_trainer import FedMlbTrainer

TRAINER_REGISTRY = {
    "fedavg": FedAvgTrainer,
    "fedprox": FedProxTrainer,
    "fedavgm": FedAvgMTrainer,
    "fedadam": FedAdamTrainer,
    "fedadagrad": FedAdagradTrainer,
    "fedyogi": FedYogiTrainer,
    "fedmedian": FedMedianTrainer,
    "fednova": FedNovaTrainer,
    "fedbn": FedBNTrainer,
    "fedpara": FedParaTrainer,
    "fedrep": FedRepTrainer,
    "moon": MoonTrainer,
    "fedmlb": FedMlbTrainer,
}


def get_trainer(algorithm: str, **kwargs):
    try:
        trainer_cls = TRAINER_REGISTRY[algorithm]
    except KeyError:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Available: {list(TRAINER_REGISTRY)}"
        )
    return trainer_cls(**kwargs)