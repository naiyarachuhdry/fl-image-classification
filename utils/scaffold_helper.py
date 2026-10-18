import tomllib
from pathlib import Path

def get_num_supernodes():
    # Resolve config.toml relative to the repository root.
    config_path = Path.home() / ".flwr" / "config.toml"

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Flower config.toml not found at: {config_path}"
        )

    with config_path.open("rb") as file:
        config = tomllib.load(file)

    active_profile = config["superlink"]["default"]
    profile = config["superlink"][active_profile]

    try:
        return int(profile["options"]["num-supernodes"])
    except KeyError as exc:
        raise ValueError(
            f"The active profile '{active_profile}' does not define "
            "'options.num-supernodes' in {config_path}."
        ) from exc


def get_trainable_parameter_names(model):
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

