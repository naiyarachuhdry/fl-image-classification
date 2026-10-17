import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from task import load_centralized_dataset
from utils.metrics import evaluate_model

from algorithms.server_strategies import get_strategy

from models import create_model

from utils.fedbn_helper import get_fedbn_state_dict
from utils.fedpara_helper import convert_to_fedpara
from utils.fedrep_helper import get_fedrep_shared_state_dict
from utils.moon_helper import prepare_moon_model
from utils.fedper_helper import get_fedper_shared_state_dict
from utils.floco_helper import convert_to_floco

# Create ServerApp
app = ServerApp()
model_name = None
algorithm = None
moon_projection_dim = None
floco_endpoints = None
@app.main()

def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    global model_name, algorithm, moon_projection_dim, floco_endpoints
    algorithm = context.run_config["algorithm"]
    moon_projection_dim = context.run_config.get("moon-projection-dim", 84)
    floco_endpoints = context.run_config.get("endpoints", 3)

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Model selected from config
    model_name = context.run_config["model_name"]

    # Load global model
    global_model = create_model(model_name)

    if algorithm == "fedpara":
        convert_to_fedpara(global_model) 

    elif algorithm == "moon":
        prepare_moon_model(global_model, projection_dim=moon_projection_dim)

    elif algorithm == "floco":
        seed = context.run_config.get("seed", 0)
        global_model = convert_to_floco(global_model, endpoints=floco_endpoints, seed=seed)   

    if algorithm == "fedbn":
        arrays = ArrayRecord(get_fedbn_state_dict(global_model))

    elif algorithm == "fedrep":
        arrays = ArrayRecord(get_fedrep_shared_state_dict(global_model))

    elif algorithm == "fedper":
        arrays = ArrayRecord(get_fedper_shared_state_dict(global_model))
    
    else:
        arrays = ArrayRecord(global_model.state_dict())

    # Build kwargs relevant to whichever strategy is picked;
    # extras are ignored by strategies that don't use them
    strategy_kwargs = {
        "fraction_evaluate": context.run_config["fraction-evaluate"],
        "min_available_nodes": context.run_config["min-available-clients"],
    }

    if algorithm == "fedprox":
        strategy_kwargs["proximal_mu"] = context.run_config["proximal-mu"]

    elif algorithm == "fedavgm":
        strategy_kwargs["server_learning_rate"] = context.run_config["server-learning-rate"]
        strategy_kwargs["server_momentum"] = context.run_config["server-momentum"]

    elif algorithm in ("fedadam", "fedyogi"):
        strategy_kwargs["eta"] = context.run_config["server-learning-rate"]
        strategy_kwargs["eta_l"] = context.run_config["learning-rate"]
        strategy_kwargs["beta_1"] = context.run_config["beta-1"]
        strategy_kwargs["beta_2"] = context.run_config["beta-2"]
        strategy_kwargs["tau"] = context.run_config["tau"]

    elif algorithm == "fedadagrad":
        strategy_kwargs["eta"] = context.run_config["server-learning-rate"]
        strategy_kwargs["eta_l"] = context.run_config["learning-rate"]
        strategy_kwargs["tau"] = context.run_config["tau"]

    elif algorithm == "fednova":
        strategy_kwargs["server_learning_rate"] = context.run_config.get("server-learning-rate")
        strategy_kwargs["server_momentum"] = context.run_config.get("server-momentum")
        
    elif algorithm in ("fedmeta_maml", "fedmeta_meta_sgd"):
        strategy_kwargs["algo"] = algorithm
        strategy_kwargs["model_name"] = model_name
        strategy_kwargs["beta"] = context.run_config.get("fedmeta-beta")
        strategy_kwargs["alpha_init"] = context.run_config.get("fedmeta-alpha")

    elif algorithm == "floco":
        strategy_kwargs["endpoints"] = context.run_config.get("endpoints")
        strategy_kwargs["tau"] = context.run_config.get("tau")
        strategy_kwargs["rho"] = context.run_config.get("rho")
        strategy_kwargs["fraction_fit"] = context.run_config.get("fraction-fit")

    strategy = get_strategy(algorithm, **strategy_kwargs)

    if algorithm in ("fedbn", "fedrep", "fedper"):
        evaluate_fn = None
    else:
        evaluate_fn = global_evaluate

    # Start strategy, run for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=evaluate_fn,
    )

    if context.run_config["save-model"]:
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()

        if algorithm in ("fedper", "fedrep"):
            torch.save(state_dict, f"{algorithm}_shared_state.pt")
            print(f"Saved {algorithm} shared parameters only. Client-specific heads are not included.")
        else:
            torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
   
    model = create_model(model_name)  

    if algorithm == "fedpara":
        convert_to_fedpara(model)

    elif algorithm == "moon":
        prepare_moon_model(model, projection_dim=moon_projection_dim)

    elif algorithm == "floco":
        seed = 0
        endpoints = floco_endpoints or 3
        model = convert_to_floco(model, endpoints=floco_endpoints, seed=seed)

    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = evaluate_model(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
