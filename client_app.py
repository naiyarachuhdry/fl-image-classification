"""pytorchexample: A Flower / PyTorch app."""

import torch
import json
import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from task import load_data
from utils.metrics import evaluate_model

from algorithms import get_trainer

from models import create_model

from utils.fedbn_helper import get_fedbn_state_dict, get_bn_state_dict, load_bn_state_dict
from utils.fedpara_helper import convert_to_fedpara
from utils.fedrep_helper import get_fedrep_head_state_dict, get_fedrep_shared_state_dict, load_fedrep_head_state_dict
from utils.moon_helper import prepare_moon_model
from utils.fedper_helper import get_fedper_head_state_dict, get_fedper_shared_state_dict, load_fedper_head_state_dict
from utils.floco_helper import convert_to_floco

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    algorithm = context.run_config["algorithm"]
    model_name = context.run_config["model_name"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global_model = None
    
    # Create model
    model = create_model(model_name) 
    
    if algorithm == "fedpara":
        convert_to_fedpara(model)

    elif algorithm == "moon":
        prepare_moon_model(
            model,
            projection_dim=context.run_config.get("moon-projection-dim", 84),
        )

    elif algorithm == "floco":
        endpoints = context.run_config.get("endpoints", 3)
        seed = context.run_config.get("seed", 0)
        model = convert_to_floco(model, endpoints=endpoints, seed=seed)

    # Load received parameters
    received_state= msg.content["arrays"].to_torch_state_dict()

    scaffold_parameter_names = []
    scaffold_server_cv = None
    scaffold_client_cv = None

    if algorithm == "scaffold":
        server_cv_prefix = "__scaffold_server_cv__"

        scaffold_parameter_names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]

        # Extract server control variates from the received arrays.
        scaffold_server_cv = []

        for name in scaffold_parameter_names:
            cv_key = server_cv_prefix + name

            if cv_key not in received_state:
                raise ValueError(
                    f"Missing server control variate: {cv_key}"
                )

            scaffold_server_cv.append(
                received_state.pop(cv_key)
            )

        # Restore this client's control variates from its persistent state.
        if "scaffold_client_cv" in context.state:
            saved_cv = (
                context.state["scaffold_client_cv"]
                .to_torch_state_dict()
            )

            scaffold_client_cv = [
                saved_cv[name]
                for name in scaffold_parameter_names
            ]

        else:
            # First participation: initialize client control variates to zero.
            named_parameters = dict(model.named_parameters())

            scaffold_client_cv = [
                torch.zeros_like(named_parameters[name].detach())
                for name in scaffold_parameter_names
            ]

    # FedMLB: create a teacher from the received global parameters.
    elif algorithm == "fedmlb":
        global_model = create_model(model_name)
        global_model.load_state_dict(received_state)
        global_model.to(device)
        global_model.eval()
        for parameter in global_model.parameters():
            parameter.requires_grad_(False)

    alpha_state = None
    if algorithm == "fedmeta_meta_sgd":
        alpha_state = {k[6:]: v for k, v in received_state.items() if k.startswith("alpha.")}
        model_state = {k: v for k, v in received_state.items() if not k.startswith("alpha.")}
        model.load_state_dict(model_state)

    elif algorithm == "fedbn":
        # Load shared/global parameters
        model.load_state_dict(received_state, strict=False,)

        # Restore this client's local BatchNorm state
        if "fedbn_bn_state" in context.state:
            print("Restoring local BN state")
            bn_state = context.state["fedbn_bn_state"].to_torch_state_dict()
            load_bn_state_dict(model, bn_state)
        else:
            print("No previous BN state")

    elif algorithm == "fedrep":
        model.load_state_dict(received_state, strict=False)

        if "fedrep_head_state" in context.state:
            head_state = context.state["fedrep_head_state"].to_torch_state_dict()
            load_fedrep_head_state_dict(model, head_state)
            print(f"[FedRep] Client {context.node_config['partition-id']}: local head RESTORED")
            
        else:
           print(f"[FedRep] Client {context.node_config['partition-id']}: no previous head; using initialized head")  

    
    elif algorithm == "fedper":
        model.load_state_dict(received_state, strict=False)

        if "fedper_head_state" in context.state:
            head_state = context.state["fedper_head_state"].to_torch_state_dict()
            load_fedper_head_state_dict(model, head_state)
            print(f"[FedPer] Client {context.node_config['partition-id']}: local head RESTORED")
        else:
            print(f"[FedPer] Client {context.node_config['partition-id']}: no previous head; using initialized head")

    else:
        model.load_state_dict(received_state)

    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # Pull algorithm-specific kwargs (e.g. proximal_mu) straight from config —
    # trainers that don't need them just ignore extras via **kwargs
    algorithm = context.run_config["algorithm"]
    cfg = msg.content.get("config", {})

    if algorithm == "scaffold":
        trainer = get_trainer(
            algorithm,
            server_cv=scaffold_server_cv,
            client_cv=scaffold_client_cv,
        )
    else:    
        trainer = get_trainer(
        algorithm,
        proximal_mu=msg.content["config"].get("proximal_mu", 0.0),
        head_epochs=msg.content["config"].get("fedrep-head-epochs", 1),
        mu=context.run_config.get("moon-mu", 1.0),
        temperature=context.run_config.get("moon-temperature", 0.5),
        global_model=global_model,
        lambda_1=context.run_config.get("fedmlb-lambda-1", 1.0),
        lambda_2=context.run_config.get("fedmlb-lambda-2", 1.0),
        alpha_state=alpha_state,
        alpha=context.run_config.get("fedmeta-alpha", 0.01),
        gradient_step=context.run_config.get("fedmeta-gradient-step", 5),
        support_ratio=context.run_config.get("fedmeta-support-ratio", 0.2),
        pers_lamda=context.run_config.get("pers_lamda", 0.0),
    )

    # Call the training function
    if algorithm == "moon":
        previous_state_dict = None
        if "moon_previous_state" in context.state:
            previous_state_dict = context.state["moon_previous_state"].to_torch_state_dict()

        train_loss = trainer.train(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
            previous_state_dict=previous_state_dict,
        )
        # Save trained local state for next round's contrastive loss
        context.state["moon_previous_state"] = ArrayRecord(
                {k: v.detach().cpu() for k, v in model.state_dict().items()})

    elif algorithm == "floco":
        # Extract subregion parameters if provided by server (from round tau onwards)
        subregion_params = None
        if "center" in cfg and "radius" in cfg:
            center_val = np.array(json.loads(cfg["center"]))
            radius_val = float(cfg["radius"])
            subregion_params = (center_val, radius_val)
        
        train_loss = trainer.train(
            model,
            trainloader,
            context.run_config["local-epochs"],
            cfg.get("lr", 0.01),
            device,
            subregion_parameters=subregion_params,
        )

    elif algorithm == "scaffold":
        print(
        f"[SCAFFOLD DEBUG] algorithm={algorithm}, "
        f"server_cv_is_none={trainer.server_cv is None}, "
        f"client_cv_is_none={trainer.client_cv is None}, "
        f"server_cv_count={len(trainer.server_cv) if trainer.server_cv is not None else 'None'}, "
        f"client_cv_count={len(trainer.client_cv) if trainer.client_cv is not None else 'None'}"
    )
        train_loss = trainer.train(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
        )

        # Retrieve model and control-variate changes.
        model_delta, client_cv_delta = trainer.get_updates()
        client_cv_new = trainer.get_new_client_cv()

        # Persist updated client control variates between rounds.
        context.state["scaffold_client_cv"] = ArrayRecord({
            name: value
            for name, value in zip(
                scaffold_parameter_names, client_cv_new
            )
        })

    else:
        train_loss = trainer.train(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
        )
        
    # Clean up teacher model to conserve VRAM on laptop GPUs
    if algorithm == "fedmlb":
        del global_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if algorithm == "fedbn":
        bn_state = get_bn_state_dict(model)
        context.state["fedbn_bn_state"] = ArrayRecord(bn_state)
        print("Saved local BN state")

    elif algorithm == "fedrep":
        head_state = get_fedrep_head_state_dict(model)
        context.state["fedrep_head_state"] = ArrayRecord(head_state)
        print(f"Client {partition_id}: saved personalized head "
            f"with {len(head_state)} tensors")

    elif algorithm == "fedper":
        head_state = get_fedper_head_state_dict(model)
        context.state["fedper_head_state"] = ArrayRecord(head_state)
        print(f"[FedPer] Client {partition_id}: saved personalized head "
            f"with {len(head_state)} tensors")

    # Construct and return reply Message
    if algorithm == "fedbn":
        model_record = ArrayRecord(get_fedbn_state_dict(model))
    elif algorithm == "fedrep":
        model_record = ArrayRecord(get_fedrep_shared_state_dict(model))
    elif algorithm == "fedper":
        model_record = ArrayRecord(get_fedper_shared_state_dict(model))
    elif algorithm in ("fedmeta_maml", "fedmeta_meta_sgd"):
        model_record = ArrayRecord(trainer.get_gradients())    
    elif algorithm == "scaffold":
        updates = {}
        # Send model parameter changes (y_i - x)
        for name, value in zip(scaffold_parameter_names, model_delta):
            updates["__scaffold_model_delta__" + name] = value

        # Send client control-variate changes (c_i_new - c_i)
        for name, value in zip(scaffold_parameter_names, client_cv_delta):
            updates["__scaffold_client_cv_delta__" + name] = value
        model_record = ArrayRecord(updates)

    else:
        model_record = ArrayRecord(model.state_dict())


    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "partition-id": partition_id,
    }
    
    if hasattr(trainer, "get_metrics"):
        metrics.update(trainer.get_metrics())

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    
    algorithm = context.run_config["algorithm"]
    model_name = context.run_config["model_name"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = create_model(model_name)
    if algorithm == "fedpara":
        convert_to_fedpara(model)

    elif algorithm == "moon":
        prepare_moon_model(
            model,
            projection_dim=context.run_config.get("moon-projection-dim", 84),
        )
    
    elif algorithm == "floco":
        endpoints = context.run_config.get("endpoints", 3)
        seed = context.run_config.get("seed", 0)
        model = convert_to_floco(model, endpoints=endpoints, seed=seed)
   
    received_state = msg.content["arrays"].to_torch_state_dict()

    if algorithm == "fedmeta_meta_sgd":
        model_state = {k: v for k, v in received_state.items() if not k.startswith("alpha.")}
        model.load_state_dict(model_state)

    elif algorithm == "fedbn":
        model.load_state_dict(received_state, strict=False,)
        if "fedbn_bn_state" in context.state:
            bn_state = context.state["fedbn_bn_state"].to_torch_state_dict()
            load_bn_state_dict(model, bn_state)

    elif algorithm == "fedrep":
        model.load_state_dict(received_state, strict=False)
        if "fedrep_head_state" in context.state:
            head_state = context.state["fedrep_head_state"].to_torch_state_dict()
            load_fedrep_head_state_dict(model, head_state)

    elif algorithm == "fedper":
        model.load_state_dict(received_state, strict=False)
        if "fedper_head_state" in context.state:
            head_state = context.state["fedper_head_state"].to_torch_state_dict()
            load_fedper_head_state_dict(model, head_state)

    else:
        model.load_state_dict(received_state,)

    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    cfg = msg.content.get("config", {})

    if algorithm in ("fedavg_meta", "fedmeta_maml", "fedmeta_meta_sgd"):
        trainer = get_trainer(
            algorithm,
            alpha=context.run_config.get("fedmeta-alpha"),
            gradient_step=context.run_config.get("fedmeta-gradient-step"),
            support_ratio=context.run_config.get("fedmeta-support-ratio"),
        )
        eval_loss, eval_acc = trainer.evaluate(model, valloader, device)

    elif algorithm == "floco":
        subregion_params = None
        cfg = msg.content["config"]
        if "center" in cfg and "radius" in cfg:
            center_val = np.array(json.loads(cfg["center"]))
            radius_val = float(cfg["radius"])
            subregion_params = (center_val, radius_val)
        trainer = get_trainer("floco")
        eval_loss, eval_acc = trainer.evaluate(model, valloader, device, subregion_parameters=subregion_params)

    else:
        # Call the evaluation function
        eval_loss, eval_acc = evaluate_model(
            model,
            valloader,
            device,
        )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
        "partition-id": partition_id,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)