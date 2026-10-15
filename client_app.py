"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from task import load_data, load_data_fedpft
from utils.metrics import evaluate_model

from algorithms import get_trainer

from models import create_model

from models.fedpft_model import FedPFTNet, build_feature_extractor

from utils.fedbn_helper import get_fedbn_state_dict, get_bn_state_dict, load_bn_state_dict
from utils.fedpara_helper import convert_to_fedpara
from utils.fedrep_helper import get_fedrep_head_state_dict, get_fedrep_shared_state_dict, load_fedrep_head_state_dict
from utils.moon_helper import prepare_moon_model

# Flower ClientApp
app = ClientApp()

def fedpft_train(msg, context):
    """FedPFT client: extract features once, fit GMMs, send them."""
    cfg = context.run_config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    extractor = build_feature_extractor(cfg["fedpft-backbone"])
    trainloader, _ = load_data_fedpft(
        context.node_config["partition-id"],
        context.node_config["num-partitions"],
        cfg["batch-size"],
        cfg["fedpft-image-size"],
    )
    trainer = get_trainer(
        "fedpft",
        n_mixtures=cfg["fedpft-n-mixtures"],
        cov_type=cfg["fedpft-cov-type"],
        seed=cfg["fedpft-gmm-seed"],
    )
    arrays, num_examples = trainer.extract_and_fit(extractor, trainloader, device)
    content = RecordDict({
        "arrays": ArrayRecord(arrays),
        "metrics": MetricRecord({"num-examples": num_examples}),
    })
    return Message(content=content, reply_to=msg)


def fedpft_evaluate(msg, context):
    """FedPFT client: evaluate the server's head on local test data."""
    cfg = context.run_config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FedPFTNet(cfg["fedpft-backbone"], cfg.get("num-classes", 10))
    model.load_head(msg.content["arrays"].to_torch_state_dict())
    model.to(device)
    _, valloader = load_data_fedpft(
        context.node_config["partition-id"],
        context.node_config["num-partitions"],
        cfg["batch-size"],
        cfg["fedpft-image-size"],
    )
    eval_loss, eval_acc = evaluate_model(model, valloader, device)
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    if context.run_config["algorithm"] == "fedpft":
        return fedpft_train(msg, context)

    algorithm = context.run_config["algorithm"]
    model_name = context.run_config["model_name"]
    # Create model
    model = create_model(model_name) 
    
    if algorithm == "fedpara":
        convert_to_fedpara(model)

    elif algorithm == "moon":
        prepare_moon_model(
            model,
            projection_dim=context.run_config.get("moon-projection-dim", 84),
        )

    # Load received parameters
    received_state= msg.content["arrays"].to_torch_state_dict()

    if algorithm == "fedbn":
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

    else:
        model.load_state_dict(received_state)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # Pull algorithm-specific kwargs (e.g. proximal_mu) straight from config —
    # trainers that don't need them just ignore extras via **kwargs
    algorithm = context.run_config["algorithm"]
    trainer = get_trainer(
        algorithm,
        proximal_mu=msg.content["config"].get("proximal_mu", 0.0),
        head_epochs=msg.content["config"].get("fedrep-head-epochs", 1),
        mu=context.run_config.get("moon-mu", 1.0),
        temperature=context.run_config.get("moon-temperature", 0.5),
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
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )
    else:
        train_loss = trainer.train(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
        )
    
    if algorithm == "fedbn":
        bn_state = get_bn_state_dict(model)
        context.state["fedbn_bn_state"] = ArrayRecord(bn_state)
        print("Saved local BN state")

    elif algorithm == "fedrep":
        head_state = get_fedrep_head_state_dict(model)
        context.state["fedrep_head_state"] = ArrayRecord(head_state)
        print(f"Client {partition_id}: saved personalized head "
            f"with {len(head_state)} tensors")

    # Construct and return reply Message
    if algorithm == "fedbn":
        model_record = ArrayRecord(get_fedbn_state_dict(model))
    elif algorithm == "fedrep":
        model_record = ArrayRecord(get_fedrep_shared_state_dict(model))
    else:
        model_record = ArrayRecord(model.state_dict())


    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    
    if hasattr(trainer, "get_metrics"):
        metrics.update(trainer.get_metrics())

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    if context.run_config["algorithm"] == "fedpft":
        return fedpft_evaluate(msg, context)
    
    algorithm = context.run_config["algorithm"]
    model_name = context.run_config["model_name"]

    model = create_model(model_name)
    if algorithm == "fedpara":
        convert_to_fedpara(model)

    elif algorithm == "moon":
        prepare_moon_model(
            model,
            projection_dim=context.run_config.get("moon-projection-dim", 84),
        )
   
    received_state = msg.content["arrays"].to_torch_state_dict()

    if algorithm == "fedbn":
        model.load_state_dict(received_state, strict=False,)
        if "fedbn_bn_state" in context.state:
            bn_state = context.state["fedbn_bn_state"].to_torch_state_dict()
            load_bn_state_dict(model, bn_state)

    elif algorithm == "fedrep":
        model.load_state_dict(received_state, strict=False)
        if "fedrep_head_state" in context.state:
            head_state = context.state["fedrep_head_state"].to_torch_state_dict()
            load_fedrep_head_state_dict(model, head_state)

    else:
        model.load_state_dict(received_state,)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

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
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
