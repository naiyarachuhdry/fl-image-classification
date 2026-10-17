"""FLOCo Server Strategy for ServerApp.

Inherits from FedAvg and manages:
1. Sampling all clients in round (tau - 1) to collect gradient updates.
2. Computing PCA client projection onto the solution simplex in round tau.
3. Distributing localized subregion parameters (center, radius) to clients from round tau onwards.
"""

import copy
import json
from typing import Dict, Iterable, List, Optional
import numpy as np
import torch

from flwr.app import ArrayRecord, ConfigRecord, Message, MessageType, RecordDict
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import sample_nodes

from utils.floco_helper import project_clients


class Floco(FedAvg):
    """FLOCo server strategy compatible with Flower ServerApp Grid API."""

    def __init__(
        self,
        tau: int = 2,
        rho: float = 0.0,
        endpoints: int = 3,
        fraction_fit: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tau = int(tau)
        self.rho = float(rho)
        self.endpoints = int(endpoints)
        self.fraction_train = float(fraction_fit)

        self.current_arrays: Optional[ArrayRecord] = None
        self.client_gradients: Dict[int, np.ndarray] = {}
        self.client_subregion_parameters: Dict[int, np.ndarray] = {}
        self.node_id_to_partition_id: Dict[int, int] = {}
        self.last_selected_node_ids: set = set()

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid
    ) -> Iterable[Message]:
        """Configure the training round."""
        if self.fraction_train == 0.0:
            return []

        self.current_arrays = arrays
        all_node_ids = list(grid.get_node_ids())
        self.last_selected_node_ids = set()

        if (server_round + 1) == self.tau:
            # Round before projection: sample ALL clients to collect gradient updates
            num_nodes = int(len(all_node_ids) * self.fraction_train)
            regular_sample_size = max(num_nodes, self.min_train_nodes)
            regular_node_ids, _ = sample_nodes(grid, self.min_available_nodes, regular_sample_size)
            self.last_selected_node_ids = set(regular_node_ids)

            # Sample all nodes for gradient collection
            node_ids, _ = sample_nodes(grid, self.min_available_nodes, len(all_node_ids))
        else:
            num_nodes = int(len(all_node_ids) * self.fraction_train)
            sample_size = max(num_nodes, self.min_train_nodes)
            node_ids, _ = sample_nodes(grid, self.min_available_nodes, sample_size)

        # At round tau: project collected client gradients onto simplex
        if server_round == self.tau and len(self.client_gradients) > 0:
            projected = project_clients(self.client_gradients, self.endpoints)
            sorted_pids = sorted(self.client_gradients.keys())
            self.client_subregion_parameters = {
                pid: projected[idx] for idx, pid in enumerate(sorted_pids)
            }

        # Construct messages with subregion configurations
        messages = []
        for node_id in node_ids:
            client_config = copy.deepcopy(config)
            client_config["server-round"] = server_round

            if server_round >= self.tau and self.client_subregion_parameters:
                pid = self.node_id_to_partition_id.get(node_id, 0)
                if pid in self.client_subregion_parameters:
                    center_vec = self.client_subregion_parameters[pid]
                    client_config["center"] = json.dumps(center_vec.tolist())
                    client_config["radius"] = float(self.rho)

            record = RecordDict({
                self.arrayrecord_key: arrays,
                self.configrecord_key: client_config,
            })
            messages.append(
                Message(content=record, message_type=MessageType.TRAIN, dst_node_id=node_id)
            )

        return messages

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ):
        """Aggregate client parameters and extract pseudo-gradients in round tau - 1."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True, validate=False)
        if not valid_replies:
            return None, None

        # Track node_id -> partition_id using msg.metadata.src_node_id
        for msg in valid_replies:
            if "partition-id" in msg.content["metrics"]:
                self.node_id_to_partition_id[msg.metadata.src_node_id] = int(
                    msg.content["metrics"]["partition-id"]
                )

        # Collect client gradients at round (tau - 1)
        if (server_round + 1) == self.tau and self.current_arrays is not None:
            init_state = self.current_arrays.to_torch_state_dict()
            for msg in valid_replies:
                pid = int(
                    msg.content["metrics"].get(
                        "partition-id",
                        self.node_id_to_partition_id.get(msg.metadata.src_node_id, 0),
                    )
                )
                client_state = msg.content["arrays"].to_torch_state_dict()

                # Extract simplex endpoint weights (keys containing '_weights.')
                simplex_keys = [k for k in init_state.keys() if "_weights." in k]
                if simplex_keys:
                    grads = [
                        (init_state[k] - client_state[k]).cpu().float().numpy().flatten()
                        for k in simplex_keys
                    ]
                    self.client_gradients[pid] = np.concatenate(grads)

            # If only a subset of clients was meant to be aggregated:
            if self.last_selected_node_ids:
                valid_replies = [
                    msg for msg in valid_replies if msg.metadata.src_node_id in self.last_selected_node_ids
                ]

        # Standard weighted FedAvg parameter aggregation
        return super().aggregate_train(server_round, valid_replies)

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid
    ) -> Iterable[Message]:
        """Configure evaluation round."""
        if self.fraction_evaluate == 0.0:
            return []

        all_node_ids = list(grid.get_node_ids())
        num_nodes = int(len(all_node_ids) * self.fraction_evaluate)
        sample_size = max(num_nodes, self.min_evaluate_nodes)
        node_ids, _ = sample_nodes(grid, self.min_available_nodes, sample_size)

        messages = []
        for node_id in node_ids:
            client_config = copy.deepcopy(config)
            client_config["server-round"] = server_round

            if server_round >= self.tau and self.client_subregion_parameters:
                pid = self.node_id_to_partition_id.get(node_id, 0)
                if pid in self.client_subregion_parameters:
                    center_vec = self.client_subregion_parameters[pid]
                    client_config["center"] = json.dumps(center_vec.tolist())
                    client_config["radius"] = float(self.rho)

            record = RecordDict({
                self.arrayrecord_key: arrays,
                self.configrecord_key: client_config,
            })
            messages.append(
                Message(content=record, message_type=MessageType.EVALUATE, dst_node_id=node_id)
            )

        return messages