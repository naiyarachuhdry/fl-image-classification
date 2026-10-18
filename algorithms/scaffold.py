import numpy as np

from flwr.app import Array, ArrayRecord
from flwr.serverapp.strategy import FedAvg

class Scaffold(FedAvg):
    """Server strategy for SCAFFOLD federated learning."""

    def __init__(self, parameter_names,  num_clients, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.parameter_names = list(parameter_names)
        self.num_clients = num_clients 
        self.server_cv = {}
        self.current_arrays = None

        self.server_cv_prefix = "__scaffold_server_cv__"
        self.model_delta_prefix = "__scaffold_model_delta__"
        self.client_cv_delta_prefix = "__scaffold_client_cv_delta__"

    def configure_train(self, server_round, arrays, config, grid):
        """Send global model parameters and server control variates."""

        self.current_arrays = arrays

        array_keys = list(arrays.keys())
        global_arrays = arrays.to_numpy_ndarrays()
        global_state = dict(zip(array_keys, global_arrays))

        # Initialize server control variates to zero.
        if not self.server_cv:
            for name in self.parameter_names:
                if name not in global_state:
                    raise ValueError(
                        f"Trainable parameter {name!r} is missing "
                        "from the global model arrays."
                    )

                self.server_cv[name] = np.zeros_like(global_state[name])

        # Include server control variates in the message sent to clients.
        outgoing = {
            name: Array(np.asarray(value))
            for name, value in global_state.items()
        }

        for name in self.parameter_names:
            outgoing[self.server_cv_prefix + name] = Array(
                np.asarray(self.server_cv[name])
            )

        outgoing_arrays = ArrayRecord(outgoing)

        return super().configure_train(
            server_round,
            outgoing_arrays,
            config,
            grid,
        )

    def aggregate_train(self, server_round, replies):
        """Aggregate client updates and update the global model."""

        valid_replies, _ = self._check_and_log_replies(
            replies,
            is_train=True,
            validate=False,
        )

        if not valid_replies:
            return None, None

        global_keys = list(self.current_arrays.keys())
        global_values = self.current_arrays.to_numpy_ndarrays()
        global_state = dict(zip(global_keys, global_values))

        weights = [
            float(msg.content["metrics"].get(self.weighted_by_key, 1.0))
            for msg in valid_replies
        ]
        total_weight = sum(weights)

        if total_weight <= 0:
            raise ValueError("Total client aggregation weight must be positive.")

        aggregated_model_delta = {}
        aggregated_cv_delta = {}

        for name in self.parameter_names:
            aggregated_model_delta[name] = np.zeros_like(global_state[name])
            aggregated_cv_delta[name] = np.zeros_like(self.server_cv[name])

        
        for msg in valid_replies:
            client_arrays = msg.content["arrays"].to_numpy_ndarrays()
            client_keys = list(msg.content["arrays"].keys())
            client_state = dict(zip(client_keys, client_arrays))

            for name in self.parameter_names:
                model_key = self.model_delta_prefix + name
                cv_key = self.client_cv_delta_prefix + name

                if model_key not in client_state or cv_key not in client_state:
                    raise ValueError(
                        f"Client reply is missing SCAFFOLD updates for {name!r}."
                    )

                # Model deltas: uniform aggregation across participating clients.
                aggregated_model_delta[name] += (client_state[model_key] / len(valid_replies))

                # Control-variate deltas: unweighted sum.
                aggregated_cv_delta[name] += client_state[cv_key]

        # Apply the aggregated model update to trainable parameters.
        updated_state = dict(global_state)

        for name in self.parameter_names:
            updated_state[name] = (global_state[name] + aggregated_model_delta[name])

        # Preserve non-trainable state entries and buffers.
        self.current_arrays = ArrayRecord({
            name: Array(np.asarray(updated_state[name]))
            for name in global_keys
        })

        # Standard SCAFFOLD server control-variate update.
        for name in self.parameter_names:
            self.server_cv[name] += (aggregated_cv_delta[name] / self.num_clients)

        metrics = self.train_metrics_aggr_fn(
            [msg.content for msg in valid_replies],
            self.weighted_by_key,
        )

        return self.current_arrays, metrics