"""Helpers for separating FedRep shared and local model state."""

from collections import OrderedDict

def get_fedrep_head_state_dict(model, head_name="fc3"):
    """Return the client-specific classification head state."""
    prefix = f"{head_name}."

    return OrderedDict(
        (key, value)
        for key, value in model.state_dict().items()
        if key.startswith(prefix)
    )


def get_fedrep_shared_state_dict(model, head_name="fc3"):
    """Return model state excluding the classification head."""
    prefix = f"{head_name}."

    return OrderedDict(
        (key, value)
        for key, value in model.state_dict().items()
        if not key.startswith(prefix)
    )


def load_fedrep_head_state_dict(model, head_state):
    """Restore a client's local classification head."""
    if head_state:
        model.load_state_dict(head_state, strict=False)