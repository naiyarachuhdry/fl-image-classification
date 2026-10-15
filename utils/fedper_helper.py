"""Helpers for separating FedPer shared and personalized model state."""

from collections import OrderedDict

def get_fedper_head_state_dict(model, head_name="fc3"):
    """Return the personalized classification head state."""
    prefix = f"{head_name}."
    
    return OrderedDict(
        (key, value)
        for key, value in model.state_dict().items()
        if key.startswith(prefix)
    )


def get_fedper_shared_state_dict(model, head_name="fc3"):
    """Return model state excluding the personalized head."""
    prefix = f"{head_name}."
    return OrderedDict(
        (key, value)
        for key, value in model.state_dict().items()
        if not key.startswith(prefix)
    )


def load_fedper_head_state_dict(model, head_state):
    """Restore this client's personalized head."""
    if head_state:
        model.load_state_dict(head_state, strict=False)
