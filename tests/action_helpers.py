import torch


def legal_action_mask(game):
    """Return a tensor mask for assertions that use PyTorch indexing helpers."""
    return torch.tensor(game.ai_action_mask(), dtype=torch.uint8)
