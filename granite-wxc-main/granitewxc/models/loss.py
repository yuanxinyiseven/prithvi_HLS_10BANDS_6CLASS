import torch

def rmse_loss(y_hat: torch.Tensor, y: dict[torch.Tensor]) -> torch.Tensor:

    return torch.sqrt(torch.mean((y_hat - y['y']) ** 2))
