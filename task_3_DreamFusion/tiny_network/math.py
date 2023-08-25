import torch

def expected_sin(x_means: torch.Tensor, x_vars: torch.Tensor) -> torch.Tensor:
    """Computes the expected value of sin(y) where y ~ N(x_means, x_vars)

    Args:
        x_means: Mean values.
        x_vars: Variance of values.

    Returns:
        torch.Tensor: The expected value of sin.
    """

    return torch.exp(-0.5 * x_vars) * torch.sin(x_means)

