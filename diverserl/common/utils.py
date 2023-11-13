from torch import nn


def soft_update(network: nn.Module, target_network: nn.Module, tau: float) -> None:
    """
    Polyak averaging for target networks.

    :param network: Network to be updated
    :param target_network: Target network to be updated
    :param tau: Interpolation factor in polyak averaging for target networks.
    """
    for param, target_param in zip(network.parameters(), target_network.parameters(), strict=True):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
