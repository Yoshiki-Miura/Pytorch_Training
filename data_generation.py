from typing import List
from typing import Tuple

import torch
from matplotlib import pyplot as plt


def prepare_data(N: int, w_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.cat([torch.ones(N, 1),
                   torch.randn(N, 2)],
                  dim=1)

    noise = torch.randn(N) * 0.5
    y = torch.mv(X, w_true) + noise
    return X, y


def plot_loss(losses: List[float]) -> None:
    # エポックごとの損失の可視化
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
