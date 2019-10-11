from typing import Tuple

import matplotlib.pyplot as plt
import torch


def prepare_data(N: int, w_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.cat([torch.ones(N, 1),
                   torch.randn(N, 2)],
                  dim=1)

    noise = torch.randn(N) * 0.5
    y = torch.mv(X, w_true) + noise
    return X, y


def main():
    losses = []
    torch.manual_seed(0)
    # 真の重み

    w_true = torch.tensor([1., 2., 3.])
    N = 100
    X, y = prepare_data(N, w_true)

    # 重みの初期化
    w = torch.randn(w_true.size(0), requires_grad=True)

    # 学習におけるパラメータ
    learning_rate = 0.1
    num_epochs = 5

    for epoch in range(1, num_epochs + 1):
        # 前エポックで計算した勾配をリセットする
        w.grad = None

        y_pred = torch.mv(X, w)  # 予測出力の計算
        loss = torch.mean((y_pred - y) ** 2)  # 損失の計算
        losses.append(loss.item())

        loss.backward()  # 誤差逆伝搬による勾配計算

        print(f'Epoch: {epoch}, loss: {loss.item()}, w={w.data}, dt/dw={w.grad.data}')

        # 重みの更新
        w.data = w - learning_rate * w.grad.data

    plt.plot(range(1, epoch + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


if __name__ == '__main__':
    main()
