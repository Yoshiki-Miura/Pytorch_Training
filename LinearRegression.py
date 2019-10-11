import torch

from data_generation import prepare_data, plot_loss


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

    plot_loss(losses)


if __name__ == '__main__':
    main()
