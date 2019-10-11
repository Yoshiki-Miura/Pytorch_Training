import torch
from torch import nn
from torch import optim

from data_generation import prepare_data, plot_loss


def main():
    losses = []
    torch.manual_seed(0)
    # 真の重み

    w_true = torch.tensor([1., 2., 3.])
    N = 100
    loss_list = []
    X, y = prepare_data(N, w_true)

    # モデルの機能

    model = nn.Linear(in_features=3,
                      out_features=1,
                      bias=False)
    # モデルの重みの確認
    print(list(model.parameters()))

    # 最適化のアルゴリズムの選択
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 損失関数の指定
    criterion = nn.MSELoss()

    num_epochs = 5

    for epoch in range(1, num_epochs + 1):
        # 前epochで計算した勾配をリセットする
        optimizer.zero_grad()
        y_pred = model(X)

        loss = criterion(y_pred.view_as(y), y)

        loss.backward()
        loss_list.append(loss)
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

        optimizer.step()

    plot_loss(loss_list)


if __name__ == '__main__':
    main()
