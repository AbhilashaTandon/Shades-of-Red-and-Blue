
from autoencoder import AE
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from pathlib import Path


local_path = Path(__file__).parent.parent

saved_models = local_path / "saved_models\\model.pt"


def train(weights, model, train_data, loss_function, optimizer, scheduler, input_features, epochs, batch_size, verbose=True):
    for epoch in range(epochs):
        avg_mse_loss = 0
        np.random.shuffle(train_data)
        num_batches = len(train_data) // batch_size
        for i in range(num_batches):
            min_index = i * batch_size
            max_index = min((i+1) * batch_size, len(train_data))
            batch = np.array([train_data[x]
                             for x in range(min_index, max_index)])
            batch_weights = torch.Tensor([weights[x]
                                          for x in range(min_index, max_index)]).repeat(input_features).reshape(batch_size, input_features)

            batch = torch.Tensor(
                batch).reshape(batch_size, input_features)

            reconstructed = model(batch)

            mse_loss = loss_function(
                batch_weights * reconstructed, batch_weights * batch)

            loss = mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # updating moving average
            mse_loss_val = mse_loss.detach().item()
            avg_mse_loss += mse_loss_val
            # avg_kl_loss += model.kl

        avg_mse_loss /= num_batches
        # avg_kl_loss /= num_batches
        if (verbose):
            print("Epoch #%d\t%.6f" % (epoch, avg_mse_loss))
        scheduler.step(avg_mse_loss)

    return model, avg_mse_loss


def test(weights, model, test_data, loss_function, input_features):
    with torch.no_grad():
        avg_mse_loss = 0
        # avg_kl_loss = 0
        for weight, data in zip(weights, test_data):
            input_ = data
            input_ = torch.Tensor(input_.reshape(input_features))

            reconstructed = model(input_)

            mse_loss = loss_function(reconstructed * weight, input_ * weight)

            # kl_loss = model.kl

            avg_mse_loss += mse_loss.detach().item()
            # avg_kl_loss += model.kl

        return avg_mse_loss / len(test_data)


def training(load=False, save=False):

    ideo = pd.DataFrame()
    demo = pd.DataFrame()

    ideo_path = local_path / "data/ideo.csv"

    with ideo_path.open() as f:
        ideo = pd.read_csv(f)

    demo_path = local_path / "data/demo.csv"

    with demo_path.open() as f:
        demo = pd.read_csv(f)

    train_data, test_data = train_test_split(
        ideo.values, test_size=.2, random_state=42)
    # exclude weights, last col

    print(ideo.head(5))

    train_weights = torch.Tensor(train_data[:, -1])
    test_weights = torch.Tensor(test_data[:, -1])
    train_data = train_data[:, 1:-1]
    test_data = test_data[:, 1:-1]

    print(train_weights.shape)

    print(train_data.shape)
    print(test_data.shape)

    latent_features = 10  # len of compressed vector

    input_features = len(train_data[1])

    layer_sizes = [input_features, 84, 59, 41]

    print(layer_sizes)

    model = AE(
        latent_features, layer_sizes)  # 20 different parameters
    LR = 1e-2

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LR)

    num_params = sum(p.numel()
                     for p in model.parameters() if p.requires_grad)

    print("Number of trainable parameters: %d" % num_params)
    # number of trainable parameters

    if (load):
        checkpoint = torch.load(saved_models)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=19, factor=0.316227766017, mode='min', verbose=True, threshold=1e-4)

    # Validation using Loss function
    loss_function = torch.nn.MSELoss()

    # print("Epoch\tTrain_MSE\tTest_MSE")

    num_epochs = 5

    for x in range(20):
        model, train_mse_loss = train(train_weights, model, train_data, loss_function,
                                      optimizer, scheduler, input_features=input_features, epochs=num_epochs, batch_size=512, verbose=False)

        test_mse_loss = test(test_weights, model, test_data,
                             loss_function, input_features=input_features)
        print("%d\t%.4f\t%.4f" % (x * num_epochs + num_epochs - 1, train_mse_loss,
              test_mse_loss))

    s = model.encode(torch.Tensor(train_data[0]))
    print(s)

    if (save):
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, saved_models)
        print("MODEL SAVED")


def demo_medians(model, ideo, demo):
    latent_features = []


if __name__ == "__main__":
    training(True, True)
