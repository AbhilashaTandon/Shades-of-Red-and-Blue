from pca import weighted_PCA, apply_pca
from medians import demo_medians, export_medians
from autoencoder import AE
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from pathlib import Path


local_path = Path(__file__).parent.parent

saved_models = local_path / "saved_models\\model.pt"


def train(weights, model, train_data, loss_function, optimizer, scheduler, input_features, epochs, batch_size, verbose=True):
    avg_mse_loss = 0
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

        avg_mse_loss /= num_batches
        if (verbose):
            print("Epoch #%d\t%.6f" % (epoch, avg_mse_loss))
        scheduler.step(avg_mse_loss)

    return model, avg_mse_loss


def test(weights, model, test_data, loss_function, input_features):
    with torch.no_grad():
        avg_mse_loss = 0
        for weight, data in zip(weights, test_data):
            input_ = data
            input_ = torch.Tensor(input_.reshape(input_features))

            reconstructed = model(input_)

            mse_loss = loss_function(reconstructed * weight, input_ * weight)

            avg_mse_loss += mse_loss.detach().item()

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

    demo = demo.fillna("None")  # nas are floats not str

    idxs = np.arange(0, len(ideo), 1)

    train_idxs, test_idxs = train_test_split(
        idxs, test_size=.2, random_state=42)
    # exclude weights, last col

    data = ideo.values

    train_weights = torch.Tensor(data[train_idxs, -1])
    test_weights = torch.Tensor(data[test_idxs, -1])  # last row is weights

    train_data = data[train_idxs, 1:-1]  # get rid of weights and ids
    test_data = data[test_idxs, 1:-1]

    train_data, pca_transform, _ = weighted_PCA(
        train_data, train_data.shape[1], train_weights)
    test_data = apply_pca(test_data, pca_transform)  # avoid overfitting

    train_demo = demo.values[train_idxs, 1:-1]
    test_demo = demo.values[test_idxs, 1:-1]

    latent_features = 5  # len of compressed vector

    input_features = len(train_data[1])

    layer_sizes = [input_features, 84]

    print(layer_sizes)

    model = AE(
        latent_features, layer_sizes)
    LR = 1e-3

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

    num_epochs = 10

    for x in range(30):
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

    latent_medians = demo_latent_medians(model, train_data,
                                         train_idxs, train_demo, np.array(train_weights), demo.columns[1:-1])  # exclude index and weights

    export_medians(latent_medians, latent_features).to_csv(
        local_path / "data/nn_medians.csv")


def demo_latent_medians(model, data, idxs, demo, weights, demo_categories):
    print(demo_categories, demo)
    latent_features = []
    for idx, elem in zip(idxs, data):  # filter by what is present in data, train or test
        latent_features.append(model.encode(
            torch.Tensor(elem)).detach().numpy())

    latent_features = (np.array(latent_features) -
                       np.mean(latent_features, axis=0)) / np.std(latent_features, axis=0)

    medians = {}

    # exclude index, age, and weights
    for idx, category in enumerate(demo_categories):  # ignore age
        # add medians for each demographic variable to dictionary
        medians |= demo_medians(
            category, demo[:, idx], latent_features, weights)

    return medians


if __name__ == "__main__":
    training(False, True)
