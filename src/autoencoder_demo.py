
from autoencoder import AE
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from pathlib import Path


local_path = Path(__file__).parent.parent

saved_models = local_path / "\\saved_models\\model.pt"

 latent_features = 10  # len of compressed vector

    input_features = len(train_data[1])

    layer_sizes = [input_features, 85, 59, 42]

    print(layer_sizes)

    model = AE(
        latent_features, layer_sizes)  # 20 different parameters


checkpoint = torch.load(saved_models)
model.load_state_dict(checkpoint['model_state_dict'])
