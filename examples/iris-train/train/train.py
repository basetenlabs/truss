from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.input_layer = nn.Linear(4, 50)
        self.hidden_layer1 = nn.Linear(50, 25)
        self.output_layer = nn.Linear(25, 3)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer1(x))
        x = F.softmax(self.output_layer(x), dim=1)
        return x


class Train:
    def __init__(
        self,
        config,
        output_dir: Path,
        variables,
    ):
        self.config = config
        self.output_dir = output_dir
        self.variables = variables
        self.data = []
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.epochs = 100

    def load_data(self):
        iris = load_iris()
        X = iris["data"]
        y = iris["target"]

        # Scale data to have mean 0 and variance 1
        # which is importance for convergence of the neural network
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data set into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, shuffle=True
        )
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)

        self.data = [X_train, X_test, y_train, y_test]

    def train(self):
        # Write your training code here, populating generate artifacts in
        # self.training_artifacts_dir
        self.model = Model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002)
        self.loss_fn = nn.CrossEntropyLoss()

        self.load_data()

        X_train, X_test, y_train, y_test = self.data

        train_losses = []
        test_losses = []

        for epoch in range(1, self.epochs + 1):
            self.optimizer.zero_grad()

            out = self.model(X_train)
            loss = self.loss_fn(out, y_train)
            loss.backward()
            train_losses.append(loss.item())

            self.optimizer.step()

            out = self.model(X_test)
            loss = self.loss_fn(out, y_test)
            test_losses.append(loss.item())

            if (epoch) % 10 == 0:
                print(
                    f"Epoch {epoch}/{self.epochs}, Train Loss: {train_losses[epoch-1]:.4f}, Test Loss: {test_losses[epoch-1]:.4f}"
                )
                with (self.output_dir / f"epoch-{epoch}-loss").open("w") as fp:
                    fp.write(
                        f"Epoch {epoch}/{self.epochs}, Train Loss: {train_losses[epoch-1]:.4f}, Test Loss: {test_losses[epoch-1]:.4f}"
                    )
            if (epoch + 1) % 25 == 0:
                PATH = f"{self.output_dir}/model{epoch}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    PATH,
                )
