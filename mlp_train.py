import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import make_interp_spline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# --- 0. Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 2. Define the Neural Network Model (MLP) ---
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # Your original MLPRegressor has 7 hidden layers of size 'x'
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.fc8 = nn.Linear(hidden_size, output_size)  # Output layer

        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.fc8(x)  # No activation on the output layer for regression

        return x


# --- 3. Training Function ---
def train_mlp(
    input_size,
    hidden_size,
    output_size,
    X_train,
    y_train,
    X_full,
    y_full,
    num_epochs=1000,
    learning_rate=0.001,
    patience=50,
    batch_size=256,
):
    model = MLP(input_size, hidden_size, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )  # Shuffle data each epoch

    best_loss = float("inf")
    counter = 0

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0

        for batch_X, batch_y in train_loader:

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(
                0
            )  # Accumulate loss, weighted by batch size

            avg_epoch_loss = epoch_loss / len(
                train_dataset
            )  # Average loss over the epoch

            # Early Stopping based on average epoch loss
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

    # Evaluation
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for inference
        y_predict_tensor = model(X_full)
        # Move predictions to CPU and convert to numpy for scikit-learn metrics
        y_predict_np = y_predict_tensor.reshape(-1).cpu().numpy()
        y_full_np_eval = (
            y_full.cpu().numpy()
        )  # Ensure y_full is also on CPU for metric calculation

        mse_test = mean_squared_error(y_full_np_eval, y_predict_np)

    return mse_test


if __name__ == "__main__":
    # If csv file is not present, rerun the logic

    if not os.path.exists("MLP/mlp_results.csv"):
        df = pd.read_csv("Ratio.csv")
        df.head()

        X_full = df[["Lb/ry", "h/tw", "B/2t", "h/B", "tf/tw", "E/Fy", "Mp/Mcr"]]
        y_full = df["Mn/Mp"]

        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=0.2, random_state=123
        )

        standard_scalar = StandardScaler()
        X_train_transformed = standard_scalar.fit_transform(X_train)
        X_full_transformed = standard_scalar.transform(X_full)

        # --- 1. Data Preparation (Dummy Data - Replace with your actual data) ---
        # Assuming X_train, y_train, X_full, y_full are already defined as NumPy arrays.
        # Make sure X_full and y_full have corresponding dimensions.

        input_dim = X_train_transformed.shape[1]  # Example input dimension
        output_dim = 1  # Example output dimension (for regression, often 1)

        # Generate dummy data for demonstration
        num_train_samples = len(X_train_transformed)
        num_full_samples = len(X_full_transformed)

        # Convert NumPy arrays to PyTorch tensors and move to device
        X_train_tensor = torch.tensor(X_train_transformed).float()
        y_train_tensor = torch.from_numpy(np.array(y_train)).float().unsqueeze(1)
        X_full_tensor = torch.tensor(X_full_transformed).float()
        y_full_tensor = torch.from_numpy(np.array(y_full)).float().unsqueeze(1)

        # --- 4. Experiment Loop ---
        layersize = list(range(5, 25))  # Hidden layer sizes from 5 to 19
        mselist = []

        num_runs_per_layersize = (
            256  # Reduced from 1000 for faster demonstration, adjust as needed
        )

        for x in layersize:
            mse_tests_for_x = []

            print(f"Running experiments for hidden_size = {x}")
            with ProcessPoolExecutor(
                max_workers=16, mp_context=mp.get_context("spawn")
            ) as executor:
                for i in range(num_runs_per_layersize):
                    future = executor.submit(
                        train_mlp,
                        input_dim,
                        x,
                        output_dim,
                        X_train_tensor.clone().to(device),
                        y_train_tensor.clone().to(device),
                        X_full_tensor.clone().to(device),
                        y_full_tensor.clone().to(device),
                    )
                    mse_tests_for_x.append(future)

            mselist.append(np.min([x.result() for x in mse_tests_for_x]))

        # Ensure 'MLP' directory exists for saving the plot
        if not os.path.exists("MLP"):
            os.makedirs("MLP")

        # Save the results to a CSV file
        with open("MLP/mlp_results.csv", "w") as f:
            f.write("layersize,mse\n")
            for size, mse in zip(layersize, mselist):
                f.write(f"{size},{mse}\n")

    layersize = []
    mselist = []
    with open("MLP/mlp_results.csv", "r") as f:
        for line in f.readlines()[1:]:
            size, mse = line.strip().split(",")
            layersize.append(int(size))
            mselist.append(float(mse))

    # MSE Plot
    layersize_new = np.linspace(np.min(layersize), np.max(layersize), 300)
    spl = make_interp_spline(layersize, mselist, k=2)
    mselist_new = spl(layersize_new)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["text.usetex"] = True
    plt.plot(layersize_new, mselist_new)
    plt.scatter(layersize, mselist, color="red", zorder=5)
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Number of Neurons in Hidden Layers")
    plt.xticks(layersize)
    plt.savefig(f"MLP/mse-neurons-pytorch.png", dpi=640, bbox_inches="tight")
    # plt.show()

    print("\nExperiment complete. Plot saved in the 'MLP' directory.")
