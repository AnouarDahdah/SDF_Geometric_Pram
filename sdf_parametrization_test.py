import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.spatial import cKDTree
import glob

# Step 1: Load multiple CSV files for the training dataset
# This uses all CSV files matching the pattern 'datasurfaceSurface*.csv'
csv_files = glob.glob('/content/datasurfaceSurface*.csv')
data_list = []

# Loop through each CSV file, load the data, and append to the data_list
for file in csv_files:
    data = pd.read_csv(file, header=None, low_memory=False)
    # Convert values to numeric and drop any rows with NaNs (in case of incomplete data)
    data = data.apply(pd.to_numeric, errors='coerce').dropna()
    data_list.append(data)

# Combine all datasets into a single DataFrame for use in training
combined_data = pd.concat(data_list, ignore_index=True)
vertices = torch.tensor(combined_data.values, dtype=torch.float32)  # Convert to PyTorch tensor

# Step 2: Create a 3D grid for initializing Signed Distance Function (SDF)
grid_resolution = 50  # Number of points per axis in the grid
x_min, x_max = vertices[:, 0].min() - 1, vertices[:, 0].max() + 1
y_min, y_max = vertices[:, 1].min() - 1, vertices[:, 1].max() + 1
z_min, z_max = vertices[:, 2].min() - 1, vertices[:, 2].max() + 1

# Define grid points along each axis
x = torch.linspace(x_min, x_max, grid_resolution)
y = torch.linspace(y_min, y_max, grid_resolution)
z = torch.linspace(z_min, z_max, grid_resolution)
# Create the grid of points in 3D space
xx, yy, zz = torch.meshgrid(x, y, z)
grid_points = torch.stack([xx.ravel(), yy.ravel(), zz.ravel()], dim=1).to(vertices.device)

# Step 3: Use KD-Tree to calculate SDF values for grid points
# We use cKDTree for efficient nearest-neighbor search on the vertices
tree = cKDTree(vertices.cpu().numpy())
sdf_values_grid, _ = tree.query(grid_points.cpu().numpy())
sdf_values_grid = torch.tensor(sdf_values_grid, dtype=torch.float32).to(vertices.device)

# Step 4: Define the Autoencoder architecture for dimensionality reduction
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder part to transform input into a latent representation
        self.encoder = nn.Sequential(
            nn.Linear(3, 64),  # Input is 3-dimensional (X, Y, Z)
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)  # Output of encoder (latent space) is 16-dimensional
        )
        # Decoder part to reconstruct the input from latent space
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Output is 3-dimensional (X, Y, Z)
        )

    def forward(self, x):
        latent_space = self.encoder(x)
        reconstructed = self.decoder(latent_space)
        return latent_space, reconstructed

# Step 5: Define the MLP for predicting SDF values from the latent space
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # MLP to predict SDF from latent space + actual SDF value
        self.network = nn.Sequential(
            nn.Linear(17, 64),  # Latent space (16) + SDF value (1) = 17 input features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Output is 1-dimensional (predicted SDF value)
        )

    def forward(self, x):
        return self.network(x)

# Step 6: Initialize models and optimizers
autoencoder = Autoencoder().to(vertices.device)
mlp = MLP().to(vertices.device)

# Set up separate optimizers for each model
ae_optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
mlp_optimizer = optim.Adam(mlp.parameters(), lr=0.001)

# Define loss functions for reconstruction and SDF prediction
loss_fn_reconstruction = nn.MSELoss()
loss_fn_sdf = nn.MSELoss()

# Training hyperparameters
epochs = 5000
batch_size = 1024
loss_history = []

# Step 7: Training loop
for epoch in range(epochs):
    # Select a random batch of points from the grid for training
    indices = torch.randperm(grid_points.shape[0])[:batch_size]
    batch_points = grid_points[indices]
    batch_sdf = sdf_values_grid[indices]

    # Autoencoder forward pass
    latent_space, reconstructed = autoencoder(batch_points)

    # Prepare input for MLP (latent space + batch_sdf)
    mlp_input = torch.cat((latent_space, batch_sdf.unsqueeze(1)), dim=1)

    # Predict SDF with MLP
    predicted_sdf = mlp(mlp_input).squeeze()

    # Compute total loss: sum of reconstruction and SDF losses
    loss_reconstruction = loss_fn_reconstruction(reconstructed, batch_points)
    loss_sdf = loss_fn_sdf(predicted_sdf, batch_sdf)
    total_loss = loss_reconstruction + loss_sdf

    # Backpropagation and optimization steps
    ae_optimizer.zero_grad()
    mlp_optimizer.zero_grad()
    total_loss.backward()
    ae_optimizer.step()
    mlp_optimizer.step()

    # Log training loss for visualization
    loss_history.append(total_loss.item())

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss.item()}')

# Step 8: Visualize the loss curve to check training progress
plt.figure(figsize=(8, 6))
plt.plot(loss_history)
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Total Loss')
plt.grid(True)
plt.show()

# Step 9: Load unseen data and test the model
# Load an additional unseen CSV file to evaluate the model
unseen_csv = '/content/datasurfaceSurface7.csv'
unseen_data = pd.read_csv(unseen_csv, header=None, low_memory=False)
unseen_data = unseen_data.apply(pd.to_numeric, errors='coerce').dropna()
unseen_grid_points = torch.tensor(unseen_data.values, dtype=torch.float32).to(vertices.device)

# Use KD-Tree to compute SDF values on unseen grid points
unseen_tree = cKDTree(unseen_grid_points.cpu().numpy())
unseen_sdf_values, _ = unseen_tree.query(unseen_grid_points.cpu().numpy())
unseen_sdf_values = torch.tensor(unseen_sdf_values, dtype=torch.float32).to(vertices.device)

# Autoencoder forward pass for unseen data
latent_space_unseen, _ = autoencoder(unseen_grid_points)

# Prepare input for MLP (latent space + unseen SDF)
mlp_input_unseen = torch.cat((latent_space_unseen, unseen_sdf_values.unsqueeze(1)), dim=1)

# Predict SDF for unseen data using the trained MLP
predicted_sdf_unseen = mlp(mlp_input_unseen).squeeze().detach().cpu().numpy()

# Step 10: 3D plot for unseen data using Plotly
# Create a 3D isosurface plot to visualize(isosurfaces) the predicted SDF for unseen data

fig = go.Figure(data=[go.Isosurface(
    x=unseen_grid_points[:, 0].cpu(), y=unseen_grid_points[:, 1].cpu(), z=unseen_grid_points[:, 2].cpu(),
    value=predicted_sdf_unseen,
    isomin=0.0, isomax=1.0,
    surface_count=10,
    colorscale='Viridis',
    caps=dict(x_show=False, y_show=False, z_show=False)
)])

fig.update_layout(scene=dict(aspectmode='data'), title='Predicted SDF Isosurface for Unseen Data')
fig.show()
