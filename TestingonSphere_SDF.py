import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import cKDTree
import os

# Set up a directory to save each plot
os.makedirs("epoch_plots", exist_ok=True)

# Function to generate points on a sphere
def generate_sphere(radius, num_points):
    phi = np.random.uniform(0, np.pi, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack((x, y, z))

# Step 1: Generate training sphere surface data
train_radius = 1.0
train_num_points = 1000
vertices = generate_sphere(train_radius, train_num_points)

# Convert to PyTorch tensor
vertices_tensor = torch.tensor(vertices, dtype=torch.float32)

# Step 2: Create a 3D grid for SDF initialization
grid_resolution = 50
x_min, x_max = vertices_tensor[:, 0].min() - 1, vertices_tensor[:, 0].max() + 1
y_min, y_max = vertices_tensor[:, 1].min() - 1, vertices_tensor[:, 1].max() + 1
z_min, z_max = vertices_tensor[:, 2].min() - 1, vertices_tensor[:, 2].max() + 1

x = torch.linspace(x_min, x_max, grid_resolution)
y = torch.linspace(y_min, y_max, grid_resolution)
z = torch.linspace(z_min, z_max, grid_resolution)
xx, yy, zz = torch.meshgrid(x, y, z)
grid_points = torch.stack([xx.ravel(), yy.ravel(), zz.ravel()], dim=1).to(vertices_tensor.device)

# Step 3: Use KD-Tree to calculate SDF values for grid points
tree = cKDTree(vertices_tensor.cpu().numpy())
sdf_values_grid, _ = tree.query(grid_points.cpu().numpy())
sdf_values_grid = torch.tensor(sdf_values_grid, dtype=torch.float32).to(vertices_tensor.device)

# Normalize SDF values for stability
sdf_values_grid = (sdf_values_grid - sdf_values_grid.mean()) / sdf_values_grid.std()

# Step 4: Define the Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        latent_space = self.encoder(x)
        reconstructed = self.decoder(latent_space)
        return latent_space, reconstructed

# Step 5: Define the MLP for SDF prediction
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(33, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

# Step 6: Initialize models and optimizers
autoencoder = Autoencoder().to(vertices_tensor.device)
mlp = MLP().to(vertices_tensor.device)

ae_optimizer = optim.Adam(autoencoder.parameters(), lr=0.0005)
mlp_optimizer = optim.Adam(mlp.parameters(), lr=0.0005)

loss_fn_reconstruction = nn.MSELoss()
loss_fn_sdf = nn.MSELoss()

epochs = 5000
batch_size = 1024
loss_history = []
plot_interval = 500

# Step 7: Training loop
for epoch in range(epochs):
    indices = torch.randperm(grid_points.shape[0])[:batch_size]
    batch_points = grid_points[indices]
    batch_sdf = sdf_values_grid[indices]

    # Autoencoder forward pass
    latent_space, reconstructed = autoencoder(batch_points)

    # Prepare input for MLP (latent space + batch_sdf)
    mlp_input = torch.cat((latent_space, batch_sdf.unsqueeze(1)), dim=1)

    # Predict SDF with MLP
    predicted_sdf = mlp(mlp_input).squeeze()

    # Loss computation
    loss_reconstruction = loss_fn_reconstruction(reconstructed, batch_points)
    loss_sdf = loss_fn_sdf(predicted_sdf, batch_sdf)
    total_loss = loss_reconstruction + loss_sdf

    # Backpropagation
    ae_optimizer.zero_grad()
    mlp_optimizer.zero_grad()
    total_loss.backward()
    ae_optimizer.step()
    mlp_optimizer.step()

    loss_history.append(total_loss.item())

    if epoch % plot_interval == 0 or epoch == epochs - 1:
        print(f'Epoch {epoch}, Loss: {total_loss.item()}')

        # Visualize the SDF for current epoch
        with torch.no_grad():
            # Predict SDF values for the grid
            latent_space_grid, _ = autoencoder(grid_points)
            mlp_input_grid = torch.cat((latent_space_grid, sdf_values_grid.unsqueeze(1)), dim=1)
            predicted_sdf_grid = mlp(mlp_input_grid).squeeze().cpu().numpy()

            # Create 3D plot for the predicted SDF
            fig = go.Figure(data=[go.Isosurface(
                x=grid_points[:, 0].cpu().numpy(),
                y=grid_points[:, 1].cpu().numpy(),
                z=grid_points[:, 2].cpu().numpy(),
                value=predicted_sdf_grid,
                isomin=-0.1,
                isomax=0.1,
                surface_count=3,
                colorscale='Viridis',
                caps=dict(x_show=False, y_show=False, z_show=False)
            )])

            fig.update_layout(
                title=f'SDF Isosurface at Epoch {epoch}',
                scene=dict(aspectmode='data')
            )

            # Save the plot
            fig.write_image(f"epoch_plots/epoch_{epoch}.png")


