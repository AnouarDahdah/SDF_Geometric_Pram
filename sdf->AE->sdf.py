import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from scipy.spatial import cKDTree

# Step 1: Load the CSV file for the training dataset
csv_filename = '/content/datasurfaceSurface20.csv'
data = pd.read_csv(csv_filename, header=None, low_memory=False)

# Convert the data to numeric and drop NaNs
data = data.apply(pd.to_numeric, errors='coerce').dropna()
vertices = torch.tensor(data.values, dtype=torch.float32)  # Convert to PyTorch tensor

# Step 2: Create a 3D grid for SDF initialization
grid_resolution = 50
x_min, x_max = vertices[:, 0].min() - 1, vertices[:, 0].max() + 1
y_min, y_max = vertices[:, 1].min() - 1, vertices[:, 1].max() + 1
z_min, z_max = vertices[:, 2].min() - 1, vertices[:, 2].max() + 1

x = torch.linspace(x_min, x_max, grid_resolution)
y = torch.linspace(y_min, y_max, grid_resolution)
z = torch.linspace(z_min, z_max, grid_resolution)
xx, yy, zz = torch.meshgrid(x, y, z)
grid_points = torch.stack([xx.ravel(), yy.ravel(), zz.ravel()], dim=1).to(vertices.device)

# Step 3: Use KD-Tree to calculate SDF values for grid points
tree = cKDTree(vertices.cpu().numpy())
sdf_values_grid, _ = tree.query(grid_points.cpu().numpy())
sdf_values_grid = torch.tensor(sdf_values_grid, dtype=torch.float32).to(vertices.device)

# Step 4: Define the Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(3, 64),  # Input is 3-dimensional (X, Y, Z)
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)  # Latent space is 16-dimensional
        )
        # Decoder
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

# Step 5: Define the MLP for SDF prediction
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
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

ae_optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
mlp_optimizer = optim.Adam(mlp.parameters(), lr=0.001)

loss_fn_reconstruction = nn.MSELoss()
loss_fn_sdf = nn.MSELoss()

# Training settings
epochs = 5000
batch_size = 1024
loss_history = []

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

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss.item()}')

# Step 9: Load unseen data and test the model
unseen_csv = '/content/datasurfaceSurface15.csv'
unseen_data = pd.read_csv(unseen_csv, header=None, low_memory=False)
unseen_data = unseen_data.apply(pd.to_numeric, errors='coerce').dropna()
unseen_grid_points = torch.tensor(unseen_data.values, dtype=torch.float32).to(vertices.device)

# Use KD-Tree for SDF values on unseen grid points
unseen_tree = cKDTree(unseen_grid_points.cpu().numpy())
unseen_sdf_values, _ = unseen_tree.query(unseen_grid_points.cpu().numpy())
unseen_sdf_values = torch.tensor(unseen_sdf_values, dtype=torch.float32).to(vertices.device)

# Autoencoder forward pass for unseen data
latent_space_unseen, _ = autoencoder(unseen_grid_points)

# Prepare input for MLP (latent space + unseen SDF)
mlp_input_unseen = torch.cat((latent_space_unseen, unseen_sdf_values.unsqueeze(1)), dim=1)

# Predict SDF for unseen data
predicted_sdf_unseen = mlp(mlp_input_unseen).squeeze().detach().cpu().numpy()

# Step 10: Save the predicted SDF values to a CSV file
predicted_sdf_df = pd.DataFrame({
    'X': unseen_grid_points[:, 0].cpu().numpy(),
    'Y': unseen_grid_points[:, 1].cpu().numpy(),
    'Z': unseen_grid_points[:, 2].cpu().numpy(),
    'Predicted_SDF': predicted_sdf_unseen
})

predicted_sdf_df.to_csv('predicted_sdf_unseen.csv', index=False)
print("Predicted SDF values have been saved to 'predicted_sdf_unseen.csv'")

# Optional: Save the model state
torch.save(autoencoder.state_dict(), 'autoencoder.pth')
torch.save(mlp.state_dict(), 'mlp.pth')
print("Models have been saved as 'autoencoder.pth' and 'mlp.pth'")
