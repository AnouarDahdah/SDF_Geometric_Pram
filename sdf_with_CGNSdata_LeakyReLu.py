# Install necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import h5py  # Make sure to import h5py to read CGNS files


# Step 1: Extract XYZ data from the CGNS file
def extract_xyz_data(file_path):
    with h5py.File(file_path, 'r') as cgns_file:
        # Define the paths to the required data
        X = "/Base/Air Body/GridCoordinates/CoordinateX/ data"
        Y = "/Base/Air Body/GridCoordinates/CoordinateY/ data"
        Z = "/Base/Air Body/GridCoordinates/CoordinateZ/ data"

        # Check if paths exist before accessing
        if X in cgns_file and Y in cgns_file and Z in cgns_file:
            # Extract data from the CGNS file
            X = np.array(cgns_file[X])
            Y = np.array(cgns_file[Y])
            Z = np.array(cgns_file[Z])
            return np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T  # Combine into a single array
        else:
            raise ValueError("One or more data paths do not exist in the CGNS file.")

# File path to your CGNS file
cgns_file_path = '/content/ID2_Surface.cgns'  # Update with your CGNS file path
vertices = extract_xyz_data(cgns_file_path)

# Print number of vertices
print(f"Number of vertices: {len(vertices)}")

# Step 2: Create a 3D grid for SDF initialization
grid_resolution = 100  # Resolution of the grid
x_min, x_max = vertices[:, 0].min() - 1, vertices[:, 0].max() + 1
y_min, y_max = vertices[:, 1].min() - 1, vertices[:, 1].max() + 1
z_min, z_max = vertices[:, 2].min() - 1, vertices[:, 2].max() + 1

# Create a 3D grid of points
x = np.linspace(x_min, x_max, grid_resolution)
y = np.linspace(y_min, y_max, grid_resolution)
z = np.linspace(z_min, z_max, grid_resolution)
xx, yy, zz = np.meshgrid(x, y, z)
grid_points = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T

# Step 3: Use KD-Tree for fast nearest neighbor search
# Create a KD-Tree for fast nearest neighbor search
tree = cKDTree(vertices)

# Step 4: Compute the SDF for each grid point
sdf_values = tree.query(grid_points)[0]  # Get distances to the nearest surface points
sdf_values = torch.tensor(sdf_values, dtype=torch.float32)  # Convert to PyTorch tensor

# Step 5: Define the Autoencoder model with more layers and sigmoid activation
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, hidden_dim5, latent_dim):
        super(Autoencoder, self).__init__()

        # Encoder with 6 layers (including the latent space)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),  # First hidden layer
            nn.LeakyReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),  # Second hidden layer
            nn.LeakyReLU(),
            nn.Linear(hidden_dim2, hidden_dim3),  # Third hidden layer
            nn.LeakyReLU(),
            nn.Linear(hidden_dim3, hidden_dim4),  # Fourth hidden layer
            nn.LeakyReLU(),
            nn.Linear(hidden_dim4, hidden_dim5),  # Fifth hidden layer
            nn.LeakyReLU(),
            nn.Linear(hidden_dim5, latent_dim)   # Latent space (sixth layer)
        )

        # Decoder with 6 layers (mirror image of encoder)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim5),  # First decoder layer (mirror of fifth encoder layer)
            nn.LeakyReLU(),
            nn.Linear(hidden_dim5, hidden_dim4),  # Second decoder layer
            nn.LeakyReLU(),
            nn.Linear(hidden_dim4, hidden_dim3),  # Third decoder layer
            nn.LeakyReLU(),
            nn.Linear(hidden_dim3, hidden_dim2),  # Fourth decoder layer
            nn.LeakyReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),  # Fifth decoder layer
            nn.LeakyReLU(),
            nn.Linear(hidden_dim1, input_dim)     # Output layer (same size as input: XYZ + SDF)
        )

    def forward(self, x):
        latent_space = self.encoder(x)  # Encode input to latent space
        reconstructed = self.decoder(latent_space)  # Decode latent space to reconstruct input
        return latent_space, reconstructed

# Initialize Autoencoder (input_dim=4 because XYZ + SDF)
autoencoder = Autoencoder(input_dim=4, hidden_dim1=128, hidden_dim2=64, hidden_dim3=32, hidden_dim4=16, hidden_dim5=8, latent_dim=8)

# Step 6: Define the MLP network for SDF prediction
class MLP_SDF_Predictor(nn.Module):
    def __init__(self, latent_dim, xyz_dim, hidden_dim, output_dim):
        super(MLP_SDF_Predictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim + xyz_dim, hidden_dim),  # Combine latent space + XYZ
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)  # Predict SDF value
        )

    def forward(self, latent_space, xyz):
        combined_input = torch.cat((latent_space, xyz), dim=1)
        return self.network(combined_input)

# Initialize MLP (latent_dim=8, xyz_dim=3 for XYZ, hidden_dim=16, output_dim=1 for SDF)
mlp_sdf_predictor = MLP_SDF_Predictor(latent_dim=8, xyz_dim=3, hidden_dim=16, output_dim=1)

# Step 7: Define the Deformation Field for data enrichment
class DeformationField(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DeformationField, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim)  # Output same dimension as input
        )

    def forward(self, x):
        return self.network(x)

# Initialize the deformation field
deformation_field = DeformationField(input_dim=3, hidden_dim=64)

# Step 8: Define the ODE function for the Neural ODE
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, t, h):
        return self.linear(h)

# Initialize the ODE function
ode_func = ODEFunc(hidden_dim=16)

# Step 9: Neural ODE solver
def neural_ode(h0, t):
    return odeint(ode_func, h0, t)

# Step 10: Set up the loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(list(autoencoder.parameters()) +
                             list(mlp_sdf_predictor.parameters()) +
                             list(deformation_field.parameters()) +
                             list(ode_func.parameters()), lr=0.0005)

# Number of training epochs and batch size
epochs = 36000
t = torch.linspace(0, 1, 50)
batch_size = 1024

# For plotting loss
loss_history = []

# Step 11: Training loop
for epoch in range(epochs):
    optimizer.zero_grad()

    # Randomly sample a subset of grid points for training
    indices = torch.randperm(grid_points.shape[0])[:batch_size]
    sampled_points = grid_points[indices]

    # Convert sampled_points to a PyTorch tensor
    sampled_points_tensor = torch.tensor(sampled_points, dtype=torch.float32)
    sampled_sdf_values = sdf_values[indices]

    # Concatenate XYZ + SDF as input to the autoencoder
    autoencoder_input = torch.cat((sampled_points_tensor, sampled_sdf_values.unsqueeze(1)), dim=1)

    # Pass through the autoencoder
    latent_space, reconstructed_input = autoencoder(autoencoder_input)

    # Extract XYZ data from the input for SDF prediction
    xyz_data = sampled_points_tensor  # Already a tensor now

    # Predict SDF using the MLP from latent space + XYZ data
    predicted_sdf = mlp_sdf_predictor(latent_space, xyz_data)

    # Reshape predicted_sdf to match the target size
    predicted_sdf = predicted_sdf.squeeze()  # Remove the extra dimension

    # Compute loss
    loss = loss_fn(predicted_sdf, sampled_sdf_values) + loss_fn(reconstructed_input, autoencoder_input)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Record loss for visualization
    loss_history.append(loss.item())

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

# Step 12: Plotting the loss history
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss/scratch/adahdah/siemens/Design_1')
plt.grid(True)
plt.show()
